"""
train.py - Robust PPO training script for Passive Twist MicroTaur
Drop-in replacement for your current train.py

Key fixes vs your version:
- Correct PPO: stores old log-probs from rollout policy
- Bootstrapped GAE using last_value
- Minibatch PPO updates + sane ppo_epochs
- Separate terminated vs truncated handling (bootstrap through truncation)
- Optional running observation normalization (enabled by default)
- Proper checkpointing, seeding, device handling

Usage:
  python train.py

Assumes:
  from envs.passive_twist_env import PassiveTwistEnv
  scene.xml lives at repo root (../scene.xml from this script)
"""

import os
import sys
import time
import datetime
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml

# Ensure repo root is on path (same as your original style)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.passive_twist_env import PassiveTwistEnv


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # More deterministic (can reduce throughput a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunningMeanStd:
    """Tracks running mean/var for normalization (Welford-style)."""

    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # x: [B, *shape]
        x = x.to(self.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=self.device)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor, clip=10.0):
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(x, -clip, clip)


# -------------------------
# Policy
# -------------------------

class PPOPolicy(nn.Module):
    """
    Gaussian policy + critic.
    IMPORTANT: actor output is UNSQUASHED mean. We sample Normal and then clip actions
    to env bounds (env also clips). This avoids tanh-squash logprob correction complexity.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(obs)
        return mean, std, value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)


# -------------------------
# PPO Agent
# -------------------------

class PPOAgent:
    def __init__(self, env, config):
        self.env = env
        self.cfg = config

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.device = torch.device("cuda" if (torch.cuda.is_available() and self.cfg["device"] == "cuda") else "cpu")

        self.policy = PPOPolicy(self.obs_dim, self.action_dim, self.cfg["policy"]["hidden_dim"]).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg["training"]["learning_rate"])

        # Optional obs normalization
        self.use_obs_norm = self.cfg["training"].get("use_obs_norm", True)
        self.obs_rms = RunningMeanStd(self.obs_dim, device=self.device) if self.use_obs_norm else None

        # Buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminals = []   # terminated (true terminal)
        self.truncs = []      # truncated (time-limit)

        # Logging dirs
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"passive_twist_{timestamp}"
        self.log_dir = Path(f"runs/{self.run_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))

        # Save config
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Action bounds
        self.act_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=self.device)

    def _prep_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, obs_dim]
        if self.use_obs_norm:
            # update running stats with single obs (still helps, better with batches)
            self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)
        return obs

    def _prep_obs_batch(self, obs_np: np.ndarray) -> torch.Tensor:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        if self.use_obs_norm:
            self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)
        return obs

    def collect_rollout(self):
        """Collect rollout_length transitions; return episode_reward_sum and bootstrapped last_value."""
        obs, _ = self.env.reset()
        ep_reward = 0.0

        rollout_len = self.cfg["training"]["rollout_length"]

        for _ in range(rollout_len):
            obs_t = self._prep_obs(obs)

            action_t, logp_t, value_t = self.policy.act(obs_t, deterministic=False)

            # Clip action to env bounds (robust + simple)
            action_t = torch.max(torch.min(action_t, self.act_high), self.act_low)

            action = action_t.squeeze(0).cpu().numpy()
            logp = logp_t.item()
            value = value_t.item()

            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            self.states.append(obs)
            self.actions.append(action)
            self.log_probs.append(logp)
            self.rewards.append(float(reward))
            self.values.append(float(value))
            self.terminals.append(bool(terminated))
            self.truncs.append(bool(truncated))

            obs = next_obs
            ep_reward += reward

            if terminated or truncated:
                obs, _ = self.env.reset()

        # Bootstrap last value from final obs
        obs_t = self._prep_obs(obs)
        with torch.no_grad():
            _, _, last_value = self.policy.forward(obs_t)
        last_value = last_value.squeeze(-1).item()

        return ep_reward, last_value

    def compute_gae(self, last_value: float):
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        terminals = np.array(self.terminals, dtype=np.float32)  # 1 if terminated
        # IMPORTANT: bootstrap through truncations; so only terminals stop bootstrap
        not_terminal = 1.0 - terminals

        gamma = self.cfg["training"]["gamma"]
        lam = self.cfg["training"]["gae_lambda"]

        adv = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * not_terminal[t] - values[t]
            last_adv = delta + gamma * lam * not_terminal[t] * last_adv
            adv[t] = last_adv

        returns = adv + values[:-1]
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def update(self, last_value: float):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        advantages, returns = self.compute_gae(last_value)

        # Convert to tensors
        states_t = self._prep_obs_batch(states)  # normalized + updates stats
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        batch_size = states_t.shape[0]
        mb_size = self.cfg["training"]["minibatch_size"]
        ppo_epochs = self.cfg["training"]["ppo_epochs"]

        clip_range = self.cfg["training"]["clip_range"]
        value_coef = self.cfg["training"]["value_coef"]
        entropy_coef = self.cfg["training"]["entropy_coef"]
        max_grad_norm = self.cfg["training"]["max_grad_norm"]

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        approx_kl = 0.0
        clip_frac = 0.0

        for _ in range(ppo_epochs):
            idx = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mb_size):
                mb_idx = idx[start:start + mb_size]

                mean, std, values = self.policy(states_t[mb_idx])
                values = values.squeeze(-1)

                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(actions_t[mb_idx]).sum(dim=-1)

                ratio = torch.exp(new_logp - old_logp_t[mb_idx])

                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values, ret_t[mb_idx])

                entropy = dist.entropy().sum(dim=-1).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = (old_logp_t[mb_idx] - new_logp).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > clip_range).float().mean().item()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy.item()

            # Optional early stop on KL
            target_kl = self.cfg["training"].get("target_kl", None)
            if target_kl is not None and approx_kl > target_kl:
                break

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.terminals.clear()
        self.truncs.clear()

        return last_policy_loss, last_value_loss, last_entropy, approx_kl, clip_frac

    def save_checkpoint(self, timestep: int, final: bool = False):
        ckpt_dir = self.log_dir / "checkpoints"
        filename = "final_model.pt" if final else f"checkpoint_{timestep:07d}.pt"

        payload = {
            "timestep": timestep,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        if self.use_obs_norm:
            payload["obs_rms"] = {
                "mean": self.obs_rms.mean.detach().cpu(),
                "var": self.obs_rms.var.detach().cpu(),
                "count": self.obs_rms.count.detach().cpu(),
            }

        torch.save(payload, ckpt_dir / filename)
        print(f"Saved checkpoint: {filename}")

    def train(self):
        print(f"Starting training: {self.run_name}")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")

        total_timesteps = self.cfg["training"]["total_timesteps"]
        rollout_len = self.cfg["training"]["rollout_length"]
        save_freq = self.cfg["training"]["save_freq"]

        timestep = 0
        episode = 0
        reward_hist = deque(maxlen=100)

        start_time = time.time()

        while timestep < total_timesteps:
            ep_reward, last_value = self.collect_rollout()
            timestep += rollout_len
            episode += 1

            reward_hist.append(ep_reward)

            pol_loss, val_loss, ent, kl, clip_frac = self.update(last_value)
            avg_reward = float(np.mean(reward_hist)) if reward_hist else 0.0

            # Logging
            self.writer.add_scalar("Reward/episode_rollout_sum", ep_reward, timestep)
            self.writer.add_scalar("Reward/avg_100", avg_reward, timestep)
            self.writer.add_scalar("Loss/policy", pol_loss, timestep)
            self.writer.add_scalar("Loss/value", val_loss, timestep)
            self.writer.add_scalar("Loss/entropy", ent, timestep)
            self.writer.add_scalar("Diag/approx_kl", kl, timestep)
            self.writer.add_scalar("Diag/clip_frac", clip_frac, timestep)

            # Print progress
            if episode % self.cfg["training"]["print_every"] == 0:
                elapsed = time.time() - start_time
                fps = timestep / max(elapsed, 1e-6)
                print(f"[Ep {episode:5d}] t={timestep:8d}/{total_timesteps} "
                      f"R={ep_reward:8.2f} Avg100={avg_reward:8.2f} "
                      f"PL={pol_loss:7.4f} VL={val_loss:7.4f} Ent={ent:7.4f} "
                      f"KL={kl:7.4f} Clip={clip_frac:5.2f} FPS={fps:7.1f}")

            # Save
            if timestep % save_freq == 0:
                self.save_checkpoint(timestep)

        self.save_checkpoint(timestep, final=True)
        self.writer.close()
        print("Training completed!")


# -------------------------
# Config
# -------------------------

def get_config():
    return {
        "device": "cuda",  # uses cuda if available, else cpu
        "seed": 42,
        "training": {
            "total_timesteps": 1_500_000,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,

            "rollout_length": 2048,
            "ppo_epochs": 10,          # sane default
            "minibatch_size": 256,

            "target_kl": 0.02,         # early stop if KL too high (optional)
            "use_obs_norm": True,

            "save_freq": 50_000,
            "print_every": 10,
        },
        "policy": {
            "hidden_dim": 256
        }
    }


# -------------------------
# Main
# -------------------------

def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    # Resolve scene.xml relative to this script
    ROOT_DIR = Path(__file__).resolve().parent.parent  # .../passive_twist_microtaur
    XML_PATH = ROOT_DIR / "scene.xml"

    env = PassiveTwistEnv(
        xml_path=str(XML_PATH),
        render_mode=None,
        control_frequency=50,
        episode_duration=10.0,
        use_viewer=False
    )

    agent = PPOAgent(env, cfg)

    try:
        agent.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save an interrupt checkpoint
        agent.save_checkpoint(int(cfg["training"]["rollout_length"]), final=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
