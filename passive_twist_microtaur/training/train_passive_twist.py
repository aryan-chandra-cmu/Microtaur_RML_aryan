"""
train_passive_twist.py
Training script for Passive Twist MicroTaur.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import our environment
from envs.passive_twist_env import PassiveTwistEnv

class ActorCritic(nn.Module):
    """Simple Actor-Critic network."""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        # Shared feature extractor
        self.feature_layers = nn.ModuleList()
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic (value) head
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        
        action_mean = torch.tanh(self.actor_mean(features))
        value = self.critic(features)
        
        return action_mean, self.actor_logstd.expand_as(action_mean), value
    
    def get_action(self, x):
        with torch.no_grad():
            mean, logstd, _ = self.forward(x)
            std = torch.exp(logstd)
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            return torch.tanh(action)
    
    def evaluate_actions(self, x, actions):
        mean, logstd, values = self.forward(x)
        std = torch.exp(logstd)
        
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1, keepdim=True)
        
        # Apply tanh transformation for bounded actions
        transformed_actions = torch.tanh(actions)
        log_probs -= torch.sum(torch.log(1 - transformed_actions.pow(2) + 1e-6), dim=-1, keepdim=True)
        
        entropy = torch.sum(normal.entropy(), dim=-1, keepdim=True)
        
        return log_probs, entropy, values


class PPO:
    """Proximal Policy Optimization algorithm."""
    
    def __init__(self, 
                 actor_critic,
                 clip_param=0.2,
                 ppo_epoch=10,
                 num_mini_batch=64,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 lr=3e-4,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True):
        
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
    
    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)
            
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ = sample
                
                # Reshape
                obs_batch = obs_batch.view(-1, *obs_batch.shape[2:])
                actions_batch = actions_batch.view(-1, *actions_batch.shape[2:])
                value_preds_batch = value_preds_batch.view(-1, 1)
                return_batch = return_batch.view(-1, 1)
                old_action_log_probs_batch = old_action_log_probs_batch.view(-1, 1)
                adv_targ = adv_targ.view(-1, 1)
                
                # Evaluate actions
                action_log_probs, dist_entropy, value_pred = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)
                
                # Value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (value_pred - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_pred - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - value_pred).pow(2).mean()
                
                # Policy loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                
                # Total loss
                loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        num_updates = self.ppo_epoch * self.num_mini_batch
        
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class RolloutStorage:
    """Storage for rollout data."""
    
    def __init__(self, num_steps, num_processes, obs_shape, action_space, device):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_space.shape[0]).to(device)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)
        
        self.num_steps = num_steps
        self.step = 0
    
    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.observations[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        
        self.step = (self.step + 1) % self.num_steps
    
    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]
    
    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        
        assert batch_size >= num_mini_batch
        
        mini_batch_size = batch_size // num_mini_batch
        indices = np.random.permutation(batch_size)
        
        observations = self.observations[:-1].view(-1, *self.observations.size()[2:])
        actions = self.actions.view(-1, self.actions.size(-1))
        value_preds = self.value_preds[:-1].view(-1, 1)
        returns = self.returns[:-1].view(-1, 1)
        action_log_probs = self.action_log_probs.view(-1, 1)
        advantages = advantages.view(-1, 1)
        
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            
            obs_batch = observations[batch_indices]
            actions_batch = actions[batch_indices]
            value_preds_batch = value_preds[batch_indices]
            return_batch = returns[batch_indices]
            old_action_log_probs_batch = action_log_probs[batch_indices]
            adv_targ = advantages[batch_indices]
            
            yield obs_batch, actions_batch, value_preds_batch, return_batch, \
                old_action_log_probs_batch, adv_targ


class PassiveTwistTrainer:
    """Main trainer class."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Create run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"passive_twist_{timestamp}"
        self.base_dir = Path(f"runs/{self.run_name}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.base_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config['training'].get('use_gpu', True) 
            else "cpu"
        )
        
        print(f"Using device: {self.device}")
        print(f"Run directory: {self.base_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'training': {
                'total_timesteps': 1000000,
                'num_processes': 1,
                'num_steps': 2048,
                'ppo_epoch': 10,
                'num_mini_batch': 64,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_param': 0.2,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5,
                'use_clipped_value_loss': True,
                'save_freq': 10000,
                'eval_freq': 5000,
                'eval_episodes': 5,
                'use_gpu': True
            },
            'env': {
                'xml_path': 'scene.xml',
                'control_frequency': 50,
                'episode_duration': 10.0,
                'use_viewer': False,
                'render_during_eval': False
            },
            'policy': {
                'hidden_dims': [256, 256],
                'activation': 'relu'
            }
        }
    
    def train(self):
        """Main training loop."""
        print("=" * 80)
        print(f"Starting training: {self.run_name}")
        print("=" * 80)
        
        # Create environment - will auto-find scene.xml
        env = PassiveTwistEnv(
            xml_path=None,  # Auto-find
            render_mode=None,
            control_frequency=self.config['env']['control_frequency'],
            episode_duration=self.config['env']['episode_duration'],
            use_viewer=False
        )
        
        # Get dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"Observation dimension: {obs_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Create actor-critic
        actor_critic = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config['policy']['hidden_dims']
        ).to(self.device)
        
        # Create PPO agent
        agent = PPO(
            actor_critic=actor_critic,
            clip_param=self.config['training']['clip_param'],
            ppo_epoch=self.config['training']['ppo_epoch'],
            num_mini_batch=self.config['training']['num_mini_batch'],
            value_loss_coef=self.config['training']['value_loss_coef'],
            entropy_coef=self.config['training']['entropy_coef'],
            lr=self.config['training']['learning_rate'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            use_clipped_value_loss=self.config['training']['use_clipped_value_loss']
        )
        
        # Create rollout storage
        rollouts = RolloutStorage(
            num_steps=self.config['training']['num_steps'],
            num_processes=self.config['training']['num_processes'],
            obs_shape=(obs_dim,),
            action_space=env.action_space,
            device=self.device
        )
        
        # Initialize variables
        current_obs, _ = env.reset()
        current_obs = torch.FloatTensor(current_obs).to(self.device).unsqueeze(0)
        rollouts.observations[0].copy_(current_obs)
        
        total_timesteps = self.config['training']['total_timesteps']
        num_timesteps = 0
        num_episodes = 0
        episode_rewards = []
        
        print(f"\nStarting training for {total_timesteps} timesteps...")
        
        while num_timesteps < total_timesteps:
            # Collect rollouts
            for step in range(self.config['training']['num_steps']):
                with torch.no_grad():
                    value, action, action_log_prob = actor_critic(
                        rollouts.observations[step])
                
                # Step environment
                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                
                # Convert to tensors
                next_obs = torch.FloatTensor(next_obs).to(self.device).unsqueeze(0)
                reward = torch.FloatTensor([reward]).to(self.device).unsqueeze(0)
                done = torch.FloatTensor([terminated or truncated]).to(self.device).unsqueeze(0)
                masks = torch.FloatTensor([0.0 if terminated else 1.0]).to(self.device).unsqueeze(0)
                
                # Record episode reward
                if terminated or truncated:
                    episode_rewards.append(info.get('episode_reward', 0))
                    num_episodes += 1
                    next_obs, _ = env.reset()
                    next_obs = torch.FloatTensor(next_obs).to(self.device).unsqueeze(0)
                
                # Insert into rollout buffer
                rollouts.insert(
                    next_obs, action, action_log_prob, value, reward, masks)
                
                num_timesteps += 1
            
            # Compute returns
            with torch.no_grad():
                next_value = actor_critic(rollouts.observations[-1])[2]
            
            rollouts.compute_returns(
                next_value, 
                self.config['training']['gamma'],
                self.config['training']['gae_lambda']
            )
            
            # Update policy
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            
            # After update, reset storage
            rollouts.observations[0].copy_(rollouts.observations[-1])
            
            # Print progress
            if num_episodes > 0 and num_episodes % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"Episode {num_episodes}, Timestep {num_timesteps}/{total_timesteps}")
                print(f"  Avg Reward (last 10): {avg_reward:.2f}")
                print(f"  Losses: Value={value_loss:.4f}, Policy={action_loss:.4f}, Entropy={dist_entropy:.4f}")
            
            # Evaluate
            if num_timesteps % self.config['training']['eval_freq'] == 0:
                eval_reward = self._evaluate_policy(actor_critic, env)
                print(f"\nEvaluation at {num_timesteps:,} timesteps:")
                print(f"  Average reward: {eval_reward:.2f}")
            
            # Save checkpoint
            if num_timesteps % self.config['training']['save_freq'] == 0:
                self._save_checkpoint(actor_critic, agent, num_timesteps)
        
        # Final save
        self._save_checkpoint(actor_critic, agent, num_timesteps, final=True)
        env.close()
        
        print(f"\nTraining completed!")
        print(f"Total episodes: {num_episodes}")
        print(f"Total timesteps: {num_timesteps}")
    
    def _evaluate_policy(self, actor_critic, training_env, num_episodes: int = 5) -> float:
        """Evaluate policy without exploration noise."""
        eval_env = PassiveTwistEnv(
            xml_path=self.config['env']['xml_path'],
            render_mode='human' if self.config['env']['render_during_eval'] else None,
            control_frequency=self.config['env']['control_frequency'],
            episode_duration=self.config['env']['episode_duration'],
            use_viewer=False
        )
        
        total_rewards = []
        
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    action = actor_critic.get_action(obs_tensor)
                    action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        eval_env.close()
        return np.mean(total_rewards)
    
    def _save_checkpoint(self, actor_critic, agent, timestep: int, final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = self.base_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        if final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_{timestep:07d}.pt"
        
        checkpoint = {
            'timestep': timestep,
            'actor_critic_state_dict': actor_critic.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"Saved checkpoint: {filename}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Passive Twist MicroTaur")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--model", type=str, help="Model to load for evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        # Run evaluation
        if not args.model:
            print("Error: --model required for evaluation")
            return
        
        # Load model and run evaluation
        checkpoint = torch.load(args.model)
        config = checkpoint['config']
        
        # Create environment
        env = PassiveTwistEnv(
            xml_path=config['env']['xml_path'],
            render_mode='human',
            control_frequency=config['env']['control_frequency'],
            episode_duration=config['env']['episode_duration'],
            use_viewer=False
        )
        
        # Create actor-critic
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        actor_critic = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config['policy']['hidden_dims']
        )
        
        # Load weights
        actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        actor_critic.eval()
        
        # Run evaluation
        trainer = PassiveTwistTrainer(args.config)
        reward = trainer._evaluate_policy(actor_critic, env, num_episodes=5)
        print(f"Evaluation average reward: {reward:.2f}")
        
        env.close()
    else:
        # Create trainer
        trainer = PassiveTwistTrainer(config_path=args.config)
        
        # Override timesteps if specified
        if args.timesteps:
            trainer.config['training']['total_timesteps'] = args.timesteps
        
        # Start training
        trainer.train()


if __name__ == "__main__":
    main()