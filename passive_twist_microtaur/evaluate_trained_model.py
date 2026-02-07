"""
evaluate_trained_model.py
Evaluate a model trained with train.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.passive_twist_env import PassiveTwistEnv


class PPOPolicy(torch.nn.Module):
    """Same policy network as in train.py"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor network
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()
        )
        
        # Critic network
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        # Log std (learnable parameter)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        """Forward pass."""
        action_mean = self.actor(x)
        action_std = torch.exp(self.log_std)
        value = self.critic(x)
        return action_mean, action_std, value
    
    def get_action(self, x, deterministic=True):
        """Get action from observation."""
        with torch.no_grad():
            action_mean, action_std, value = self.forward(x)
            
            if deterministic:
                action = action_mean
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
            
            # Clip action
            action = torch.tanh(action)
            
            return action.cpu().numpy(), value.cpu().numpy()


def load_model(model_path, obs_dim, action_dim):
    """Load a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create policy with same architecture
    policy = PPOPolicy(obs_dim, action_dim)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Trained for {checkpoint['timestep']} timesteps")
    
    return policy, device


def evaluate_model(model_path, num_episodes=5, render=True):
    """Evaluate a trained model."""
    
    # Create environment
    env = PassiveTwistEnv(
        xml_path='scene.xml',
        render_mode='human' if render else None,
        control_frequency=50,
        episode_duration=10.0,
        use_viewer=False
    )
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Try to find the model file
    model_file = Path(model_path)
    
    # Check various possible locations
    possible_paths = [
        model_path,  # Original path
        f"runs/{model_path}",  # With runs prefix
        f"runs/{model_path}/checkpoints/final_model.pt",  # Full path
        f"{model_path}/checkpoints/final_model.pt",  # Without runs
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if not found_path:
        # Search for any .pt files
        print(f"Model not found at specified path. Searching for .pt files...")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".pt"):
                    full_path = os.path.join(root, file)
                    print(f"Found: {full_path}")
                    if "final" in file or "checkpoint" in file:
                        found_path = full_path
                        break
            if found_path:
                break
    
    if not found_path:
        print(f"Error: Could not find any model files!")
        return
    
    print(f"Using model: {found_path}")
    
    # Load model
    policy, device = load_model(found_path, obs_dim, action_dim)
    policy.to(device)
    
    # Run evaluation
    total_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action
            obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
            action, _ = policy.get_action(obs_tensor, deterministic=True)
            action = action[0]  # Remove batch dimension
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.0f} steps")
    print(f"{'='*60}")
    
    return total_rewards


def list_available_models():
    """List all available trained models."""
    print("Searching for trained models...")
    print("-" * 60)
    
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("No 'runs' directory found!")
        return
    
    models_found = []
    
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir():
            checkpoints_dir = run_dir / "checkpoints"
            if checkpoints_dir.exists():
                # Find the most recent checkpoint
                checkpoint_files = list(checkpoints_dir.glob("*.pt"))
                if checkpoint_files:
                    # Sort by modification time
                    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    latest_checkpoint = checkpoint_files[0]
                    
                    # Load checkpoint info
                    try:
                        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                        timestep = checkpoint.get('timestep', 'unknown')
                        models_found.append({
                            'path': str(latest_checkpoint),
                            'name': run_dir.name,
                            'timestep': timestep,
                            'date': time.ctime(os.path.getmtime(latest_checkpoint))
                        })
                    except:
                        models_found.append({
                            'path': str(latest_checkpoint),
                            'name': run_dir.name,
                            'timestep': 'unknown',
                            'date': time.ctime(os.path.getmtime(latest_checkpoint))
                        })
    
    if models_found:
        print(f"Found {len(models_found)} trained models:")
        print("-" * 60)
        for i, model in enumerate(models_found, 1):
            print(f"{i}. {model['name']}")
            print(f"   Path: {model['path']}")
            print(f"   Timesteps: {model['timestep']}")
            print(f"   Modified: {model['date']}")
            print()
    else:
        print("No trained models found in runs/ directory!")
    
    return models_found


def test_specific_command(model_path, command, duration=10.0, render=True):
    """Test the model with a specific velocity command."""
    import time
    
    # Create environment
    env = PassiveTwistEnv(
        xml_path='scene.xml',
        render_mode='human' if render else None,
        control_frequency=50,
        episode_duration=duration,
        use_viewer=False
    )
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load model
    policy, device = load_model(model_path, obs_dim, action_dim)
    policy.to(device)
    
    # Reset environment and set command
    obs, _ = env.reset()
    env.set_command(command)
    
    print(f"\nTesting with command: {command}")
    print(f"Duration: {duration} seconds")
    print("-" * 40)
    
    total_reward = 0
    start_time = time.time()
    
    try:
        while True:
            # Get action
            obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
            action, _ = policy.get_action(obs_tensor, deterministic=True)
            action = action[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Check termination
            if terminated or truncated:
                break
            
            # Check time limit
            if time.time() - start_time > duration + 2:  # Add buffer
                break
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    env.close()
    
    elapsed = time.time() - start_time
    print(f"\nTest completed:")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward/sec: {total_reward/elapsed:.2f}")


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Evaluate a trained Passive Twist MicroTaur model")
    parser.add_argument("--model", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--test-command", type=float, nargs=3, metavar=('FORWARD', 'LATERAL', 'YAW'),
                       help="Test with specific command [forward lateral yaw]")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.model:
        if args.test_command:
            # Test with specific command
            command = np.array(args.test_command, dtype=np.float32)
            test_specific_command(args.model, command, args.duration, not args.no_render)
        else:
            # Standard evaluation
            evaluate_model(
                model_path=args.model,
                num_episodes=args.episodes,
                render=not args.no_render
            )
    else:
        print("Error: Please provide a model path or use --list")
        print("\nUsage examples:")
        print("  python evaluate_trained_model.py --list")
        print("  python evaluate_trained_model.py --model runs/passive_twist_20260129_212548/checkpoints/final_model.pt")
        print("  python evaluate_trained_model.py --model runs/passive_twist_20260129_212548/checkpoints/final_model.pt --episodes 10")
        print("  python evaluate_trained_model.py --model runs/passive_twist_20260129_212548/checkpoints/final_model.pt --test-command 0.2 0.0 0.0")
        print("\nNote: If you don't know the exact path, use --list to see available models")