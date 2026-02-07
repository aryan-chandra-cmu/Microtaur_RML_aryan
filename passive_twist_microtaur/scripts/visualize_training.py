"""
visualize_training.py
Visualize the trained robot walking
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import mujoco.viewer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.passive_twist_env import PassiveTwistEnv


class PPOPolicy(nn.Module):
    """PPO policy network - MUST match the architecture from train.py"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Log std (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
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


class SimplePolicy:
    """Policy wrapper for visualization"""
    
    def __init__(self, obs_dim, action_dim, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use the same architecture as in training
        self.model = PPOPolicy(obs_dim, action_dim).to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Try to load the policy state dict
                if 'policy_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['policy_state_dict'])
                    print(f"Loaded policy from {model_path}")
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                    print(f"Loaded model from {model_path}")
                else:
                    # Try to load the entire checkpoint as state dict
                    self.model.load_state_dict(checkpoint)
                    print(f"Loaded entire checkpoint as state dict from {model_path}")
                
                print(f"Model loaded successfully")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random policy instead")
        else:
            print("No model provided or model not found. Using random policy.")
        
        self.model.eval()
    
    def get_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.model.get_action(obs_tensor, deterministic=True)
        return action[0]  # Remove batch dimension


def find_latest_model():
    """Find the most recent trained model"""
    runs_dir = "../runs"  # Go up from scripts to project root
    if not os.path.exists(runs_dir):
        print("No runs directory found!")
        return None
    
    # List all run directories
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not run_dirs:
        print("No trained runs found!")
        return None
    
    # Sort by creation time (newest first)
    run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(runs_dir, x)), reverse=True)
    
    # Look for checkpoints in the newest run
    latest_run = run_dirs[0]
    checkpoints_dir = os.path.join(runs_dir, latest_run, "checkpoints")
    
    if os.path.exists(checkpoints_dir):
        # Look for final model first, then any checkpoint
        final_model = os.path.join(checkpoints_dir, "final_model.pt")
        if os.path.exists(final_model):
            return final_model
        
        # Otherwise get the most recent checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(reverse=True)
            return os.path.join(checkpoints_dir, checkpoint_files[0])
    
    print(f"No checkpoints found in {latest_run}")
    return None


def visualize_robot(model_path=None, command=None, duration=30):
    """Visualize the robot walking"""
    
    # Get the absolute path to scene.xml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, 'scene.xml')
    
    print(f"Using XML path: {xml_path}")
    
    # Create environment WITH RENDERING
    env = PassiveTwistEnv(
        xml_path=xml_path,
        render_mode='human',
        control_frequency=50,
        episode_duration=duration,
        use_viewer=True
    )
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    policy = SimplePolicy(obs_dim, action_dim, model_path)
    
    # Set command
    if command is None:
        # Start with standing still to see if robot can balance
        command = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    env.set_command(command)
    
    print("\n" + "="*60)
    print("VISUALIZING ROBOT WALKING")
    print("="*60)
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Command: {command}")
    print(f"Duration: {duration} seconds")
    print("="*60)
    print("\nControls in viewer:")
    print("  - ESC: Exit viewer")
    print("  - SPACE: Pause/Resume simulation")
    print("  - ↑/↓/←/→: Move camera")
    print("  - Mouse drag: Rotate camera")
    print("  - Scroll: Zoom in/out")
    print("\n")
    
    # Reset environment
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    try:
        start_time = time.time()
        
        while True:
            # Get action
            action = policy.get_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Time: {elapsed:.1f}s, Steps: {step_count}, Reward: {total_reward:.2f}")
            
            # Check termination
            if terminated or truncated:
                print(f"\nEpisode ended!")
                print(f"Total steps: {step_count}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Base position: {info.get('base_position', 'N/A')}")
                
                # Ask if user wants to continue
                response = input("\nRun again? (y/n): ").lower()
                if response != 'y':
                    break
                
                # Reset for another run
                obs, _ = env.reset()
                total_reward = 0
                step_count = 0
                start_time = time.time()
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    finally:
        env.close()
        print("\nVisualization ended")


def interactive_mode(model_path):
    """Interactive mode to test different commands"""
    if not model_path:
        print("No model provided for interactive mode")
        return
    
    # Get the absolute path to scene.xml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_root, 'scene.xml')
    
    # Create environment
    env = PassiveTwistEnv(
        xml_path=xml_path,
        render_mode='human',
        control_frequency=50,
        episode_duration=1500.0,
        use_viewer=True
    )
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    policy = SimplePolicy(obs_dim, action_dim, model_path)
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  s: Stand still [0, 0, 0]")
    print("  f: Forward [0.2, 0, 0]")
    print("  b: Backward [-0.2, 0, 0]")
    print("  r: Right [0, 0.1, 0]")
    print("  l: Left [0, -0.1, 0]")
    print("  tr: Turn right [0, 0, 0.3]")
    print("  tl: Turn left [0, 0, -0.3]")
    print("  c: Custom command")
    print("  q: Quit")
    print("="*60)
    
    commands = {
        's': [0.0, 0.0, 0.0],
        'f': [0.05, 0.0, 0.0],
        'b': [-0.05, 0.0, 0.0],
        'r': [0.0, 0.01, 0.0],
        'l': [0.0, -0.01, 0.0],
        'tr': [0.0, 0.0, 0.3],
        'tl': [0.0, 0.0, -0.3],
    }
    
    # Initial reset
    obs, _ = env.reset()
    env.set_command([0.0, 0.0, 0.0])
    
    try:
        while True:
            cmd = input("\nEnter command: ").lower().strip()
            
            if cmd == 'q':
                break
            elif cmd == 'c':
                try:
                    fwd = float(input("Forward velocity: "))
                    lat = float(input("Lateral velocity: "))
                    yaw = float(input("Yaw rate: "))
                    command = np.array([fwd, lat, yaw], dtype=np.float32)
                    env.set_command(command)
                    print(f"Set command to: {command}")
                except:
                    print("Invalid input")
                    continue
            elif cmd in commands:
                command = np.array(commands[cmd], dtype=np.float32)
                env.set_command(command)
                print(f"Set command to: {command}")
            else:
                print("Unknown command")
                continue
            
            # Run for 5 seconds with this command
            print(f"Running with command {command} for 5 seconds...")
            start_time = time.time()
            
            while time.time() - start_time < 5:
                action = policy.get_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                
                if terminated or truncated:
                    print("Robot fell! Resetting...")
                    obs, _ = env.reset()
                    env.set_command(command)
                    break
            
            # Check if user wants to continue
            if cmd != 'q':
                continue
    
    except KeyboardInterrupt:
        print("\n\nInteractive mode interrupted")
    
    env.close()
    print("Interactive mode ended")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize robot walking")
    parser.add_argument("--model", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--auto-find", action="store_true", help="Automatically find latest model")
    parser.add_argument("--command", type=float, nargs=3, metavar=('FORWARD', 'LATERAL', 'YAW'),
                       help="Velocity command [forward lateral yaw]")
    parser.add_argument("--duration", type=float, default=30, help="Visualization duration in seconds")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Find model
    model_path = None
    if args.model:
        model_path = args.model
    elif args.auto_find:
        model_path = find_latest_model()
        if model_path:
            print(f"Found model: {model_path}")
        else:
            print("No model found. Using random policy.")
    else:
        # Ask user
        print("No model specified. Options:")
        print("  1. Use random policy")
        print("  2. Find latest trained model")
        print("  3. Specify model path")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '2':
            model_path = find_latest_model()
            if not model_path:
                print("No model found. Using random policy.")
        elif choice == '3':
            model_path = input("Enter model path: ").strip()
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Using random policy.")
                model_path = None
    
    # Set command
    command = None
    if args.command:
        command = np.array(args.command, dtype=np.float32)
    
    # Run visualization
    if args.interactive:
        interactive_mode(model_path)
    else:
        visualize_robot(model_path, command, args.duration)






        #python visualize_training.py --model ../runs/passive_twist_20260129_212548/checkpoints/final_model.pt