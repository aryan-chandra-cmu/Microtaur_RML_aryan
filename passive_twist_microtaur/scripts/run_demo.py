#!/usr/bin/env python3
"""
run_demo.py - Run a trained policy
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from envs.passive_twist_microtaur_env import PassiveTwistMicrotaurEnv

def run_demo(checkpoint_path: str, num_episodes: int = 3):
    """Run a trained policy."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create environment with rendering
    env = PassiveTwistMicrotaurEnv(
        xml_path='assets/passive_twist_microtaur_ptm.xml',
        render_mode='human',
        control_frequency=50,
        episode_duration=10.0
    )
    
    # Simple policy for demo
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, action_dim),
        torch.nn.Tanh()
    )
    
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    
    policy.eval()
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor).numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}: Reward = {total_reward:.2f}")
        
        print(f"Episode completed: {step} steps, total reward = {total_reward:.2f}")
    
    env.close()
    print("\nDemo completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run policy demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        # Try to find in runs directory
        possible_path = f"runs/{args.checkpoint}/checkpoints/final_model.pt"
        if os.path.exists(possible_path):
            args.checkpoint = possible_path
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    
    run_demo(args.checkpoint, args.episodes)