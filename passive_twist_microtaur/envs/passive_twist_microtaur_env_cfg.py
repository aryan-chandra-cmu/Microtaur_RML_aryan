"""
passive_twist_microtaur_env.py
Complete RL environment for Passive Twist MicroTaur robot.
"""

import os
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import math
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class PassiveTwistMicrotaurEnv(gym.Env):
    """RL Environment for Passive Twist MicroTaur quadruped robot."""
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }
    
    def __init__(self, 
                 xml_path: str = 'scene.xml',
                 render_mode: Optional[str] = None,
                 control_frequency: int = 50,
                 episode_duration: float = 10.0,
                 use_viewer: bool = False):
        """
        Initialize the environment.
        
        Args:
            xml_path: Path to MuJoCo XML file
            render_mode: 'human', 'rgb_array', or None
            control_frequency: Control frequency in Hz
            episode_duration: Episode duration in seconds
            use_viewer: Use mujoco.viewer for visualization
        """
        super().__init__()
        
        # Store parameters
        self.xml_path = xml_path
        self.control_frequency = control_frequency
        self.episode_duration = episode_duration
        self.max_steps = int(episode_duration * control_frequency)
        self.render_mode = render_mode
        self.use_viewer = use_viewer
        
        # Check if XML exists
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        
        # ====================
        # 1. MuJoCo Setup
        # ====================
        print(f"Loading model from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set simulation timestep
        self.model.opt.timestep = 1.0 / (control_frequency * 10)
        
        # ====================
        # 2. Setup Data Structures
        # ====================
        self._setup_mujoco_data()
        
        # ====================
        # 3. Action/Observation Spaces
        # ====================
        self._define_action_space()
        self._define_observation_space()
        
        # ====================
        # 4. Initialize Variables
        # ====================
        self.current_step = 0
        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [forward, lateral, yaw]
        self.phase = 0.0
        self.phase_rate = 2.0 * np.pi / (control_frequency * 0.5)
        
        # For reward calculation
        self.previous_action = np.zeros(self.action_space.shape[0])
        self.total_energy = 0.0
        self.base_initial_position = None
        
        # Contact detection
        self.contact_threshold = 0.05  # 5cm
        
        # ====================
        # 5. Initialize Renderer
        # ====================
        self.viewer = None
        if render_mode == 'human' and use_viewer:
            # Viewer will be created in reset()
            pass
        
        print(f"Environment initialized: {self.observation_space.shape} obs, {self.action_space.shape} actions")
        print(f"Number of actuators: {self.model.nu}")
        print(f"Number of joints: {self.model.njnt}")
    
    def _setup_mujoco_data(self):
        """Extract important MuJoCo IDs and setup data structures."""
        self.actuator_ids = list(range(self.model.nu))
        self.joint_ids = list(range(self.model.njnt))
        
        # Try to find base body
        self.base_body_id = 1  # Default to first body after world
        
        # Look for specific body names
        body_names = [
            'rotating_microtaur_top_half___free_joint__v2_ptm',
            'rotating_microtaur_bottom_half_motor_mount___free_pin__v2_ptm',
            'base',
            'torso'
        ]
        
        for name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                self.base_body_id = body_id
                print(f"Found base body: {name} (id: {body_id})")
                break
        
        # Find foot sites
        self.foot_site_ids = []
        foot_patterns = ['foot', 'leg', 'site']
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name:
                if any(pattern in site_name.lower() for pattern in foot_patterns):
                    self.foot_site_ids.append(i)
        
        print(f"Found {len(self.foot_site_ids)} foot sites")
    
    def _define_action_space(self):
        """Define the action space for the robot."""
        # Conservative joint limits
        action_dim = self.model.nu
        self.action_space = spaces.Box(
            low=np.full(action_dim, -1.0, dtype=np.float32),
            high=np.full(action_dim, 1.0, dtype=np.float32),
            dtype=np.float32
        )
    
    def _define_observation_space(self):
        """Define the observation space."""
        # Observation components:
        # 1. Joint positions (nq)
        # 2. Joint velocities (nv)
        # 3. Base orientation (4)
        # 4. Base angular velocity (3)
        # 5. Base linear velocity (3)
        # 6. Foot contacts (num_feet)
        # 7. Previous action (action_dim)
        # 8. Command (3)
        # 9. Phase (2)
        
        obs_dim = (
            self.model.nq +  # Joint positions
            self.model.nv +  # Joint velocities
            4 + 3 + 3 +      # Base state
            len(self.foot_site_ids) +  # Foot contacts
            self.model.nu +  # Previous action
            3 + 2            # Command + phase
        )
        
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply randomization
        self._randomize_initial_conditions(options)
        
        # Reset variables
        self.current_step = 0
        self.phase = 0.0
        self.total_energy = 0.0
        
        # Store initial base position
        if self.base_body_id < self.model.nbody:
            self.base_initial_position = self.data.body(self.base_body_id).xpos.copy()
        else:
            self.base_initial_position = np.zeros(3)
        
        # Reset previous action
        self.previous_action = np.zeros(self.action_space.shape[0])
        
        # Settle robot
        self._settle_robot()
        
        # Initialize viewer if needed
        if self.render_mode == 'human' and self.use_viewer and self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Warning: Could not create viewer: {e}")
                self.render_mode = None
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment."""
        # Clip and apply action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        
        # Step simulation
        for _ in range(10):  # 10 physics steps per control step
            mujoco.mj_step(self.model, self.data)
        
        # Update variables
        self.current_step += 1
        self.previous_action = action.copy()
        self.phase += self.phase_rate
        self.phase %= (2 * np.pi)
        
        # Get observation
        observation = self._get_obs()
        
        # Compute reward (scaled to reasonable values)
        reward = self._compute_reward(action) * 0.01
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get info
        info = self._get_info()
        
        # Render
        if self.viewer is not None:
            self.viewer.sync()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to the robot."""
        for i in range(min(len(action), self.model.nu)):
            self.data.ctrl[i] = action[i]
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        obs = []
        
        # 1. Joint positions
        obs.extend(self.data.qpos)
        
        # 2. Joint velocities
        obs.extend(self.data.qvel)
        
        # 3-5. Base state
        if self.base_body_id < self.model.nbody:
            # Orientation (quaternion)
            obs.extend(self.data.body(self.base_body_id).xquat)
            
            # Angular velocity
            obs.extend(self.data.body(self.base_body_id).cvel[3:6])
            
            # Linear velocity
            obs.extend(self.data.body(self.base_body_id).cvel[0:3])
        else:
            # Fallback values
            obs.extend([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            obs.extend([0.0, 0.0, 0.0])  # Zero angular velocity
            obs.extend([0.0, 0.0, 0.0])  # Zero linear velocity
        
        # 6. Foot contacts
        foot_contacts = []
        for site_id in self.foot_site_ids:
            if site_id < self.model.nsite:
                z_pos = self.data.site_xpos[site_id][2]
                foot_contacts.append(1.0 if z_pos < self.contact_threshold else 0.0)
            else:
                foot_contacts.append(0.0)
        obs.extend(foot_contacts)
        
        # 7. Previous action
        obs.extend(self.previous_action)
        
        # 8. Command
        obs.extend(self.command)
        
        # 9. Phase
        obs.extend([np.sin(self.phase), np.cos(self.phase)])
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current step."""
        reward = 0.0
        
        if self.base_body_id >= self.model.nbody:
            return float(reward)
        
        # 1. Velocity tracking
        base_vel = self.data.body(self.base_body_id).cvel[0:3]
        desired_forward = self.command[0]
        forward_vel = base_vel[0]
        forward_error = forward_vel - desired_forward
        reward += np.exp(-forward_error**2 / 0.1)
        
        # 2. Upright reward
        base_quat = self.data.body(self.base_body_id).xquat
        w, x, y, z = base_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        reward += np.exp(-roll**2 / 0.3) + np.exp(-pitch**2 / 0.3)
        
        # 3. Energy efficiency
        energy_penalty = -0.01 * np.sum(action**2)
        reward += energy_penalty
        
        # 4. Movement reward (encourage forward motion)
        if desired_forward > 0:
            reward += 0.1 * forward_vel
        
        # 5. Foot clearance (encourage lifting feet)
        foot_contacts = []
        for site_id in self.foot_site_ids:
            if site_id < self.model.nsite:
                z_pos = self.data.site_xpos[site_id][2]
                foot_contacts.append(1.0 if z_pos < self.contact_threshold else 0.0)
        contact_ratio = np.mean(foot_contacts)
        reward += 0.1 * contact_ratio  # Encourage some contact
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check termination conditions."""
        if self.base_body_id >= self.model.nbody:
            return False
        
        # Check if fallen
        base_quat = self.data.body(self.base_body_id).xquat
        w, x, y, z = base_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            return True
        
        # Check if too low
        base_z = self.data.body(self.base_body_id).xpos[2]
        if base_z < 0.1:
            return True
        
        return False
    
    def _randomize_initial_conditions(self, options: Optional[Dict]):
        """Randomize initial conditions."""
        if options and 'no_randomization' in options:
            return
        
        if hasattr(self, 'np_random'):
            # Randomize joint positions slightly
            for i in range(min(8, len(self.data.qpos))):
                noise = self.np_random.uniform(-0.05, 0.05)
                self.data.qpos[i] += noise
    
    def _settle_robot(self):
        """Let robot settle under gravity."""
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        if self.base_body_id < self.model.nbody:
            base_pos = self.data.body(self.base_body_id).xpos.copy()
            base_vel = self.data.body(self.base_body_id).cvel[0:3].copy()
        else:
            base_pos = np.zeros(3)
            base_vel = np.zeros(3)
        
        info = {
            'step': self.current_step,
            'base_position': base_pos,
            'base_velocity': base_vel,
            'command': self.command.copy(),
            'phase': self.phase,
            'total_energy': self.total_energy
        }
        
        return info
    
    def render(self):
        """Render the environment."""
        # Already handled by viewer if using mujoco.viewer
        pass
    
    def close(self):
        """Clean up resources."""
        if self.viewer:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None
    
    def set_command(self, command: np.ndarray):
        """Set desired velocity command."""
        self.command = np.array(command, dtype=np.float32)


# ====================
# Test Function
# ====================

def test_environment():
    """Test the environment."""
    print("Testing PassiveTwistMicrotaurEnv...")
    
    try:
        # Create environment
        env = PassiveTwistMicrotaurEnv(
            xml_path='scene.xml',
            render_mode='human',
            control_frequency=50,
            episode_duration=5.0,
            use_viewer=True
        )
        
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space.shape}")
        
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial base position: {info['base_position']}")
        
        # Test a few steps
        total_reward = 0
        for step in range(100):
            # Random action
            action = env.action_space.sample()
            
            # Set a forward command
            if step == 0:
                env.set_command([0.1, 0.0, 0.0])  # Forward 0.1 m/s
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 20 == 0:
                print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        env.close()
        print(f"Test completed! Total reward: {total_reward:.3f}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_environment()