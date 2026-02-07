#!/usr/bin/env python3
"""
test_robot.py - Test robot simulation
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # Load your scene
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    print(f"Model loaded: {model.nq} DOFs, {model.nu} actuators")
    
    # Open viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer opened. Press ESC to exit.")
        
        # Simulate for 30 seconds
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            # Apply some simple oscillating controls
            for i in range(model.nu):
                if i % 2 == 0:  # Every other actuator
                    data.ctrl[i] = 0.1 * np.sin(2 * time.time() + i)
                else:
                    data.ctrl[i] = 0.05 * np.cos(2 * time.time() + i)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            time.sleep(0.01)
    
    print("Simulation complete.")

if __name__ == "__main__":
    main()