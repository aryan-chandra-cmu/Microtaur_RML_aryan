import os
import mujoco
import mujoco.viewer


# file_path = "active_twist_microtaur/scene.xml"
# file_path = "passive_twist_microtaur/scene.xml"
# file_path = "active_yaw_microtaur/scene.xml"
# file_path = "rigid_microtaur/scene.xml"
file_path = "family_scene.xml"

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)
# Create a simulation data structure
data = mujoco.MjData(model)
# mujoco.mj_saveLastXML("old/old_robot_files/mugatu_nice_feet_fixed_urdf/robot.xml", model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)