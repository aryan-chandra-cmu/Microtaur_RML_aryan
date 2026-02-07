import mujoco
import mujoco.viewer
import time
import numpy as np

mjcf_path = "active_twist_microtaur/scene.xml"
# mjcf_path = "rigid_microtaur/scene.xml"

model = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model)

timestep = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
  # Record the start time of the first step.
  step_start = time.monotonic()

  while viewer.is_running():

    for j in range(8):
        data.ctrl[j+1] = 0.2 * np.sin(5*data.time) * 2 * (j % 2 - 0.5)

    mujoco.mj_step(model, data)
    viewer.sync()
    time_until_next_step = timestep - (time.monotonic() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
    step_start = time.monotonic()