import mujoco
import mujoco.viewer
import time
import numpy as np

mjcf_path = "family_scene.xml"

model = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model)

timestep = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
  # Record the start time of the first step.
  step_start = time.monotonic()

  while viewer.is_running():
    
    idxs = np.arange(32)
    ctrl_ary = 0.2 * np.sin(5 * data.time) * 2 * (idxs % 2 - 0.5)

    # print("ctrl_ary", ctrl_ary)

    # insert zero at idx 16 and 0, where the spine motors are
    ctrl_ary = np.insert(ctrl_ary, [0, 16], 0.0)

    # print("ctrl_ary with zeroes", ctrl_ary)

    data.ctrl[:] = ctrl_ary

    mujoco.mj_step(model, data)
    viewer.sync()
    time_until_next_step = timestep - (time.monotonic() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
    step_start = time.monotonic()