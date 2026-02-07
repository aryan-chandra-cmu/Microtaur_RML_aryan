import mujoco
import mujoco.viewer
import numpy as np

MODEL_XML = "rigid_microtaur/scene.xml"

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    # Map actuator names -> indices
    act_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    act_id = {name: i for i, name in enumerate(act_names)}
    print("Actuators:", act_id)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = data.time

            # command one actuator (position target)
            data.ctrl[act_id["leg1_a_joint_act"]] = 0.6 * np.sin(2 * np.pi * 0.5 * t)

            # step
            mujoco.mj_step(model, data)

            # read IMU sensors (gyro, accel)
            imu_gyro = data.sensor("imu_gyro").data.copy()   # shape (3,)
            imu_acc  = data.sensor("imu_acc").data.copy()    # shape (3,)

            # print at a low rate so you don't spam the terminal
            if int(t * 50) % 10 == 0:  # ~5 Hz if dt ~0.02
                print("gyro:", imu_gyro, "acc:", imu_acc)

            viewer.sync()

if __name__ == "__main__":
    main()
