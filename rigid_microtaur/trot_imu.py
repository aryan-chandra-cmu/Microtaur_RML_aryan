import mujoco
import mujoco.viewer
import numpy as np

MODEL_XML = "rigid_microtaur/scene.xml"

# ----------------------------
# Helpers
# ----------------------------
def deg2rad(d):
    return d * np.pi / 180.0

def list_actuators(model):
    names = []
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(n)
    return names

def get_actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator '{name}' not found in model.")
    return aid

def get_sensor_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Sensor '{name}' not found in model.")
    return sid

def read_sensor_vec(model, data, sensor_id):
    """Return the sensor reading as a numpy view (correct length)."""
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr:adr + dim]

# ----------------------------
# Arduino-equivalent motor mapping
# ----------------------------
class MicrotaurArduinoPort:
    def __init__(self, model, data, motor_to_actuator_name, offset_deg=0.0,
                 motor_count=8, has_spine=False, spine_rigid=False, spine_goal_deg=0.0,
                 imu_sensor_names=("imu_gyro", "imu_acc"),
                 imu_print_hz=50):
        self.model = model
        self.data = data
        self.motor_count = motor_count
        self.offset = offset_deg
        self.has_spine = has_spine
        self.spine_rigid = spine_rigid
        self.spine_goal_deg = spine_goal_deg

        # Motor id -> actuator id
        self.motor_to_act = {}
        for mid, aname in motor_to_actuator_name.items():
            self.motor_to_act[mid] = get_actuator_id(model, aname)

        # IMU sensors
        self.imu_gyro_id = get_sensor_id(model, imu_sensor_names[0])
        self.imu_acc_id  = get_sensor_id(model, imu_sensor_names[1])
        #self.imu_quat_id = get_sensor_id(model, imu_sensor_names[2])

        # Print throttling
        self.imu_print_hz = float(imu_print_hz)
        self._next_imu_print_t = 0.0  # sim-time seconds

    def arduino_moveMotor_goal_deg(self, mid, position_deg):
        """Replicate Arduino moveMotor() -> goal_position[mid] (in degrees)."""
        if mid == self.motor_count and self.has_spine and self.spine_rigid:
            goal = self.spine_goal_deg

        elif (mid % 2) == 0:  # even motor
            if position_deg < 0:
                goal = max(position_deg, -30.0)
            else:
                goal = position_deg

        else:  # odd motor
            if 0 < position_deg < 90:
                goal = min(position_deg, 30.0)
            else:
                goal = -(360.0 - position_deg)

        # offsets
        if mid in (2, 8):
            goal = goal + self.offset
        elif mid in (1, 7):
            goal = goal - self.offset

        return goal

    def set_motor_target_deg(self, mid, cmd_position_deg):
        goal_deg = self.arduino_moveMotor_goal_deg(mid, cmd_position_deg)
        self.data.ctrl[self.motor_to_act[mid]] = deg2rad(goal_deg)

    def maybe_print_imu(self):
        """Print IMU at imu_print_hz using simulation time."""
        if self.imu_print_hz <= 0:
            return
        t = self.data.time
        if t < self._next_imu_print_t:
            return

        gyro = read_sensor_vec(self.model, self.data, self.imu_gyro_id)
        acc  = read_sensor_vec(self.model, self.data, self.imu_acc_id)
        #quat = read_sensor_vec(self.model, self.data, self.imu_quat_id)

        # copy() so it doesn't change under us between prints
        gyro = np.array(gyro, dtype=float)
        acc  = np.array(acc, dtype=float)
        #quat = np.array(quat, dtype=float)

        print(f"[t={t:8.3f}s] gyro={gyro}  acc={acc} ")

        self._next_imu_print_t = t + 1.0 / self.imu_print_hz

    def hold_ms(self, viewer, ms):
        """Advance sim for ms milliseconds, printing IMU continuously."""
        t_end = self.data.time + ms / 1000.0
        while self.data.time < t_end and viewer.is_running():
            mujoco.mj_step(self.model, self.data)

            # print IMU every step (throttled by imu_print_hz)
            self.maybe_print_imu()

            viewer.sync()

    # ----------------------------
    # Gaits translated 1:1 from Arduino
    # ----------------------------
    def ARest(self, time_ms):
        frontsAngle = 315
        backsAngle = 360 - frontsAngle
        for mid, ang in [(1, frontsAngle), (2, backsAngle), (5, frontsAngle), (6, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def AStepUp(self, time_ms):
        frontsAngle = 300
        backsAngle = 360 - frontsAngle
        for mid, ang in [(1, frontsAngle), (2, backsAngle), (5, frontsAngle), (6, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def AStepFoward(self, time_ms):
        right_front = 135
        right_back = 315
        left_front = 360 - right_front
        left_back = 360 - right_back
        for mid, ang in [(1, right_back), (2, right_front), (5, left_front), (6, left_back)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def AStepDown(self, time_ms):
        right_front = 60
        right_back = 325
        left_front = 360 - right_front
        left_back = 360 - right_back
        for mid, ang in [(1, right_back), (2, right_front), (5, left_front), (6, left_back)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def ASweep(self, time_ms):
        frontsAngle = 30
        backsAngle = 360 - frontsAngle
        for mid, ang in [(1, backsAngle), (2, frontsAngle), (5, backsAngle), (6, frontsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BRest(self, time_ms):
        frontsAngle = 45
        backsAngle = 360 - frontsAngle
        for mid, ang in [(8, frontsAngle), (7, backsAngle), (4, frontsAngle), (3, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BStepUp(self, time_ms):
        frontsAngle = 60
        backsAngle = 360 - frontsAngle
        for mid, ang in [(8, frontsAngle), (7, backsAngle), (4, frontsAngle), (3, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BStepFoward(self, time_ms):
        left_front = 225
        left_back = 45
        right_front = 360 - left_front
        right_back = 360 - left_back
        for mid, ang in [(8, left_back), (7, left_front), (4, right_front), (3, right_back)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BStepDown(self, time_ms):
        left_front = 300
        left_back = 35
        right_front = 360 - left_front
        right_back = 360 - left_back
        for mid, ang in [(8, left_back), (7, left_front), (4, right_front), (3, right_back)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BSweep(self, time_ms):
        frontsAngle = 330
        backsAngle = 360 - frontsAngle
        for mid, ang in [(8, backsAngle), (7, frontsAngle), (4, backsAngle), (3, frontsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    print("Actuators in model:")
    for n in list_actuators(model):
        print("  ", n)

    # TODO: EDIT THESE actuator names to match your scene.xml
    motor_to_actuator_name = {
        6: "leg1_a_joint_act",
        5: "leg1_e_joint_act",
        3: "leg2_a_joint_act",
        4: "leg2_e_joint_act",
        1: "leg3_a_joint_act",
        2: "leg3_e_joint_act",
        8: "leg4_a_joint_act",
        7: "leg4_e_joint_act",
    }


    robot = MicrotaurArduinoPort(
        model, data,
        motor_to_actuator_name=motor_to_actuator_name,
        offset_deg=0.0,
        motor_count=8,
        imu_sensor_names=("imu_gyro", "imu_acc"),
        imu_print_hz=5,   # change to 200 if you really want dense prints
    )

    stepUpTime = 35*4
    stepFowardTime = 35*4
    stepDownTime = 35*4
    sweepTime = 75*4
    restTime = 100*4

    with mujoco.viewer.launch_passive(model, data) as viewer:
        robot.ARest(1000); robot.BRest(1000)
        robot.hold_ms(viewer, 1000)

        while viewer.is_running():
            for _ in range(500):
                robot.BStepUp(stepUpTime);         robot.hold_ms(viewer, stepUpTime)
                robot.BStepFoward(stepFowardTime); robot.hold_ms(viewer, stepFowardTime)
                robot.BStepDown(stepDownTime);     robot.hold_ms(viewer, stepDownTime)
                robot.BSweep(sweepTime);           robot.hold_ms(viewer, sweepTime)
                robot.BRest(restTime);             robot.hold_ms(viewer, restTime)
                robot.hold_ms(viewer, 300)

                robot.AStepUp(stepUpTime);         robot.hold_ms(viewer, stepUpTime)
                robot.AStepFoward(stepFowardTime); robot.hold_ms(viewer, stepFowardTime)
                robot.AStepDown(stepDownTime);     robot.hold_ms(viewer, stepDownTime)
                robot.ASweep(sweepTime);           robot.hold_ms(viewer, sweepTime)
                robot.ARest(restTime);             robot.hold_ms(viewer, restTime)
                robot.hold_ms(viewer, 300)

                

            break

if __name__ == "__main__":
    main()
