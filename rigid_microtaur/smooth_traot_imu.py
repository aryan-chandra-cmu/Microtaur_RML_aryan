
import time as pytime
from datetime import datetime
import mujoco
import mujoco.viewer
import numpy as np
import csv
import os

MODEL_XML = "rigid_microtaur/scene.xml"


def deg2rad(d: float) -> float:
    return float(d) * np.pi / 180.0


def list_actuators(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]


def get_actuator_id(model, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator '{name}' not found in model.")
    return int(aid)


def get_sensor_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Sensor '{name}' not found in model.")
    return int(sid)


def read_sensor_vec(model, data, sensor_id: int):
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return data.sensordata[adr : adr + dim]


def actuator_joint_id(model, actuator_id: int) -> int:
    jid = int(model.actuator_trnid[actuator_id, 0])
    if jid < 0 or jid >= model.njnt:
        raise ValueError(f"Actuator {actuator_id} has invalid joint id {jid}")
    return jid


def joint_angle_deg(model, data, joint_id: int) -> float:
    adr = int(model.jnt_qposadr[joint_id])
    return float(np.degrees(float(data.qpos[adr])))


def unwrap_like_arduino(mid: int, deg: float) -> float:
    d = float(deg)
    if (mid % 2) == 0:
        if d > 270.0:
            d -= 360.0
    else:
        if d > 90.0:
            d -= 360.0
    return d


class MicrotaurArduinoPort:
    def __init__(
        self,
        model,
        data,
        motor_to_actuator_name,
        offset_deg=-15.0,
        motor_count=8,
        imu_sensor_names=("imu_gyro", "imu_acc", "imu_quat"),
        imu_print_hz=10,
        csv_log_path=None,
        log_relative_to_stand=True,
    ):
        self.model = model
        self.data = data
        self.motor_count = int(motor_count)

        # Arduino "front/back" per-motor offset magnitude (applied ONLY via _arduino_front_back_offset)
        self.offset = float(offset_deg)

        # scaled gait amplitude (KEY FIX)
        self.gait_scale = 1.0  # will be set in main during trot ramp

        self.log_relative_to_stand = bool(log_relative_to_stand)
        self.stand_zero_deg = None

        self.spawn_qpos = {}
        self.stand_goal_deg = {}
        self.sign = {mid: 0.0 for mid in range(1, self.motor_count + 1)}
        self._calibrated = False

        self.imu_print_hz = float(imu_print_hz)
        self._next_imu_print_t = 0.0

        self.motor_to_actuator_name = dict(motor_to_actuator_name)

        self.motor_to_act = {
            int(mid): get_actuator_id(self.model, aname)
            for mid, aname in self.motor_to_actuator_name.items()
        }
        self.motor_to_jnt = {
            int(mid): actuator_joint_id(self.model, int(aid))
            for mid, aid in self.motor_to_act.items()
        }

        self.imu_quat_id = get_sensor_id(model, imu_sensor_names[2])

        # CSV logging
        self.csv_log_path = csv_log_path
        self.csv_fd = None
        self.csv_writer = None
        if self.csv_log_path:
            d = os.path.dirname(self.csv_log_path)
            if d:
                os.makedirs(d, exist_ok=True)
            first = (not os.path.exists(self.csv_log_path)) or (
                os.path.getsize(self.csv_log_path) == 0
            )
            try:
                self.csv_fd = open(self.csv_log_path, "a", newline="")
                self.csv_writer = csv.writer(self.csv_fd)
                if first:
                    header = ["time_ms", "roll_deg", "pitch_deg", "yaw_deg"]
                    header += [f"m{i}_deg" for i in range(1, self.motor_count + 1)]
                    if self.log_relative_to_stand:
                        header += [
                            f"m{i}_deg_rel" for i in range(1, self.motor_count + 1)
                        ]
                    header += [f"m{i}_eff" for i in range(1, self.motor_count + 1)]
                    self.csv_writer.writerow(header)
                    self.csv_fd.flush()
            except Exception:
                self.csv_fd = None
                self.csv_writer = None

        self.ctrl_target = np.array(self.data.ctrl, dtype=float)
        self.ctrl_start = np.array(self.data.ctrl, dtype=float)

    def set_gait_scale(self, s: float):
        self.gait_scale = float(np.clip(s, 0.0, 1.0))

    # Arduino-style front/back offset only on motors 1,2,7,8
    # Front odd legs: 1,7 get  -offset
    # Back  even legs: 2,8 get  +offset
    # (matches your Arduino C++ moveMotor() behavior)
    def _arduino_front_back_offset(self, mid: int) -> float:
        mid = int(mid)
        if mid in (2, 8):  # back, even legs
            return +float(self.offset)
        if mid in (1, 7):  # front, odd legs
            return -float(self.offset)
        return 0.0

    def _capture_spawn_qpos(self):
        self.spawn_qpos = {}
        for mid in range(1, self.motor_count + 1):
            jid = self.motor_to_jnt.get(mid, None)
            if jid is None:
                continue
            adr = int(self.model.jnt_qposadr[jid])
            self.spawn_qpos[mid] = float(self.data.qpos[adr])

    def _compute_stand_goal_deg(self, stand_fronts: float, stand_backs: float):
        self.stand_goal_deg = {}
        for mid in range(1, self.motor_count + 1):
            cmd = stand_fronts if (mid % 2 == 1) else stand_backs
            cmd += self._arduino_front_back_offset(mid)  # apply offset ONCE here
            self.stand_goal_deg[mid] = self.arduino_moveMotor_goal_deg(mid, cmd)

    def _calibrate_joint_signs(self, viewer, eps_rad=0.02, settle_steps=80):
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            viewer.sync()

        for mid in range(1, self.motor_count + 1):
            if self.sign.get(mid, 0.0) != 0.0:
                continue
            aid = self.motor_to_act.get(mid, None)
            jid = self.motor_to_jnt.get(mid, None)
            if aid is None or jid is None:
                self.sign[mid] = 1.0
                continue

            adr = int(self.model.jnt_qposadr[jid])
            q0 = float(self.data.qpos[adr])

            old = float(self.data.ctrl[aid])
            self.data.ctrl[aid] = q0 + eps_rad
            for _ in range(12):
                mujoco.mj_step(self.model, self.data)

            q1 = float(self.data.qpos[adr])
            self.data.ctrl[aid] = old

            self.sign[mid] = 1.0 if (q1 - q0) >= 0.0 else -1.0

    def calibrate_once(self, viewer, stand_fronts=315.0, stand_backs=45.0):
        if self._calibrated:
            return
        self._capture_spawn_qpos()
        self._compute_stand_goal_deg(stand_fronts, stand_backs)
        self._calibrate_joint_signs(viewer)
        self._calibrated = True

    def arduino_moveMotor_goal_deg(self, mid: int, position_deg: float) -> float:
        """
        PURE Arduino mapping: wrap/clamp only.
        IMPORTANT: Does NOT apply self.offset. Offset is applied ONLY by _arduino_front_back_offset().
        """
        position_deg = float(position_deg)
        mid = int(mid)

        if (mid % 2) == 0:  # even
            if position_deg < 0:
                goal = max(position_deg, -30.0)
            else:
                goal = position_deg
        else:  # odd
            if 0 < position_deg < 90:
                goal = min(position_deg, 30.0)
            else:
                goal = -(360.0 - position_deg)

        return float(goal)

    def set_motor_target_deg(self, mid, cmd_position_deg):
        mid = int(mid)
        aid = self.motor_to_act.get(mid, None)
        if aid is None:
            return

        # Apply Arduino front/back offset ONCE in command space
        cmd_position_deg = float(cmd_position_deg) + self._arduino_front_back_offset(mid)

        # Arduino wrap/clamp mapping (odd/even)
        goal_deg = self.arduino_moveMotor_goal_deg(mid, cmd_position_deg)

        # Absolute target in radians
        target_q = deg2rad(goal_deg)

        # If actuator is tied to a limited joint, clamp to joint range
        jid = self.motor_to_jnt.get(mid, None)
        if jid is not None and int(self.model.jnt_limited[jid]) != 0:
            lo, hi = float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1])
            target_q = float(np.clip(target_q, lo, hi))

        self.ctrl_target[aid] = target_q


    def moveAllMotors(self, frontsAngle, backsAngle, time_ms):
        frontsAngle = float(frontsAngle)
        backsAngle = float(backsAngle)
        for mid in range(1, self.motor_count + 1):
            ang = frontsAngle if (mid % 2 == 1) else backsAngle
            self.set_motor_target_deg(mid, ang)
        return int(time_ms)

    def maybe_print_imu(self):
        if self.imu_print_hz <= 0:
            return
        t = float(self.data.time)
        if t < self._next_imu_print_t:
            return

        t_ms = int(t * 1000.0)

        roll = pitch = yaw = 0.0
        quat = np.array(read_sensor_vec(self.model, self.data, self.imu_quat_id), dtype=float)
        if quat.size >= 4:
            qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

            sinr_cosp = 2.0 * (qw * qx + qy * qz)
            cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
            roll = float(np.degrees(np.arctan2(sinr_cosp, cosr_cosp)))

            sinp = 2.0 * (qw * qy - qz * qx)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = float(np.degrees(np.arcsin(sinp)))

            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))

        motor_positions = []
        motor_positions_rel = []
        motor_effort = []

        for mid in range(1, self.motor_count + 1):
            pos_deg = 0.0
            eff = 0.0
            aid = self.motor_to_act.get(mid, None)
            jid = self.motor_to_jnt.get(mid, None)
            if (aid is not None) and (jid is not None):
                pos_deg = unwrap_like_arduino(mid, joint_angle_deg(self.model, self.data, jid))
                try:
                    eff = float(self.data.actuator_force[aid])
                except Exception:
                    eff = 0.0

            motor_positions.append(pos_deg)
            if self.log_relative_to_stand and (self.stand_zero_deg is not None):
                motor_positions_rel.append(pos_deg - float(self.stand_zero_deg.get(mid, 0.0)))
            motor_effort.append(eff)

        out = [t_ms, roll, pitch, yaw] + motor_positions
        if self.log_relative_to_stand:
            out += motor_positions_rel
        out += motor_effort

        if self.csv_writer is not None:
            self.csv_writer.writerow(out)
            try:
                self.csv_fd.flush()
            except Exception:
                pass

        self._next_imu_print_t = t + 1.0 / self.imu_print_hz

    def hold_ms(self, viewer, ms, realtime=False):
        duration = float(ms) / 1000.0
        if duration <= 0:
            return
        t0 = float(self.data.time)
        t_end = t0 + duration
        self.ctrl_start[:] = self.data.ctrl[:]
        wall_t0 = pytime.time()

        while float(self.data.time) < t_end and viewer.is_running():
            alpha = (float(self.data.time) - t0) / max(duration, 1e-9)
            if alpha > 1.0:
                alpha = 1.0

            self.data.ctrl[:] = (1.0 - alpha) * self.ctrl_start + alpha * self.ctrl_target
            mujoco.mj_step(self.model, self.data)
            self.maybe_print_imu()
            viewer.sync()

            if realtime:
                sim_elapsed = float(self.data.time) - t0
                wall_elapsed = pytime.time() - wall_t0
                sleep_s = sim_elapsed - wall_elapsed
                if sleep_s > 0:
                    pytime.sleep(sleep_s)

    def ARest(self, time_ms):
        frontsAngle = 315
        backsAngle = 45
        for mid, ang in [(1, frontsAngle), (2, backsAngle), (5, frontsAngle + 100), (6, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def AStepUp(self, time_ms):
        frontsAngle = 300
        backsAngle = 60
        for mid, ang in [(1, frontsAngle), (2, backsAngle), (5, frontsAngle), (6, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def AStepFoward(self, time_ms):
        right_front = 315
        right_back = 135
        left_front = 45
        left_back = 225
        for mid, ang in [(1, right_front), (2, right_back), (5, left_back), (6, left_front)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def AStepDown(self, time_ms):
        right_front = 325
        right_back = 60
        left_front = 35
        left_back = 300
        for mid, ang in [(1, right_front), (2, right_back), (5, left_back), (6, left_front)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def ASweep(self, time_ms):
        frontsAngle = 330
        backsAngle = 30
        for mid, ang in [(1, frontsAngle), (2, backsAngle), (5, frontsAngle), (6, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BRest(self, time_ms):
        frontsAngle = 45
        backsAngle = 315
        for mid, ang in [(8, frontsAngle), (7, backsAngle), (4, frontsAngle - 100), (3, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BStepUp(self, time_ms):
        frontsAngle = 60
        backsAngle = 300
        for mid, ang in [(8, frontsAngle), (7, backsAngle), (4, frontsAngle), (3, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BStepFoward(self, time_ms):
        left_front = 45
        left_back = 225
        right_front = 315
        right_back = 135
        for mid, ang in [(8, left_front), (7, left_back), (4, right_back), (3, right_front)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BStepDown(self, time_ms):
        left_front = 35
        left_back = 300
        right_front = 325
        right_back = 60
        for mid, ang in [(8, left_front), (7, left_back), (4, right_back), (3, right_front)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms

    def BSweep(self, time_ms):
        frontsAngle = 30
        backsAngle = 330
        for mid, ang in [(8, frontsAngle), (7, backsAngle), (4, frontsAngle), (3, backsAngle)]:
            self.set_motor_target_deg(mid, ang)
        return time_ms


def run_step_commands(robot, viewer, commands, realtime=False):
    for cmd, ms in commands:
        if not viewer.is_running():
            break
        getattr(robot, cmd)(ms)
        robot.hold_ms(viewer, ms, realtime=realtime)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    print("Actuators in model:")
    for n in list_actuators(model):
        print("  ", n)

    motor_to_actuator_name = {
        5: "leg1_a_joint_act",
        6: "leg1_e_joint_act",
        4: "leg2_a_joint_act",
        3: "leg2_e_joint_act",
        2: "leg3_a_joint_act",
        1: "leg3_e_joint_act",
        7: "leg4_a_joint_act",
        8: "leg4_e_joint_act",
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "rigid_microtaur/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_fname = os.path.join(log_dir, f"raw_log_sim_{ts}.csv")

    robot = MicrotaurArduinoPort(
        model,
        data,
        motor_to_actuator_name=motor_to_actuator_name,
        offset_deg=-15.0,
        motor_count=8,
        imu_sensor_names=("imu_gyro", "imu_acc", "imu_quat"),
        imu_print_hz=10,
        csv_log_path=log_fname,
        log_relative_to_stand=True,
    )

    REALTIME_PLAYBACK = True

    # gentler timings
    stepUpTime = 35
    stepFowardTime = 35
    stepDownTime = 35
    sweepTime = 75
    restTime = 100


    trot_steps = [
        ("BStepUp", stepUpTime),
        ("BStepFoward", stepFowardTime),
        ("BStepDown", stepDownTime),
        ("BSweep", sweepTime),
        ("BRest", restTime),
        ("AStepUp", stepUpTime),
        ("AStepFoward", stepFowardTime),
        ("AStepDown", stepDownTime),
        ("ASweep", sweepTime),
        ("ARest", restTime),
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # settle contacts
        for _ in range(400):
            mujoco.mj_step(model, data)
            viewer.sync()

        # go to stand
        robot.moveAllMotors(315, 45, 1000)
        robot.hold_ms(viewer, 1000, realtime=REALTIME_PLAYBACK)

        # run trot cycles (no scaling)
        cycles = 100
        for _ in range(cycles):
            if not viewer.is_running():
                break
            run_step_commands(robot, viewer, trot_steps, realtime=REALTIME_PLAYBACK)

        # return to sit or stand, your choice
        robot.moveAllMotors(315, 45, 1500)
        robot.hold_ms(viewer, 1500, realtime=REALTIME_PLAYBACK)


    if robot.csv_fd is not None:
        try:
            robot.csv_fd.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
