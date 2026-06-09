import threading
import time
import numpy as np
import urx
from src.ur10e import T_TOOL_PEN, UR10e
import src.utils as utils

from src.utils import collision_check

RUN_RUNTIME_SAFETY_CHECKS = False
MOVEJ_SUCCESS_TOL_RAD = np.deg2rad(3.0)

# urx 0.9.0's movej(wait=True) returns early (it exits at 80% of the start
# distance and depends on the flaky is_program_running flag), which preempts
# the in-flight motion. Instead we send non-blocking and poll until the robot
# is genuinely within tolerance for a few consecutive reads.
MOVEJ_POLL_DT = 0.02  # s between joint polls while waiting for a move
MOVEJ_SETTLE_READS = 3  # consecutive in-tolerance reads required to call it done


def _wrapped_joint_error(current_joints, target_joints):
    return (current_joints - target_joints + np.pi) % (2 * np.pi) - np.pi


class URXControlThread(threading.Thread):
    def __init__(self, shared_state, robot_ip, hz=100, vj=0.5, aj=0.1,
                 joint_pos_limits=None, min_link_dist=0.05):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.robot_ip = robot_ip
        self.dt = 1.0 / hz
        # speedj watchdog: must stay safely longer than the interval between
        # commands, otherwise the command expires before the next one arrives
        # and the robot decelerates/stops every cycle (violent shaking).
        self.cmd_min_time = max(0.1, 3.0 * self.dt)
        self.robot = None
        self.running = True
        self.vj = vj
        self.aj = aj
        self.joint_pos_limits = joint_pos_limits
        self.min_link_dist = min_link_dist
        self.robot_model = UR10e()
        self._was_following = False

    def run(self):
        try:
            print(f"[URX] Connecting to robot at {self.robot_ip}...")
            self.robot = urx.Robot(self.robot_ip)

            self.shared_state.joint_pos = self.robot.getj()

            with self.shared_state.lock:
                self.shared_state.robot_connected = True

            print("[URX] Connected.")

            # TODO: FIRST MAKE SURE THAT WE ARE AT THE CONSTANT STARTING POINT

            while self.running:
                self.shared_state.joint_pos = self.robot.getj()
                if self.shared_state.homing:
                    modified_joint_pos = self.robot_model.DHClassicaltoModified(self.shared_state.joint_pos)
                    if np.linalg.norm(modified_joint_pos - self.shared_state.home_joints) < 1e-2:
                        self.shared_state.homing = False

                with self.shared_state.lock:
                    shutdown = self.shared_state.shutdown
                    enabled = self.shared_state.robot_enabled
                    following = self.shared_state.following_trajectory
                    u_curr = self.shared_state.u_curr.copy()
                    home_req = self.shared_state.home_requested
                    motion_req = getattr(self.shared_state, "motion_requested", False)

                if shutdown:
                    self.send_zero()
                    break

                if motion_req:
                    with self.shared_state.lock:
                        target_joints = np.array(self.shared_state.motion_target_joints, dtype=float).reshape(6,)
                        label = self.shared_state.motion_label
                        self.shared_state.motion_requested = False
                        self.shared_state.motion_in_progress = True
                        self.shared_state.motion_done = False
                        self.shared_state.motion_error = None
                    try:
                        print(f"[URX] Moving to {label}...")
                        reached = self._movej_blocking(target_joints, label)
                        if reached:
                            print(f"[URX] Reached {label}.")
                        else:
                            print(f"[URX] Move '{label}' did not settle within timeout.")
                        with self.shared_state.lock:
                            self.shared_state.motion_error = None
                    except Exception as e:
                        print(f"[URX] Move '{label}' returned URX error; continuing: {type(e).__name__}: {e!r}")
                        try:
                            current_joints = np.asarray(self.robot.getj(), dtype=float).reshape(6,)
                            self.shared_state.joint_pos = current_joints.tolist()
                        except Exception as state_error:
                            print(f"[URX] Could not read joints after '{label}': {type(state_error).__name__}: {state_error!r}")
                        with self.shared_state.lock:
                            self.shared_state.motion_error = None
                    finally:
                        motion_error = None
                        try:
                            current_joints = np.asarray(self.robot.getj(), dtype=float).reshape(6,)
                            self.shared_state.joint_pos = current_joints.tolist()
                            joint_error = _wrapped_joint_error(current_joints, target_joints)
                            joint_error_max = float(np.max(np.abs(joint_error)))
                            if joint_error_max > MOVEJ_SUCCESS_TOL_RAD:
                                motion_error = (
                                    f"not at requested joint target after '{label}': "
                                    f"max_error={np.rad2deg(joint_error_max):.2f} deg, "
                                    f"error_deg={np.array2string(np.rad2deg(joint_error), precision=2, suppress_small=True)}"
                                )
                                print(f"[WARN] {motion_error}")
                        except Exception as state_error:
                            motion_error = (
                                f"could not verify requested joint target after '{label}': "
                                f"{type(state_error).__name__}: {state_error!r}"
                            )
                            print(f"[WARN] {motion_error}")

                        with self.shared_state.lock:
                            self.shared_state.motion_error = motion_error
                            self.shared_state.motion_in_progress = False
                            self.shared_state.motion_done = True
                            self.shared_state.robot_enabled = False
                    continue

                if home_req:
                    with self.shared_state.lock:
                        self.shared_state.home_requested = False
                    try:
                        home_q = self.shared_state.home_joints.tolist()
                        classical_joint_angles = np.rad2deg(self.robot_model.DHModifiedToClassical(home_q))

                        safe = utils.SafetyCheck(self.robot_model, classical_joint_angles, T_TOOL_PEN)

                        print(self.robot_model.FK(classical_joint_angles))
                        # print(self.robot_model.FK(classical_joint_angles, T_TOOL_PEN))

                        if not safe:
                            shutdown = True
                            print("NOT SAFE, ABORTING")
                            break

                        print(f"[URX] Moving to home position...")
                        self.robot.movej(np.deg2rad(classical_joint_angles), vel=self.vj, acc=self.aj)
                        self.shared_state.joint_pos = self.robot.getj()

                        print(f"[URX] Home reached.")
                    except Exception as e:
                        print(f"[URX] Home error: {e}")
                    with self.shared_state.lock:
                        self.shared_state.robot_enabled = False
                    continue

                try:
                    if enabled:
                        if following:
                            self.send_command(u_curr)
                            self._was_following = True
                    elif self._was_following and not following:
                        self.send_zero()
                        self._was_following = False


                except Exception as e:
                    print(f"[URX] Command error: {e}")
                    time.sleep(0.1)

                time.sleep(self.dt)

        except Exception as e:
            print(f"[URX] Connection error: {e}")

        finally:
            try:
                if self.robot is not None:
                    self.send_zero()
                    self.robot.close()
            except Exception:
                pass

            with self.shared_state.lock:
                self.shared_state.robot_connected = False

            print("[URX] Thread exited.")

    def _movej_blocking(self, target_joints, label, timeout_s=30.0):
        """Send a movej and block (polling getj) until the robot is actually at
        the target, instead of trusting urx's unreliable wait.

        Returns True when the robot settles within tolerance, False on timeout.
        """
        target = np.asarray(target_joints, dtype=float).reshape(6,)
        self.robot.movej(target.tolist(), vel=self.vj, acc=self.aj, wait=False)

        start = time.time()
        settle = 0
        while time.time() - start < timeout_s:
            current = np.asarray(self.robot.getj(), dtype=float).reshape(6,)
            self.shared_state.joint_pos = current.tolist()

            err = float(np.max(np.abs(_wrapped_joint_error(current, target))))
            if err < MOVEJ_SUCCESS_TOL_RAD:
                settle += 1
                if settle >= MOVEJ_SETTLE_READS:
                    return True
            else:
                settle = 0

            time.sleep(MOVEJ_POLL_DT)

        return False

    def send_command(self, u):
        cmd = np.array(u).reshape(-1)
        joint_vels = np.clip(cmd, -self.vj, self.vj)

        if (
            RUN_RUNTIME_SAFETY_CHECKS
            and self.shared_state.joint_pos is not None
            and self.joint_pos_limits is not None
        ):
            theta_pred = np.array(self.shared_state.joint_pos) + joint_vels * self.dt
            safe, reason = collision_check(
                self.shared_state._collision_robot, theta_pred,
                self.joint_pos_limits, self.min_link_dist)
            if not safe:
                self.send_zero()
                self.shared_state.hard_stop(reason)
                print("AWOOGA NOT SAFE")
                return

        self.robot.speedj(
            joint_vels.tolist(),
            acc=self.aj,
            min_time=self.cmd_min_time,
        )

    def send_zero(self):
        if self.robot is not None:
            self.robot.speedj([0, 0, 0, 0, 0, 0], acc=self.aj, min_time=self.cmd_min_time)

    def stop(self):
        self.running = False
