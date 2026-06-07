import threading
import time
import inspect
import numpy as np
import urx
from src.ur10e import T_TOOL_PEN, UR10e
import src.utils as utils

from src.utils import collision_check

MOVEJ_SUCCESS_TOL_RAD = np.deg2rad(1.0)
RUN_RUNTIME_SAFETY_CHECKS = False


class URXControlThread(threading.Thread):
    def __init__(self, shared_state, robot_ip, hz=100, vj=0.5, aj=0.1,
                 joint_pos_limits=None, min_link_dist=0.05):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.robot_ip = robot_ip
        self.dt = 1.0 / hz
        self.robot = None
        self.running = True
        self.vj = vj
        self.aj = aj
        self.joint_pos_limits = joint_pos_limits
        self.min_link_dist = min_link_dist
        self.robot_model = UR10e()

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
                        self.robot.movej(target_joints.tolist(), vel=self.vj, acc=self.aj)
                        self.shared_state.joint_pos = self.robot.getj()
                        print(f"[URX] Reached {label}.")
                        with self.shared_state.lock:
                            self.shared_state.motion_error = None
                    except Exception as e:
                        error_parts = [
                            f"{type(e).__name__}: {e!r}",
                            f"target_joints_rad={np.array2string(target_joints, precision=5, suppress_small=True)}",
                        ]
                        reached_target = False
                        try:
                            current_joints = np.asarray(self.robot.getj(), dtype=float).reshape(6,)
                            self.shared_state.joint_pos = current_joints.tolist()
                            joint_error = current_joints - target_joints
                            joint_error_norm = np.linalg.norm(joint_error)
                            reached_target = joint_error_norm <= MOVEJ_SUCCESS_TOL_RAD
                            error_parts.extend([
                                f"current_joints_rad={np.array2string(current_joints, precision=5, suppress_small=True)}",
                                f"joint_error_rad={np.array2string(joint_error, precision=5, suppress_small=True)}",
                                f"joint_error_deg={np.array2string(np.rad2deg(joint_error), precision=2, suppress_small=True)}",
                                f"joint_error_norm_rad={joint_error_norm:.6f}",
                            ])
                        except Exception as state_error:
                            error_parts.append(
                                f"state_read_error={type(state_error).__name__}: {state_error!r}"
                            )
                        detailed_error = " | ".join(error_parts)
                        if reached_target:
                            print(
                                f"[URX] Move '{label}' reached target despite URX error: "
                                f"{detailed_error}"
                            )
                            detailed_error = None
                        else:
                            print(f"[URX] Move '{label}' error: {detailed_error}")
                        with self.shared_state.lock:
                            self.shared_state.motion_error = detailed_error
                    finally:
                        with self.shared_state.lock:
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
                        if self.shared_state.following_trajectory:
                            self.send_command(u_curr)
                        else:
                            self.send_zero()    
                        
                        
                except Exception as e:
                    print(f"[URX] Command error: {e}")
                    self.send_zero()
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

        self._speedj(joint_vels.tolist(), acc=self.aj, min_time=0.4)

    def send_zero(self):
        if self.robot is not None:
            self._speedj([0, 0, 0, 0, 0, 0], acc=self.aj, min_time=0.4)

    def _speedj(self, joint_vels, acc, min_time):
        """Call urx speedj across versions with different time-arg names."""
        try:
            params = inspect.signature(self.robot.speedj).parameters
        except (TypeError, ValueError):
            params = {}

        if "min_time" in params:
            self.robot.speedj(joint_vels, acc=acc, min_time=min_time)
        elif "t_min" in params:
            self.robot.speedj(joint_vels, acc=acc, t_min=min_time)
        elif "t" in params:
            self.robot.speedj(joint_vels, acc=acc, t=min_time)
        else:
            self.robot.speedj(joint_vels, acc, min_time)

    def stop(self):
        self.running = False
