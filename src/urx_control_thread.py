import threading
import time
import numpy as np
import urx
from src.ur10e import UR10e
import src.utils as utils

from src.utils import collision_check

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
                # print("joint angles")
                # print(self.shared_state.joint_pos)
                modified_joint_pos = self.robot_model.DHClassicaltoModified(self.shared_state.joint_pos)
                # print(np.linalg.norm(modified_joint_pos - self.shared_state.home_joints))
                if np.linalg.norm(modified_joint_pos - self.shared_state.home_joints) < 1e-2:
                    if self.shared_state.homing:
                        self.shared_state.homing = False
                
                with self.shared_state.lock:
                    shutdown = self.shared_state.shutdown
                    enabled = self.shared_state.robot_enabled
                    u_curr = self.shared_state.u_curr.copy()
                    home_req = self.shared_state.home_requested
                    
                if shutdown:
                    self.send_zero()
                    break

                if home_req:
                    with self.shared_state.lock:
                        self.shared_state.home_requested = False
                    try:
                        home_q = self.shared_state.home_joints.tolist()
                        classical_joint_angles = np.rad2deg(self.robot_model.DHModifiedToClassical(home_q))

                        Ttp_pen = utils.trans_z(0.3)

                        safe = utils.SafetyCheck(self.robot_model, classical_joint_angles)
                        
                        print(self.robot_model.FK(classical_joint_angles))
                        # print(self.robot_model.FK(classical_joint_angles, Ttp_pen))
                        
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
                        curr_joints = np.array(self.robot.getj()).reshape((6,1))
                        future_joints = (curr_joints + 0.5 * u_curr)
                        
                        classical_joint_angles = np.rad2deg(future_joints)

                        Ttp_pen = utils.trans_z(0.3)

                        safe = utils.SafetyCheck(self.robot_model, classical_joint_angles)
                        
                        # print("FKIN")
                        # print(self.robot_model.FK(np.rad2deg(curr_joints)))
                        # print(u_curr)
                        
                        if not safe:
                            shutdown = True
                            print("NOT SAFE, ABORTING")
                            break
                        
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

        if self.shared_state.joint_pos is not None and self.joint_pos_limits is not None:
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
            min_time=0.4
        )

    def send_zero(self):
        if self.robot is not None:
            self.robot.speedj([0, 0, 0, 0, 0, 0], acc=self.aj, min_time=0.4)

    def stop(self):
        self.running = False