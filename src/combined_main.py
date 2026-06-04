"""
Combined script that runs both IMU GUI and MPC simulation simultaneously.
Run this instead of running main.py and imu.py separately.
"""

import threading
import sys
import time
import tkinter as tk
import numpy as np
from collections import deque
from src.utils import *
from src.tracking import GUI
from src.trajectory_tracking import MPCSimulationThread

try:
    from src.urx_control_thread import URXControlThread
    HAS_URX = True
except ImportError:
    HAS_URX = False

# TODO: MODIFY ALL THE PARAMETERS BELOW BEFORE TESTING
SAMPLING_RATE = 75 # Hz
MPC_HORIZON = SAMPLING_RATE // 12 # sec = horizon_samples / sampling_rate
# The workspace offset MUST place the robot away from wrist singularities.
# [0.5, 0.5, 0.5, 0, 0, 0] has zero rotation which aligns wrist axes (singular).
# Adding a small rotation breaks the singularity and makes IK stable.
WORKSPACE_OFFSET = pose6_to_T([0, -0.8, -0.015, np.pi, 0.05, 0.05])

# ── UR10e joint limits (CHANGE THESE for your actual robot) ──────────────
# Hardware max from datasheet:
#   Joints 0-1 (base, shoulder): 2.094 rad/s  (120 deg/s)
#   Joints 2-5 (elbow, wrists):  3.142 rad/s  (180 deg/s)
# Working limits (conservative for safety):
VJ = 0.7   # rad/s  -- uniform working velocity limit for MPC
AJ = 1.2   # rad/s² -- uniform working acceleration limit for MPC

# Per-joint working velocity limits (rad/s).
# UPDATE THESE when you know exact per-joint limits for your setup.
JOINT_VEL_LIMITS = np.array([
    VJ,   # joint 0 - base (shoulder pan)
    VJ,   # joint 1 - shoulder (lift)
    VJ,   # joint 2 - elbow
    VJ,   # joint 3 - wrist 1
    VJ,   # joint 4 - wrist 2
    VJ,   # joint 5 - wrist 3
])

# Per-joint absolute position limits (rad). UR10e datasheet: ±360 deg.
# Working limits set to ±350 deg for a safety margin.
JOINT_POS_LIMITS = np.array([6.1087, 6.1087, 6.1087, 6.1087, 6.1087, 6.1087])

# Minimum allowed distance (m) between non-adjacent link segments.
MIN_LINK_DISTANCE = 0.05


def _cartesian_interp_z_clamped(theta_prev, theta_next, N, robot, workspace_z):
    """Interpolate in Cartesian space, clamping the Z coordinate at each step.

    Returns a list of N joint arrays (excluding theta_prev, including
    theta_next), or None if any IK fails or a branch discontinuity is
    detected.
    """
    prev_class_deg = np.rad2deg(robot.DHModifiedToClassical(theta_prev))
    next_class_deg = np.rad2deg(robot.DHModifiedToClassical(theta_next))

    T_prev = robot.FK(prev_class_deg)
    T_next = robot.FK(next_class_deg)

    pose_prev = T_to_pose6(T_prev)
    pose_next = T_to_pose6(T_next)

    points = []
    prev_joints = theta_prev

    for i in range(1, N + 1):
        alpha = i / N
        pose_interp = (1 - alpha) * pose_prev + alpha * pose_next
        pose_interp[2] = workspace_z

        T_interp = pose6_to_T(pose_interp)

        try:
            joints = robot.IK("elbow_up_2", T_interp)
        except Exception:
            return None

        if joints is None or np.any(np.isnan(joints)):
            return None

        delta = joints - prev_joints
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        if np.max(np.abs(delta)) > np.pi / 2:
            return None

        points.append(joints)
        prev_joints = joints

    return points


def interpolate_joint_segment(theta_prev, theta_next, dt, v_max,
                              robot=None, workspace_z=None):
    """Subdivide a joint-space segment so every joint respects its velocity limit.

    When *robot* and *workspace_z* are provided the function first attempts
    Cartesian-space interpolation with the end-effector Z coordinate clamped,
    falling back to plain joint-space interpolation on IK failure.

    Parameters
    ----------
    theta_prev  : (6,) current joint angles
    theta_next  : (6,) target joint angles
    dt          : float, timestep between the two points
    v_max       : (6,) per-joint velocity limits (rad/s)
    robot       : UR10e instance (optional, enables Cartesian interpolation)
    workspace_z : float (optional, desired Z in base frame)

    Returns
    -------
    list of (6,) arrays -- intermediate waypoints (excluding theta_prev,
    including theta_next).  If the segment is already feasible the list
    contains only theta_next.
    """
    delta = theta_next - theta_prev
    delta = (delta + np.pi) % (2 * np.pi) - np.pi

    if dt <= 0:
        return [theta_next]

    v_required = np.abs(delta) / dt
    ratios = v_required / v_max
    r = np.max(ratios)

    N = max(1, int(np.ceil(r)))

    if N == 1:
        return [theta_next]

    if robot is not None and workspace_z is not None:
        try:
            cart_pts = _cartesian_interp_z_clamped(
                theta_prev, theta_next, N, robot, workspace_z
            )
            if cart_pts is not None:
                return cart_pts
        except Exception:
            pass

    points = []
    for i in range(1, N + 1):
        pt = theta_prev + (i / N) * delta
        pt = (pt + np.pi) % (2 * np.pi) - np.pi
        points.append(pt)
    return points


MAX_QUEUE_LEN = 500

STATIONARY_JOINT_EPS = 0.02
STATIONARY_COUNT_THRESH = 20


class SharedTrajectoryState:
    """Thread-safe shared state for IMU and MPC communication.

    trajectory_queue is a FIFO: IMU appends to the right, MPC reads from
    the left and pops after executing each step.
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.following_trajectory = False
        self.trajectory_queue = deque()

        self.u_curr = np.zeros((6, 1))
        self.robot_enabled = False
        self.robot_connected = False
        self.shutdown = False
        self.joint_pos = None
        self.home_requested = False
        self.homing = False
        self.collision_detected = False
        self.collision_reason = ""
        self.prerecorded_flag = False

        self._stable_target = None
        self._stable_count = 0

        # Compute home joint angles from WORKSPACE_OFFSET via IK
        from src.ur10e import UR10e
        self._collision_robot = UR10e()
        self.home_joints = self._collision_robot.IK("elbow_up_2", WORKSPACE_OFFSET)
        self._workspace_z = float(WORKSPACE_OFFSET[2, 3])
        print(f"[HOME] Home joints (deg): {np.degrees(self.home_joints).round(1)}")
        print(f"[HOME] Workspace Z (clamped): {self._workspace_z:.4f} m")

        self._last_joint_target = None

    @property
    def trajectory_window(self):
        """MPC-facing view: return the next MPC_HORIZON+1 points as a list.

        If the queue has fewer points than needed, pad by repeating the
        last available point so the MPC always gets a full horizon.
        """
        with self.lock:
            buf = list(self.trajectory_queue)

        n_need = MPC_HORIZON + 1
        if len(buf) == 0:
            return np.zeros((n_need, 6))
        if len(buf) >= n_need:
            return np.array(buf[:n_need])

        pad = [buf[-1]] * (n_need - len(buf))
        return np.array(buf + pad)

    def consume_one(self):
        """Pop the first point after the MPC has executed it."""
        with self.lock:
            if len(self.trajectory_queue) > 1:
                self.trajectory_queue.popleft()

    def _update_stationary_target_state(self, theta_next):
        if self._stable_target is None:
            self._stable_target = theta_next.copy()
            self._stable_count = 1
            return

        delta = theta_next - self._stable_target
        delta = (delta + np.pi) % (2 * np.pi) - np.pi

        if np.max(np.abs(delta)) < STATIONARY_JOINT_EPS:
            self._stable_count += 1
        else:
            self._stable_target = theta_next.copy()
            self._stable_count = 1

    def append_joint_target(self, theta_next, dt):
        theta_next = np.asarray(theta_next, dtype=float).reshape(-1)
        if theta_next.shape[0] != 6:
            print(f"[WARN] append_joint_target got bad shape: {theta_next.shape}")
            return

        collision_reason = None

        with self.lock:
            if self.collision_detected:
                return

            self._update_stationary_target_state(theta_next)

            # If the desired target has stayed almost unchanged for long enough,
            # stop feeding the queue. This gives the robot time to catch up.
            if self._stable_count >= STATIONARY_COUNT_THRESH:
                if self._stable_count == STATIONARY_COUNT_THRESH:
                    print(
                        f"[QUEUE HOLD] Target stationary for "
                        f"{self._stable_count} samples; pausing queue appends. "
                        f"Current queue len={len(self.trajectory_queue)}"
                    )
                return

            if self._last_joint_target is None:
                safe, reason = collision_check(
                    self._collision_robot, theta_next,
                    JOINT_POS_LIMITS, MIN_LINK_DISTANCE
                )
                if not safe:
                    collision_reason = reason
                else:
                    self.trajectory_queue.append(theta_next.copy())
                    self._last_joint_target = theta_next.copy()

            else:
                points = interpolate_joint_segment(
                    self._last_joint_target, theta_next, dt, JOINT_VEL_LIMITS,
                    robot=self._collision_robot, workspace_z=self._workspace_z
                )

                if len(points) > 1:
                    delta = theta_next - self._last_joint_target
                    delta = (delta + np.pi) % (2 * np.pi) - np.pi
                    v_req = np.abs(delta) / dt
                    worst = np.argmax(v_req / JOINT_VEL_LIMITS)
                    print(
                        f"[INTERP] Joint {worst} needs "
                        f"{v_req[worst]:.2f} rad/s (limit {JOINT_VEL_LIMITS[worst]:.2f}), "
                        f"N={len(points)}"
                    )

                for pt in points:
                    safe, reason = collision_check(
                        self._collision_robot, pt,
                        JOINT_POS_LIMITS, MIN_LINK_DISTANCE
                    )
                    if not safe:
                        collision_reason = reason
                        break
                    self.trajectory_queue.append(pt.copy())

                if collision_reason is None:
                    self._last_joint_target = theta_next.copy()

            while len(self.trajectory_queue) > MAX_QUEUE_LEN:
                self.trajectory_queue.popleft()

        if collision_reason is not None:
            self.hard_stop(collision_reason)

    def request_home(self):
        with self.lock:
            self.home_requested = True
            self.homing = True
            self.following_trajectory = False
            self.robot_enabled = True
            print(f"[HOME] Homing to {np.degrees(self.home_joints).round(1)} deg")

    def set_prerecorded_flag(self):
        if self.prerecorded_flag:
            self.prerecorded_flag = False
        else:
            self.prerecorded_flag = True
        print("Pre-recorded flag set to " + str(self.prerecorded_flag))

    def start_following(self):
        with self.lock:
            self.collision_detected = False
            self.collision_reason = ""
            self.trajectory_queue = deque()
            self.following_trajectory = True
            self.robot_enabled = True
            self._last_joint_target = None
            self._stable_target = None
            self._stable_count = 0

    def stop_following(self):
        with self.lock:
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6, 1))
            self._last_joint_target = None
            self._stable_target = None
            self._stable_count = 0

    def hard_stop(self, reason):
        with self.lock:
            self.collision_detected = True
            self.collision_reason = reason
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6, 1))
            self._last_joint_target = None
            self._stable_target = None
            self._stable_count = 0
        print(f"[COLLISION] HARD STOP: {reason}")

    def clear_collision(self):
        with self.lock:
            self.collision_detected = False
            self.collision_reason = ""
        print("[COLLISION] Cleared.")


def run_imu_gui(shared_state):
    """Run the IMU GUI in the main thread (required for Tkinter)."""
    root = tk.Tk()
    app = GUI(root, shared_state, SAMPLING_RATE, MPC_HORIZON, WORKSPACE_OFFSET)
    root.mainloop()


def run_mpc_background(shared_state, mpc_horizon, status_callback=None):
    """
    Run MPC simulation in a background thread, toggling between active/idle based on
    shared_state.following_trajectory.
    """
    sim_thread = None
    last_status = None

    try:
        while True:
            with shared_state.lock:
                following = shared_state.following_trajectory
                traj_len = len(shared_state.trajectory_queue)

            if following:
                if sim_thread is None:
                    if traj_len >= mpc_horizon+1:
                        print("[MPC Thread] Trajectory following activated with a valid trajectory!")
                        msg = "MPC: Trajectory following activated, starting simulation..."
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg

                        sim_thread = MPCSimulationThread(
                            shared_state=shared_state,
                            mpc_horizon=mpc_horizon,
                            dt= 1 / SAMPLING_RATE,
                            workspace_offset=WORKSPACE_OFFSET,
                            vj=VJ,
                            aj=AJ
                        )
                        sim_thread.start()
                    else:
                        msg = "MPC: Waiting for full trajectory window..."
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        time.sleep(0.01)

                elif sim_thread.is_alive():
                    msg = f"MPC Status: {sim_thread.status}"
                    if status_callback and msg != last_status:
                        status_callback(msg)
                        last_status = msg
                    time.sleep(0.05)

                else:
                    if sim_thread.status == "completed":
                        msg = "MPC simulation completed!"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        print("[MPC Thread] Simulation completed!")
                    else:
                        msg = f"MPC Error: {sim_thread.status} - {sim_thread.error_msg}"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        print(f"[MPC Thread] Simulation crashed with status: {sim_thread.status}")

                        shared_state.stop_following()
                        print("[MPC Thread] Automatically disabled trajectory following due to crash")

                    sim_thread = None
                    time.sleep(0.01)

            else:
                if sim_thread is not None:
                    if sim_thread.is_alive():
                        print("[MPC Thread] Trajectory following disabled, stopping simulation...")
                        msg = "MPC: Paused (waiting for trajectory following)"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        sim_thread.join(timeout=2)

                    sim_thread = None

                msg = "MPC: Idle (waiting for 'Start Following')"
                if status_callback and msg != last_status:
                    status_callback(msg)
                    last_status = msg

                time.sleep(0.01)

    except Exception as e:
        msg = f"MPC Error: {str(e)}"
        if status_callback and msg != last_status:
            status_callback(msg)
        print(f"[MPC Thread] Error: {e}")


def main():
    """Launch both IMU GUI and MPC simulation."""
    # Create shared state
    shared_state = SharedTrajectoryState()

    # Start MPC simulation in a daemon thread
    # It will wait until user clicks "Start Following" button
    mpc_thread = threading.Thread(target=run_mpc_background, args=(shared_state, MPC_HORIZON), daemon=True)
    mpc_thread.start()

    urx_thread = None
    if HAS_URX:
        urx_thread = URXControlThread(
            shared_state=shared_state,
            robot_ip="192.168.0.2",
            hz=100,
            vj=VJ,
            aj=AJ,
            joint_pos_limits=JOINT_POS_LIMITS,
            min_link_dist=MIN_LINK_DISTANCE
        )
        urx_thread.start()
    else:
        print("[WARN] urx not installed -- running without robot arm")

    print("Starting combined IMU + MPC application...")
    print("MPC simulation is waiting for 'Start Following' button to be pressed in the GUI.")
    run_imu_gui(shared_state)

    with shared_state.lock:
        shared_state.shutdown = True

    if urx_thread is not None:
        urx_thread.stop()
        urx_thread.join(timeout=2)


if __name__ == "__main__":
    main()