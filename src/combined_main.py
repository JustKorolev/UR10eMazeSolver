"""
Main orchestration for maze planning and MPC trajectory execution.

Expected high-level sequence:
1. Move the robot to a known overhead camera pose.
2. Capture an overhead maze image.
3. Run A* on the image and fit a spline in pixel coordinates.
4. Convert spline pixels to local maze lengths using ArUco scale.
5. Transform local maze coordinates into robot world coordinates.
6. Add a drawing-plane z coordinate and convert poses to joint targets via IK.
7. Feed the joint trajectory to MPC and send velocity commands to the robot.
"""

import threading
import time
from collections import deque

import numpy as np

from src.astar import astar, build_spline, create_nodes
from src.cbf import CBFSafetyFilter, JointLimitCBF
from src.trajectory_tracking import MPCSimulationThread
from src.ur10e import UR10e
from src.utils import pixel_points_to_lengths, pose6_to_T

try:
    from src.urx_control_thread import URXControlThread
    HAS_URX = True
except ImportError:
    HAS_URX = False


SAMPLING_RATE = 75  # Hz
MPC_HORIZON = SAMPLING_RATE // 12

WORKSPACE_OFFSET = pose6_to_T([0, -0.8, -0.015, np.pi, 0.05, 0.05])
DRAWING_PLANE_Z = float(WORKSPACE_OFFSET[2, 3])

VJ = 0.7  # rad/s
AJ = 1.2  # rad/s^2

JOINT_POS_LIMITS = np.array([6.1087, 6.1087, 6.1087, 6.1087, 6.1087, 6.1087])
MIN_LINK_DISTANCE = 0.05
ROBOT_IP = "192.168.0.2"


class SharedTrajectoryState:
    """Thread-safe state shared by maze planning, MPC, and robot control."""

    def __init__(self, mpc_horizon=MPC_HORIZON, workspace_offset=WORKSPACE_OFFSET):
        self.lock = threading.Lock()
        self.mpc_horizon = mpc_horizon

        self.following_trajectory = False
        self.trajectory_queue = deque()

        self.u_curr = np.zeros((6, 1))
        self.robot_enabled = False
        self.robot_connected = False
        self.shutdown = False
        self.joint_pos = None

        # Kept for URXControlThread compatibility until homing/capture poses are refactored.
        self.home_requested = False
        self.homing = False

        self.collision_detected = False
        self.collision_reason = ""

        self.robot_model = UR10e(workspace_offset=workspace_offset)
        self._collision_robot = self.robot_model
        self.home_joints = self.robot_model.IK("elbow_up_2", workspace_offset)
        self._workspace_z = float(workspace_offset[2, 3])
        self.cbf_filter = CBFSafetyFilter(
            constraints=[
                JointLimitCBF(
                    q_min=-JOINT_POS_LIMITS,
                    q_max=JOINT_POS_LIMITS,
                    margin=0.05,
                )
            ],
            alpha=5.0,
            u_min=-np.full(6, VJ),
            u_max=np.full(6, VJ),
        )

    @property
    def trajectory_window(self):
        """Return the next MPC horizon of joint targets, padding with the last point."""
        with self.lock:
            buf = list(self.trajectory_queue)

        n_need = self.mpc_horizon + 1
        if len(buf) == 0:
            return np.zeros((n_need, 6))
        if len(buf) >= n_need:
            return np.array(buf[:n_need])

        pad = [buf[-1]] * (n_need - len(buf))
        return np.array(buf + pad)

    def load_joint_trajectory(self, joint_trajectory):
        """Load a complete joint trajectory for MPC to consume one point at a time."""
        joint_trajectory = np.asarray(joint_trajectory, dtype=float)
        if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] != 6:
            raise ValueError("joint_trajectory must have shape (num_points, 6)")

        with self.lock:
            self.trajectory_queue = deque(q.copy() for q in joint_trajectory)
            self.collision_detected = False
            self.collision_reason = ""

    def start_following(self):
        with self.lock:
            self.following_trajectory = True
            self.robot_enabled = True

    def stop_following(self):
        with self.lock:
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6, 1))

    def consume_one(self):
        """Pop the first target after MPC has executed one timestep."""
        with self.lock:
            if len(self.trajectory_queue) > 1:
                self.trajectory_queue.popleft()
            else:
                self.following_trajectory = False
                self.robot_enabled = False

    def hard_stop(self, reason):
        with self.lock:
            self.collision_detected = True
            self.collision_reason = reason
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6, 1))
        print(f"[COLLISION] HARD STOP: {reason}")


def request_overhead_capture_pose(shared_state):
    """Placeholder: move robot to the known overhead pose for maze imaging."""
    # TODO: Replace this with the calibrated overhead camera pose, not home_joints.
    with shared_state.lock:
        shared_state.home_requested = True
        shared_state.homing = True
        shared_state.robot_enabled = True


def capture_maze_image():
    """Placeholder for camera capture at the overhead pose."""
    # TODO: Capture and return the overhead maze image as a numpy array.
    raise NotImplementedError("Maze image capture is not implemented yet.")


def nearest_free_node(nodes, pixel_xy):
    """Return the free sampled A* node nearest to a pixel coordinate."""
    x, y = pixel_xy
    free_nodes = [node for node in nodes if not node.blocked]
    if not free_nodes:
        return None
    return min(free_nodes, key=lambda node: (node.x - x) ** 2 + (node.y - y) ** 2)


def plan_spline_pixel_path(
    maze_image,
    start_pixel,
    goal_pixel,
    N=100,
    samples_per_segment=20,
    control_point_stride=10,
):
    """Run A* on the maze image and return a smoothed pixel path."""
    nodes = create_nodes(N, maze_image)
    start_node = nearest_free_node(nodes, start_pixel)
    goal_node = nearest_free_node(nodes, goal_pixel)
    if start_node is None or goal_node is None:
        return None

    path = astar(nodes, start_node, goal_node)
    if path is None:
        return None
    return build_spline(
        path,
        samples_per_segment=samples_per_segment,
        control_point_stride=control_point_stride,
    )


def local_maze_to_world(local_xy_points, T_world_maze):
    """Transform local maze XY points into robot world XY points."""
    points = np.asarray(local_xy_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("local_xy_points must have shape (num_points, 2)")

    homog = np.column_stack((points, np.zeros(len(points)), np.ones(len(points))))
    world = (np.asarray(T_world_maze, dtype=float).reshape(4, 4) @ homog.T).T
    return world[:, :2]


def add_drawing_plane_z(xy_points, z=DRAWING_PLANE_Z):
    """Convert XY points to XYZ points on the drawing plane."""
    xy_points = np.asarray(xy_points, dtype=float)
    return np.column_stack((xy_points, np.full(len(xy_points), z)))


def world_points_to_joint_trajectory(
    world_xyz_points,
    robot=None,
    orientation_rotvec=(np.pi, 0.05, 0.05),
    ik_solution="elbow_up_2",
):
    """Convert world XYZ waypoints to a joint trajectory using UR10e IK."""
    robot = robot or UR10e()
    joint_targets = []

    for x, y, z in np.asarray(world_xyz_points, dtype=float):
        T_world_tool = pose6_to_T([x, y, z, *orientation_rotvec])
        joints = robot.IK(ik_solution, T_world_tool)
        joint_targets.append(joints)

    return np.asarray(joint_targets)


def build_joint_trajectory_from_maze(
    maze_image,
    start_pixel,
    goal_pixel,
    meters_per_pixel,
    T_world_maze,
    robot=None,
):
    """Full maze-image-to-joint-trajectory pipeline for implemented stages."""
    spline_pixels = plan_spline_pixel_path(maze_image, start_pixel, goal_pixel)
    if spline_pixels is None:
        return None

    local_xy = pixel_points_to_lengths(spline_pixels, meters_per_pixel)
    world_xy = local_maze_to_world(local_xy, T_world_maze)
    world_xyz = add_drawing_plane_z(world_xy)
    return world_points_to_joint_trajectory(world_xyz, robot=robot)


def run_mpc_background(shared_state, mpc_horizon, status_callback=None):
    """Start MPC once a full joint trajectory window is available."""
    sim_thread = None
    last_status = None

    try:
        while True:
            with shared_state.lock:
                following = shared_state.following_trajectory
                traj_len = len(shared_state.trajectory_queue)
                shutdown = shared_state.shutdown

            if shutdown:
                break

            if following:
                if sim_thread is None:
                    if traj_len >= mpc_horizon + 1:
                        msg = "MPC: starting trajectory tracking"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg

                        sim_thread = MPCSimulationThread(
                            shared_state=shared_state,
                            mpc_horizon=mpc_horizon,
                            dt=1 / SAMPLING_RATE,
                            workspace_offset=WORKSPACE_OFFSET,
                            vj=VJ,
                            aj=AJ,
                        )
                        sim_thread.start()
                    else:
                        time.sleep(0.01)
                elif sim_thread.is_alive():
                    time.sleep(0.05)
                else:
                    if sim_thread.status != "completed":
                        shared_state.stop_following()
                        print(f"[MPC] Error: {sim_thread.status} - {sim_thread.error_msg}")
                    sim_thread = None
            else:
                if sim_thread is not None and sim_thread.is_alive():
                    sim_thread.join(timeout=2)
                sim_thread = None
                time.sleep(0.01)

    except Exception as e:
        print(f"[MPC] Error: {e}")


def main():
    shared_state = SharedTrajectoryState()

    mpc_thread = threading.Thread(
        target=run_mpc_background,
        args=(shared_state, MPC_HORIZON),
        daemon=True,
    )
    mpc_thread.start()

    urx_thread = None
    if HAS_URX:
        urx_thread = URXControlThread(
            shared_state=shared_state,
            robot_ip=ROBOT_IP,
            hz=100,
            vj=VJ,
            aj=AJ,
            joint_pos_limits=JOINT_POS_LIMITS,
            min_link_dist=MIN_LINK_DISTANCE,
        )
        urx_thread.start()
    else:
        print("[WARN] urx not installed -- running without robot arm")

    # TODO: Hook these stages to the real camera, ArUco scale calculation,
    # start/goal detection, and maze-to-world calibration.
    print("Maze/MPC orchestrator started. Pipeline integration placeholders remain.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        with shared_state.lock:
            shared_state.shutdown = True
        if urx_thread is not None:
            urx_thread.stop()
            urx_thread.join(timeout=2)


if __name__ == "__main__":
    main()
