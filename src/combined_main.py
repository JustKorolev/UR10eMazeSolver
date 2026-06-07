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

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import cv2
import numpy as np

from src.astar import (
    astar,
    build_spline,
    create_nodes,
    detect_openings,
    largest_free_component,
    save_plan_overlay,
    save_spline_overlay,
)
from src.ur10e import T_TOOL_PEN, UR10e
from src.utils import pose6_to_T

try:
    from src.urx_control_thread import URXControlThread
    HAS_URX = True
except ImportError:
    HAS_URX = False


SAMPLING_RATE = 75  # Hz
MPC_HORIZON = SAMPLING_RATE // 12

WORKSPACE_OFFSET = pose6_to_T([0, -0.8, 0.1, np.pi, 0.01, 0.01])
DRAWING_PLANE_Z = float(WORKSPACE_OFFSET[2, 3])
DRAWING_TOOL_ORIENTATION = (np.pi, 0.01, 0.01)

VJ = 0.3  # rad/s
AJ = 1.2  # rad/s^2

JOINT_POS_LIMITS = np.array([6.1087, 6.1087, 6.1087, 6.1087, 6.1087, 6.1087])
MIN_LINK_DISTANCE = 0.05
ROBOT_IP = "192.168.0.2"
DEFAULT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "myphoto.png")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ENDPOINT_HEIGHT_AXIS = 2  # 0=x, 1=y, 2=z in the robot base frame.
ENDPOINT_HEIGHT_MARGIN = 0.0
APPROACH_HEIGHT_M = 0.1
MOVE_TIMEOUT_S = 30.0
MOVE_SETTLE_S = 1.0
APRILTAG_SIDE_M = 0.0428625
RECTIFICATION_PPM = 1500.0
ANCHOR_TAG_ID = 8

# Physical choreography and calibration constants.
CAMERA_JOINTS = np.deg2rad([90.0, -93.0, 103.0, -99.0, -90.0, 0.0])

# If maze_localizer anchors its "world" frame to a tag, set this to that tag's
# pose in the UR base frame. p_base = T_BASE_TAG @ p_tag.
T_BASE_TAG = np.eye(4, dtype=float)
T_BASE_TAG[:3, 3] = [
    13.25 * 0.0254,
    -36.75 * 0.0254,
    0.0,
]

# Optional fixed correction from the detected tag/world frame to the maze frame.
# Use this if you measure the tag-to-maze transform manually instead of trusting
# the localizer's saved T_world_maze directly.
USE_MANUAL_TAG_TO_MAZE = True
T_TAG_MAZE_CORRECTION = np.eye(4, dtype=float)
T_TAG_MAZE_CORRECTION[:3, 3] = [
    -1.625 * 0.0254,
    -0.25 * 0.0254,
    0.0,
]

# Set True when maze +x (image columns to the right) should map to negative
# robot-base x. This is applied in the maze frame before the base transform.
MAZE_X_AXIS_POINTS_NEGATIVE_BASE_X = True


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
        self.motion_requested = False
        self.motion_target_joints = None
        self.motion_label = ""
        self.motion_in_progress = False
        self.motion_done = False
        self.motion_error = None

        self.collision_detected = False
        self.collision_reason = ""

        self.robot_model = UR10e(workspace_offset=workspace_offset)
        self._collision_robot = self.robot_model
        self.home_joints = self.robot_model.IK("elbow_up_1", workspace_offset)
        self._workspace_z = float(workspace_offset[2, 3])
        self._workspace_endpoint_min = float(workspace_offset[ENDPOINT_HEIGHT_AXIS, 3])
        from src.cbf import CBFSafetyFilter, EndpointHeightCBF

        self.cbf_filter = CBFSafetyFilter(
            constraints=[
                EndpointHeightCBF(
                    robot=self.robot_model,
                    min_height=self._workspace_endpoint_min,
                    axis=ENDPOINT_HEIGHT_AXIS,
                    margin=ENDPOINT_HEIGHT_MARGIN,
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

    def request_joint_move(self, classical_joint_radians, label="move"):
        """Ask the URX thread to run a blocking movej to classical robot joints."""
        q = np.asarray(classical_joint_radians, dtype=float).reshape(6,)
        with self.lock:
            self.motion_target_joints = q.copy()
            self.motion_label = str(label)
            self.motion_requested = True
            self.motion_in_progress = False
            self.motion_done = False
            self.motion_error = None
            self.robot_enabled = False

    def wait_for_motion(self, timeout_s=MOVE_TIMEOUT_S, raise_on_error=False):
        start = time.time()
        while True:
            with self.lock:
                done = self.motion_done
                error = self.motion_error
                label = self.motion_label

            if done:
                if error:
                    msg = f"Robot move '{label}' failed: {error}"
                    if raise_on_error:
                        raise RuntimeError(msg)
                    print(f"[WARN] {msg}")
                    return False
                return True

            if time.time() - start > timeout_s:
                msg = f"Timed out waiting for robot move '{label}'"
                if raise_on_error:
                    raise TimeoutError(msg)
                print(f"[WARN] {msg}")
                return False

            time.sleep(0.05)

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
    """Capture one RealSense color frame and return it as a BGR numpy array."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for --capture") from exc

    color_w, color_h, fps = 1280, 720, 30
    warmup_frames = 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)

    pipeline.start(config)
    try:
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received from RealSense")
        return np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


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


def plan_maze_opening_path(
    maze_image,
    output_dir,
    N=100,
    samples_per_segment=20,
    control_point_stride=3,
):
    """Plan from the two detected maze openings and save debug overlays."""
    nodes = create_nodes(N, maze_image, obstacle_inflation_radius=4)
    interior = largest_free_component(nodes)
    if not interior:
        raise RuntimeError("No connected free maze interior found")

    openings = detect_openings(maze_image)
    print(f"[PLAN] Detected {len(openings)} boundary openings: {openings}")

    if len(openings) >= 2:
        def nearest_interior(pixel_xy):
            x, y = pixel_xy
            return min(interior, key=lambda node: (node.x - x) ** 2 + (node.y - y) ** 2)

        start_node = nearest_interior(openings[0])
        goal_node = nearest_interior(openings[1])
    else:
        print("[PLAN] Expected two openings; falling back to opposite interior corners")
        start_node = min(interior, key=lambda node: node.row + node.col)
        goal_node = max(interior, key=lambda node: node.row + node.col)

    print(
        "[PLAN] Start node "
        f"(row,col)=({start_node.row},{start_node.col}) "
        f"px=({start_node.x:.0f},{start_node.y:.0f})"
    )
    print(
        "[PLAN] Goal node  "
        f"(row,col)=({goal_node.row},{goal_node.col}) "
        f"px=({goal_node.x:.0f},{goal_node.y:.0f})"
    )

    path = astar(nodes, start_node, goal_node)
    if path is None:
        raise RuntimeError("A* could not find a path through the maze")

    if len(openings) >= 2:
        path = [openings[0]] + list(path) + [openings[1]]

    os.makedirs(output_dir, exist_ok=True)
    save_plan_overlay(maze_image, path, os.path.join(output_dir, "astar_overlay.png"))
    save_spline_overlay(
        maze_image,
        path,
        os.path.join(output_dir, "astar_spline_overlay.png"),
        samples_per_segment=samples_per_segment,
        control_point_stride=control_point_stride,
    )

    spline_pixels = build_spline(
        path,
        samples_per_segment=samples_per_segment,
        control_point_stride=control_point_stride,
    )
    print(f"[PLAN] Path found with {len(path)} A* points and {len(spline_pixels)} spline points")
    return spline_pixels


def local_maze_to_world(local_xy_points, T_world_maze):
    """Transform local maze XY points into robot world XY points."""
    points = np.asarray(local_xy_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("local_xy_points must have shape (num_points, 2)")

    homog = np.column_stack((points, np.zeros(len(points)), np.ones(len(points))))
    world = (np.asarray(T_world_maze, dtype=float).reshape(4, 4) @ homog.T).T
    return world[:, :2]


def apply_maze_axis_calibration(T_base_maze):
    """Apply fixed maze-axis calibration while keeping the top-left origin."""
    T = np.asarray(T_base_maze, dtype=float).reshape(4, 4).copy()
    if MAZE_X_AXIS_POINTS_NEGATIVE_BASE_X:
        T[:3, 0] *= -1.0
    return T


def compose_base_maze_transform(T_world_maze):
    """Compose p_base = T_base_maze @ p_maze from calibration constants."""
    T_tag_maze = (
        np.asarray(T_TAG_MAZE_CORRECTION, dtype=float).reshape(4, 4)
        if USE_MANUAL_TAG_TO_MAZE
        else (
            np.asarray(T_TAG_MAZE_CORRECTION, dtype=float).reshape(4, 4)
            @ np.asarray(T_world_maze, dtype=float).reshape(4, 4)
        )
    )
    T_base_maze = (
        np.asarray(T_BASE_TAG, dtype=float).reshape(4, 4)
        @ T_tag_maze
    )
    return apply_maze_axis_calibration(T_base_maze)


def add_drawing_plane_z(xy_points, z=DRAWING_PLANE_Z):
    """Convert XY points to XYZ points on the drawing plane."""
    xy_points = np.asarray(xy_points, dtype=float)
    return np.column_stack((xy_points, np.full(len(xy_points), z)))


def pixel_points_to_maze_lengths(pixel_points, image_shape, maze_w_m, maze_h_m):
    """Map rectified maze image pixels into the local maze frame in metres."""
    points = np.asarray(pixel_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixel_points must have shape (num_points, 2)")

    height, width = image_shape[:2]
    if width <= 1 or height <= 1:
        raise ValueError("maze image must be at least 2 pixels wide and tall")

    local = np.empty_like(points, dtype=float)
    local[:, 0] = points[:, 0] / (width - 1) * float(maze_w_m)
    local[:, 1] = points[:, 1] / (height - 1) * float(maze_h_m)
    return local


def world_points_to_joint_trajectory(
    world_xyz_points,
    robot=None,
    orientation_rotvec=DRAWING_TOOL_ORIENTATION,
    ik_solution="elbow_up_1",
    Ttp_pen=T_TOOL_PEN,
):
    """Convert world XYZ waypoints to a joint trajectory using UR10e IK."""
    robot = robot or UR10e()
    joint_targets = []

    for x, y, z in np.asarray(world_xyz_points, dtype=float):
        T_world_tool = pose6_to_T([x, y, z, *orientation_rotvec])
        joints = robot.IK(ik_solution, T_world_tool, Ttp_pen=Ttp_pen)
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

    local_xy = pixel_points_to_maze_lengths(
        spline_pixels,
        maze_image.shape,
        maze_image.shape[1] * meters_per_pixel,
        maze_image.shape[0] * meters_per_pixel,
    )
    world_xy = local_maze_to_world(local_xy, T_world_maze)
    world_xyz = add_drawing_plane_z(world_xy)
    return world_points_to_joint_trajectory(world_xyz, robot=robot)


def run_maze_localizer(image_path, output_dir):
    """Run the existing AprilTag/localization pipeline and return artifact paths."""
    annotated_path = os.path.join(output_dir, "localized_input.png")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "maze_localizer.py"),
        image_path,
        annotated_path,
        "--size",
        str(APRILTAG_SIDE_M),
        "--ppm",
        str(RECTIFICATION_PPM),
        "--outdir",
        output_dir,
    ]
    if ANCHOR_TAG_ID is not None:
        cmd.extend(["--anchor", str(ANCHOR_TAG_ID)])
    print("[LOCALIZE] Running maze_localizer.py")
    env = os.environ.copy()
    if os.path.isdir(VENDOR_DIR):
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = VENDOR_DIR if not existing else VENDOR_DIR + os.pathsep + existing
    subprocess.run(cmd, check=True, env=env)

    artifacts = {
        "maze_image": os.path.join(output_dir, "maze_rectified.png"),
        "meta": os.path.join(output_dir, "maze_frame.npz"),
        "annotated": annotated_path,
    }
    missing = [path for path in artifacts.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Localization did not produce expected files: {missing}")
    return artifacts


def build_joint_trajectory_from_localized_outputs(output_dir, robot=None, return_details=False):
    """Load localizer artifacts, plan the maze path, and convert it to IK targets."""
    maze_path = os.path.join(output_dir, "maze_rectified.png")
    meta_path = os.path.join(output_dir, "maze_frame.npz")

    maze_image = cv2.imread(maze_path, cv2.IMREAD_GRAYSCALE)
    if maze_image is None:
        raise FileNotFoundError(f"Could not read rectified maze image: {maze_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not read maze metadata: {meta_path}")

    meta = np.load(meta_path)
    T_world_maze = meta["T_world_maze"]
    T_base_maze = compose_base_maze_transform(T_world_maze)
    maze_w_m = float(meta["maze_w_m"])
    maze_h_m = float(meta["maze_h_m"])

    spline_pixels = plan_maze_opening_path(maze_image, output_dir)
    local_xy = pixel_points_to_maze_lengths(
        spline_pixels,
        maze_image.shape,
        maze_w_m,
        maze_h_m,
    )
    base_xy = local_maze_to_world(local_xy, T_base_maze)
    base_xyz = add_drawing_plane_z(base_xy)
    joint_trajectory = world_points_to_joint_trajectory(base_xyz, robot=robot)

    start_contact_xyz = base_xyz[0].copy()
    start_approach_xyz = start_contact_xyz.copy()
    start_approach_xyz[2] += APPROACH_HEIGHT_M
    start_contact_joints = world_points_to_joint_trajectory(
        [start_contact_xyz],
        robot=robot,
    )[0]
    start_approach_joints = world_points_to_joint_trajectory(
        [start_approach_xyz],
        robot=robot,
    )[0]

    if not np.all(np.isfinite(joint_trajectory)):
        raise RuntimeError("IK produced non-finite joint targets; check maze-to-robot calibration")

    np.save(os.path.join(output_dir, "spline_pixels.npy"), np.asarray(spline_pixels))
    np.save(os.path.join(output_dir, "local_maze_xy_m.npy"), local_xy)
    np.save(os.path.join(output_dir, "base_waypoints_xyz_m.npy"), base_xyz)
    np.save(os.path.join(output_dir, "joint_trajectory.npy"), joint_trajectory)
    np.save(os.path.join(output_dir, "T_base_maze.npy"), T_base_maze)
    np.save(os.path.join(output_dir, "start_approach_joints.npy"), start_approach_joints)
    np.save(os.path.join(output_dir, "start_contact_joints.npy"), start_contact_joints)

    print(f"[TRAJ] Saved joint trajectory with shape {joint_trajectory.shape}")
    if not return_details:
        return joint_trajectory

    return {
        "joint_trajectory": joint_trajectory,
        "base_waypoints_xyz": base_xyz,
        "local_maze_xy": local_xy,
        "spline_pixels": np.asarray(spline_pixels),
        "T_base_maze": T_base_maze,
        "start_approach_joints": start_approach_joints,
        "start_contact_joints": start_contact_joints,
    }


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

                        from src.trajectory_tracking import MPCSimulationThread

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Combined maze localization, planning, IK, and MPC entrypoint.")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_PATH,
        help="input maze image; ignored when --capture is used",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="capture a fresh overhead image from RealSense before localizing",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated localization, planning, and trajectory artifacts",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="after planning, start URX/MPC trajectory execution",
    )
    return parser.parse_args(argv)


def wait_for_robot_connected(shared_state, timeout_s=10.0):
    start = time.time()
    while True:
        with shared_state.lock:
            connected = shared_state.robot_connected
            shutdown = shared_state.shutdown

        if connected:
            return
        if shutdown:
            raise RuntimeError("Robot thread shut down before connecting")
        if time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for URX robot connection")
        time.sleep(0.05)


def modified_to_classical_joints(robot, q_modified):
    return robot.DHModifiedToClassical(np.asarray(q_modified, dtype=float).reshape(6,))


def request_and_wait_move(
    shared_state,
    classical_joints,
    label,
    timeout_s=MOVE_TIMEOUT_S,
    raise_on_error=False,
):
    shared_state.request_joint_move(classical_joints, label=label)
    ok = shared_state.wait_for_motion(
        timeout_s=timeout_s,
        raise_on_error=raise_on_error,
    )
    time.sleep(MOVE_SETTLE_S)
    return ok


def combined_main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    shared_state = None
    urx_thread = None
    robot_model = UR10e(workspace_offset=WORKSPACE_OFFSET)

    if args.execute:
        if not HAS_URX:
            raise RuntimeError("urx is not installed; cannot execute on the robot")

        shared_state = SharedTrajectoryState()
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
        wait_for_robot_connected(shared_state)
        request_and_wait_move(shared_state, CAMERA_JOINTS, "overhead camera pose")

    image_path = args.image
    if args.capture:
        print("[CAPTURE] Capturing overhead maze image")
        image = capture_maze_image()
        image_path = os.path.join(args.outdir, "captured_maze.png")
        cv2.imwrite(image_path, image)
        print(f"[CAPTURE] Saved {image_path}")

    run_maze_localizer(image_path, args.outdir)
    plan = build_joint_trajectory_from_localized_outputs(
        args.outdir,
        robot=robot_model,
        return_details=True,
    )
    joint_trajectory = plan["joint_trajectory"]

    if not args.execute:
        print("[DONE] Planning complete. Re-run with --execute to command the robot.")
        return joint_trajectory

    mpc_thread = threading.Thread(
        target=run_mpc_background,
        args=(shared_state, MPC_HORIZON),
        daemon=True,
    )
    mpc_thread.start()

    start_approach_classical = modified_to_classical_joints(
        robot_model,
        plan["start_approach_joints"],
    )
    start_contact_classical = modified_to_classical_joints(
        robot_model,
        plan["start_contact_joints"],
    )

    request_and_wait_move(shared_state, start_approach_classical, "above maze start")
    request_and_wait_move(shared_state, start_contact_classical, "maze start contact")

    shared_state.load_joint_trajectory(joint_trajectory)
    print("[EXECUTE] Maze/MPC orchestrator started")
    shared_state.start_following()

    try:
        while shared_state.following_trajectory:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shared_state.stop_following()
    finally:
        if shared_state is not None and urx_thread is not None:
            request_and_wait_move(shared_state, CAMERA_JOINTS, "return overhead camera pose")
            with shared_state.lock:
                shared_state.shutdown = True
            urx_thread.stop()
            urx_thread.join(timeout=2)

    return joint_trajectory


def main():
    combined_main()


if __name__ == "__main__":
    main()
