#!/usr/bin/env python3
"""Plot a physical robot run over the planned maze path overlay.

This script loads recorded robot joint values from a physical execution, uses
FK to compute the end-effector XY path in the robot base frame, projects that
path into maze image pixels using the saved maze calibration, and draws it on
an existing A* overlay if available.
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ur10e import T_TOOL_PEN, UR10e


def load_maze_metadata(outdir):
    t_base_maze = np.load(os.path.join(outdir, "T_base_maze.npy"))
    meta = np.load(os.path.join(outdir, "maze_frame.npz"))
    maze_w_m = float(meta["maze_w_m"])
    maze_h_m = float(meta["maze_h_m"])
    return t_base_maze, maze_w_m, maze_h_m


def base_xy_to_pixel(local_xy, image_shape, maze_w_m, maze_h_m):
    height, width = image_shape[:2]
    pixels = np.empty((len(local_xy), 2), dtype=np.float64)
    pixels[:, 0] = local_xy[:, 0] / max(maze_w_m, 1e-9) * (width - 1)
    pixels[:, 1] = local_xy[:, 1] / max(maze_h_m, 1e-9) * (height - 1)
    return pixels


def project_base_points_to_pixels(points_base, T_base_maze, image_shape, maze_w_m, maze_h_m):
    T_base_maze = np.asarray(T_base_maze, dtype=float)
    inv_T = np.linalg.inv(T_base_maze)
    pts = np.asarray(points_base, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points_base must have shape (N, 2) or (N, 3)")

    if pts.shape[1] == 2:
        z = np.zeros(len(pts))
    else:
        z = pts[:, 2]

    homog = np.column_stack((pts[:, 0], pts[:, 1], z, np.ones(len(pts))))
    local = (inv_T @ homog.T).T[:, :2]
    pixels = base_xy_to_pixel(local, image_shape, maze_w_m, maze_h_m)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, image_shape[1] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, image_shape[0] - 1)
    return pixels


def draw_path(image, points_px, color=(0, 255, 0), thickness=2):
    pts = np.asarray(points_px, dtype=np.int32)
    if len(pts) < 2:
        return image
    overlay = image.copy()
    cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)
    cv2.circle(overlay, tuple(pts[0]), 5, (0, 255, 255), -1)
    cv2.circle(overlay, tuple(pts[-1]), 5, (0, 128, 255), -1)
    return overlay


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot actual robot run over the maze path overlay.")
    parser.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "outputs"), help="output artifacts directory")
    parser.add_argument("--output", default="actual_run_overlay.png", help="output overlay image filename")
    parser.add_argument("--trace", default="robot_joint_trace.npy", help="recorded joint trace file")
    parser.add_argument("--overlay-image", default=None, help="optional existing overlay image to draw on")
    args = parser.parse_args(argv)

    outdir = os.path.abspath(args.outdir)
    trace_path = os.path.join(outdir, args.trace)
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Missing recorded trace: {trace_path}")

    joint_trace = np.load(trace_path)
    if joint_trace.ndim != 2 or joint_trace.shape[1] != 7:
        raise ValueError("robot_joint_trace.npy must have shape (N, 7) with time and 6 joint values")

    image_path = args.overlay_image or os.path.join(outdir, "astar_spline_overlay.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(outdir, "astar_overlay.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(outdir, "maze_rectified.png")
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            "Could not find any overlay or rectified maze image to draw on. "
            "Expected one of: astar_spline_overlay.png, astar_overlay.png, maze_rectified.png"
        )

    maze_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if maze_image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    t_base_maze, maze_w_m, maze_h_m = load_maze_metadata(outdir)
    robot = UR10e()

    fk_xyz = []
    for sample in joint_trace:
        q_rad = sample[1:]
        q_deg = np.rad2deg(q_rad)
        T = robot.FK(q_deg, T_TOOL_PEN)
        fk_xyz.append(T[:3, 3])
    fk_xyz = np.asarray(fk_xyz, dtype=float)

    fk_pixels = project_base_points_to_pixels(fk_xyz[:, :2], t_base_maze, maze_image.shape, maze_w_m, maze_h_m)
    overlay = draw_path(maze_image, fk_pixels, color=(0, 255, 0), thickness=2)

    out_path = os.path.join(outdir, args.output)
    cv2.imwrite(out_path, overlay)
    print(f"Saved physical run overlay: {out_path}")

    csv_path = os.path.join(outdir, "actual_run_xy.csv")
    np.savetxt(
        csv_path,
        np.column_stack((joint_trace[:, 0], fk_xyz[:, 0], fk_xyz[:, 1], fk_xyz[:, 2])),
        delimiter=",",
        header="time_s,x_m,y_m,z_m",
        comments="",
    )
    print(f"Saved FK XY trace CSV: {csv_path}")

    print(f"Recorded points: {len(fk_xyz)}")
    print(f"Overlay image used: {image_path}")


if __name__ == "__main__":
    main()
