import argparse
import os
import sys
import xml.sax.saxutils as xml_utils
from collections import deque

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mpc import MPC  # noqa: E402
from src.simulation import _z_nullspace_project  # noqa: E402
from src.ur10e import UR10e  # noqa: E402


class OfflineSharedState:
    def __init__(self, joint_trajectory, horizon):
        self.mpc_horizon = int(horizon)
        self.trajectory_queue = deque(np.asarray(q, dtype=float).copy() for q in joint_trajectory)
        self.following_trajectory = True
        self.robot_enabled = True
        self.u_curr = np.zeros((6, 1))

    @property
    def trajectory_window(self):
        buf = list(self.trajectory_queue)
        n_need = self.mpc_horizon + 1
        if len(buf) == 0:
            return np.zeros((n_need, 6))
        if len(buf) >= n_need:
            return np.array(buf[:n_need])
        return np.array(buf + [buf[-1]] * (n_need - len(buf)))

    def consume_one(self):
        if len(self.trajectory_queue) > 1:
            self.trajectory_queue.popleft()
        else:
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6, 1))


def wrap_joints(q):
    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2 * np.pi) - np.pi


def svg_line(points, color, width=1.5, dash=None):
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{data}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" '
        f'stroke-linecap="round"{dash_attr}/>'
    )


def svg_text(x, y, text, color="#333", size=12):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Arial" fill="{color}">{xml_utils.escape(str(text))}</text>'
    )


def plot_series_svg(path, t, series, labels, title, ylabel, colors):
    t = np.asarray(t, dtype=float)
    y = np.asarray(series, dtype=float)
    width, height = 1200, 650
    left, right, top, bottom = 80, 30, 70, 80
    x_min, x_max = float(t.min()), float(t.max())
    y_min, y_max = float(y.min()), float(y.max())
    pad = max(0.05 * (y_max - y_min), 1e-6)
    y_min -= pad
    y_max += pad

    def map_point(x, val):
        px = left + (x - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)
        py = height - bottom - (val - y_min) / max(y_max - y_min, 1e-9) * (height - top - bottom)
        return px, py

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(30, 35, title, "#222", 22),
        svg_text(30, 58, f"x axis: time [s], y axis: {ylabel}", "#555", 13),
    ]

    for frac in np.linspace(0, 1, 6):
        x = x_min + frac * (x_max - x_min)
        p0 = map_point(x, y_min)
        p1 = map_point(x, y_max)
        svg.append(f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" stroke="#eee"/>')
        svg.append(svg_text(p0[0] - 15, height - bottom + 25, f"{x:.1f}", "#666", 11))
    for frac in np.linspace(0, 1, 7):
        val = y_min + frac * (y_max - y_min)
        p0 = map_point(x_min, val)
        p1 = map_point(x_max, val)
        svg.append(f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" stroke="#eee"/>')
        svg.append(svg_text(12, p0[1] + 4, f"{val:+.2f}", "#666", 11))

    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#777"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#777"/>')

    legend_y = 92
    for i, (label, color) in enumerate(zip(labels, colors)):
        pts = [map_point(tt, vv) for tt, vv in zip(t, y[:, i])]
        svg.append(svg_line(pts, color, width=1.7))
        lx = 90 + 120 * (i % 6)
        ly = legend_y + 20 * (i // 6)
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        svg.append(svg_text(lx + 34, ly + 4, label, "#333", 12))

    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description="Replay saved joint trajectory through MPC offline and plot velocity commands.")
    parser.add_argument("outdir", nargs="?", default=os.path.join(PROJECT_ROOT, "outputs"))
    parser.add_argument("--rate", type=float, default=75.0)
    parser.add_argument("--horizon", type=int, default=75 // 12)
    parser.add_argument("--vj", type=float, default=0.6)
    parser.add_argument("--aj", type=float, default=3.0)
    parser.add_argument("--workspace-z", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--disable-z-project", action="store_true")
    args = parser.parse_args()

    dt = 1.0 / args.rate
    joint_path = os.path.join(args.outdir, "joint_trajectory.npy")
    joint_ref = np.load(joint_path)
    max_steps = args.max_steps or len(joint_ref)

    shared = OfflineSharedState(joint_ref, args.horizon)
    model = UR10e(dt=dt)
    x_lim, u_lim, acc_u_lim = model.get_limits(args.vj, args.aj)
    controller = MPC(
        model=model,
        dynamics=model.model,
        param="P1",
        N=args.horizon,
        xlb=-x_lim,
        xub=x_lim,
        ulb=-u_lim,
        uub=u_lim,
        acc_ulb=-acc_u_lim,
        acc_uub=acc_u_lim,
        shared_state=shared,
    )

    x = joint_ref[0].reshape(6, 1)
    xs = [x.flatten()]
    us = []
    errors = []
    times = []

    for k in range(max_steps):
        if not shared.following_trajectory:
            break
        u_des, error = controller.mpc_controller(x, k * dt)
        u = np.asarray(u_des, dtype=float).reshape(6, 1)
        if not args.disable_z_project:
            u = _z_nullspace_project(model, x.flatten(), u, args.workspace_z)
        u = np.clip(u, -args.vj, args.vj)
        x = model.dynamics(x, u)
        x = wrap_joints(x).reshape(6, 1)
        shared.u_curr = u.copy()
        xs.append(x.flatten())
        us.append(u.flatten())
        errors.append(np.asarray(error, dtype=float).reshape(6,))
        times.append(k * dt)

    t = np.asarray(times)
    u_arr = np.asarray(us)
    x_arr = np.asarray(xs[:-1])
    err_arr = np.asarray(errors)
    speed_norm = np.linalg.norm(u_arr, axis=1)
    err_norm = np.linalg.norm(err_arr, axis=1)

    csv_path = os.path.join(args.outdir, "mpc_command_diagnostic.csv")
    np.savetxt(
        csv_path,
        np.column_stack((t, u_arr, speed_norm, err_arr, err_norm, x_arr)),
        delimiter=",",
        header=(
            "t,u1,u2,u3,u4,u5,u6,u_norm,"
            "e1,e2,e3,e4,e5,e6,e_norm,"
            "q1,q2,q3,q4,q5,q6"
        ),
        comments="",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    labels = [f"qdot{i+1}" for i in range(6)]
    vel_svg = os.path.join(args.outdir, "mpc_velocity_commands.svg")
    err_svg = os.path.join(args.outdir, "mpc_tracking_error.svg")
    norm_svg = os.path.join(args.outdir, "mpc_norms.svg")
    plot_series_svg(vel_svg, t, u_arr, labels, "MPC Joint Velocity Commands", "rad/s", colors)
    plot_series_svg(err_svg, t, err_arr, [f"e{i+1}" for i in range(6)], "MPC Joint Tracking Error", "rad", colors)
    plot_series_svg(
        norm_svg,
        t,
        np.column_stack((speed_norm, err_norm)),
        ["||u||", "||error||"],
        "MPC Command and Error Norms",
        "rad/s and rad",
        ["#111111", "#d62728"],
    )

    print("MPC command diagnostic")
    print(f"  simulated steps: {len(t)}")
    print(f"  dt: {dt:.6f} s, duration: {t[-1] if len(t) else 0:.3f} s")
    print(f"  velocity per-joint min: {np.min(u_arr, axis=0) if len(u_arr) else 'n/a'}")
    print(f"  velocity per-joint max: {np.max(u_arr, axis=0) if len(u_arr) else 'n/a'}")
    print(f"  ||u|| mean/max: {speed_norm.mean():.6f} / {speed_norm.max():.6f} rad/s")
    print(f"  ||error|| mean/max: {err_norm.mean():.6f} / {err_norm.max():.6f} rad")
    print(f"  wrote CSV: {csv_path}")
    print(f"  wrote velocity SVG: {vel_svg}")
    print(f"  wrote error SVG: {err_svg}")
    print(f"  wrote norms SVG: {norm_svg}")


if __name__ == "__main__":
    main()
