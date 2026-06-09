import os
import sys
import xml.sax.saxutils as xml_utils

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ur10e import T_TOOL_PEN, UR10e  # noqa: E402

INCH = 0.0254


class WorldToSvg:
    def __init__(self, x_min, x_max, y_min, y_max, width, height, margin):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)
        sx = (width - 2 * margin) / max(x_max - x_min, 1e-9)
        sy = (height - 2 * margin) / max(y_max - y_min, 1e-9)
        self.scale = min(sx, sy)
        self.x_mid = 0.5 * (x_min + x_max)
        self.y_mid = 0.5 * (y_min + y_max)
        self.px_mid = 0.5 * width
        self.py_mid = 0.5 * height

    def point(self, xy):
        x, y = np.asarray(xy, dtype=float)[:2]
        px = self.px_mid + (x - self.x_mid) * self.scale
        py = self.py_mid - (y - self.y_mid) * self.scale
        return px, py

    def points(self, points):
        return [self.point(p) for p in np.asarray(points, dtype=float)[:, :2]]


def svg_polyline(points, color, width=2, dash=None):
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{data}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" '
        f'stroke-linecap="round"{dash_attr}/>'
    )


def svg_circle(point, color, label=None, r=5):
    x, y = point
    out = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}"/>'
    if label:
        out += svg_text((x, y), label, color=color, dx=8, dy=-8)
    return out


def svg_text(point, text, color="#333", size=13, dx=8, dy=-8, anchor="start"):
    x, y = point
    safe_text = xml_utils.escape(str(text))
    return (
        f'<text x="{x + dx:.2f}" y="{y + dy:.2f}" font-size="{size}" '
        f'font-family="Arial" fill="{color}" text-anchor="{anchor}">{safe_text}</text>'
    )


def svg_line(p0, p1, color="#aaa", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" '
        f'x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def transform_points(T, xy_points):
    pts = np.asarray(xy_points, dtype=float)
    homog = np.column_stack((pts, np.zeros(len(pts)), np.ones(len(pts))))
    return (np.asarray(T, dtype=float).reshape(4, 4) @ homog.T).T[:, :3]


def nice_ticks(v_min, v_max, step=0.1):
    start = np.ceil(v_min / step) * step
    end = np.floor(v_max / step) * step
    if end < start:
        return []
    return np.arange(start, end + 0.5 * step, step)


def load_maze_corners(outdir):
    t_base_maze_path = os.path.join(outdir, "T_base_maze.npy")
    meta_path = os.path.join(outdir, "maze_frame.npz")
    if not (os.path.exists(t_base_maze_path) and os.path.exists(meta_path)):
        return None, None, None

    T_base_maze = np.load(t_base_maze_path)
    meta = np.load(meta_path)
    maze_w_m = float(meta["maze_w_m"])
    maze_h_m = float(meta["maze_h_m"])
    corners_local = np.array(
        [
            [0.0, 0.0],
            [maze_w_m, 0.0],
            [maze_w_m, maze_h_m],
            [0.0, maze_h_m],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    return transform_points(T_base_maze, corners_local), maze_w_m, maze_h_m


def fk_pen_xyz(robot, joints_mod):
    """FK the pen tip for an array of modified-DH joint vectors."""
    out = []
    for q_mod in np.asarray(joints_mod, dtype=float):
        q_class_deg = np.rad2deg(robot.DHModifiedToClassical(q_mod))
        T = robot.FK(q_class_deg, Ttp_pen=T_TOOL_PEN)
        out.append(T[:3, 3])
    return np.asarray(out)


def load_actual_run(outdir, robot):
    """FK the ACTUAL executed joints recorded during the run (mpc_trace.npz)."""
    trace_path = os.path.join(outdir, "mpc_trace.npz")
    if not os.path.exists(trace_path):
        return None
    trace = np.load(trace_path)
    if "q_meas" not in trace:
        return None
    q_meas = np.asarray(trace["q_meas"], dtype=float)
    if q_meas.ndim != 2 or q_meas.shape[0] < 2:
        return None
    return fk_pen_xyz(robot, q_meas)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "outputs")
    joint_path = os.path.join(outdir, "joint_trajectory.npy")
    ref_path = os.path.join(outdir, "base_waypoints_xyz_m.npy")
    svg_path = os.path.join(outdir, "fk_xy_diagnostic.svg")
    csv_path = os.path.join(outdir, "fk_xy_diagnostic.csv")

    joint_traj = np.load(joint_path)
    ref_xyz = np.load(ref_path)
    maze_corners_base, maze_w_m, maze_h_m = load_maze_corners(outdir)
    robot = UR10e()

    fk_xyz = fk_pen_xyz(robot, joint_traj)

    # ACTUAL robot run (FK of measured joints during execution), if available.
    actual_xyz = load_actual_run(outdir, robot)

    n = min(len(fk_xyz), len(ref_xyz))
    fk_xyz = fk_xyz[:n]
    ref_xyz = ref_xyz[:n]
    err = fk_xyz - ref_xyz
    xy_err = np.linalg.norm(err[:, :2], axis=1)
    xyz_err = np.linalg.norm(err, axis=1)

    print("FK trajectory diagnostic")
    print(f"  points: {n}")
    print(f"  reference start xyz: {ref_xyz[0]}")
    print(f"  FK start xyz:        {fk_xyz[0]}")
    print(f"  reference end xyz:   {ref_xyz[-1]}")
    print(f"  FK end xyz:          {fk_xyz[-1]}")
    print(f"  XY error mean/max:   {xy_err.mean():.6f} / {xy_err.max():.6f} m")
    print(f"  XYZ error mean/max:  {xyz_err.mean():.6f} / {xyz_err.max():.6f} m")
    print(f"  XY error mean/max:   {xy_err.mean()/INCH:.4f} / {xy_err.max()/INCH:.4f} in")

    if actual_xyz is not None:
        ref_dx = ref_xyz[:, 0].max() - ref_xyz[:, 0].min()
        ref_dy = ref_xyz[:, 1].max() - ref_xyz[:, 1].min()
        act_dx = actual_xyz[:, 0].max() - actual_xyz[:, 0].min()
        act_dy = actual_xyz[:, 1].max() - actual_xyz[:, 1].min()
        print("  --- ACTUAL run (FK of measured joints) ---")
        print(f"  actual points:       {len(actual_xyz)}")
        print(f"  actual start xyz:    {actual_xyz[0]}")
        print(f"  actual end xyz:      {actual_xyz[-1]}")
        print(f"  X extent  ref/actual: {ref_dx:.4f} / {act_dx:.4f} m  ratio={act_dx/ref_dx if ref_dx>1e-9 else float('nan'):.3f}")
        print(f"  Y extent  ref/actual: {ref_dy:.4f} / {act_dy:.4f} m  ratio={act_dy/ref_dy if ref_dy>1e-9 else float('nan'):.3f}")
    else:
        print("  (no mpc_trace.npz found -> actual run overlay omitted)")

    np.savetxt(
        csv_path,
        np.column_stack((ref_xyz, fk_xyz, err, xy_err, xyz_err)),
        delimiter=",",
        header=(
            "ref_x,ref_y,ref_z,fk_x,fk_y,fk_z,"
            "err_x,err_y,err_z,xy_err,xyz_err"
        ),
        comments="",
    )

    extra_xy = [ref_xyz[:, :2], fk_xyz[:, :2], np.array([[0.0, 0.0]])]
    if maze_corners_base is not None:
        extra_xy.append(maze_corners_base[:, :2])
    if actual_xyz is not None:
        extra_xy.append(actual_xyz[:, :2])
    all_xy = np.vstack(extra_xy)
    pad = 0.08
    x_min, y_min = all_xy.min(axis=0) - pad
    x_max, y_max = all_xy.max(axis=0) + pad

    width, height, margin = 1100, 850, 95
    mapper = WorldToSvg(x_min, x_max, y_min, y_max, width, height, margin)
    ref_px = mapper.points(ref_xyz[:, :2])
    fk_px = mapper.points(fk_xyz[:, :2])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="35" font-size="22" font-family="Arial" fill="#222">FK XY Diagnostic</text>',
        '<text x="30" y="60" font-size="14" font-family="Arial" fill="#555">Robot base frame top view. Grid spacing = 0.1 m = 3.94 in.</text>',
    ]

    for x in nice_ticks(x_min, x_max, 0.1):
        svg.append(svg_line(mapper.point((x, y_min)), mapper.point((x, y_max)), "#e8e8e8"))
        svg.append(svg_text((mapper.point((x, y_min))[0], height - margin + 22), f"{x:+.1f} m", "#666", 11, dx=-18, dy=0))
    for y in nice_ticks(y_min, y_max, 0.1):
        svg.append(svg_line(mapper.point((x_min, y)), mapper.point((x_max, y)), "#e8e8e8"))
        svg.append(svg_text((margin - 76, mapper.point((x_min, y))[1]), f"{y:+.1f} m", "#666", 11, dx=0, dy=4))

    if x_min <= 0.0 <= x_max:
        svg.append(svg_line(mapper.point((0, y_min)), mapper.point((0, y_max)), "#777", 2))
    if y_min <= 0.0 <= y_max:
        svg.append(svg_line(mapper.point((x_min, 0)), mapper.point((x_max, 0)), "#777", 2))

    robot_origin = mapper.point((0.0, 0.0))
    svg.append(svg_circle(robot_origin, "#111", "robot origin (base x=0,y=0)", r=6))
    svg.append(svg_line(robot_origin, mapper.point((0.1, 0.0)), "#111", 3))
    svg.append(svg_line(robot_origin, mapper.point((0.0, 0.1)), "#111", 3))
    svg.append(svg_text(mapper.point((0.1, 0.0)), "+robot x", "#111", 13, dx=8, dy=4))
    svg.append(svg_text(mapper.point((0.0, 0.1)), "+robot y", "#111", 13, dx=8, dy=-8))

    if maze_corners_base is not None:
        maze_px = mapper.points(maze_corners_base[:, :2])
        svg.append(svg_polyline(maze_px, "#2ca02c", width=2, dash="4 4"))
        svg.append(svg_circle(maze_px[0], "#2ca02c", "maze origin/top-left", r=5))
        svg.append(svg_text(maze_px[1], "maze +x corner", "#2ca02c", 12))
        svg.append(svg_text(maze_px[3], "maze +y corner", "#2ca02c", 12))
        svg.append(
            svg_text(
                maze_px[2],
                f"maze {maze_w_m:.3f} x {maze_h_m:.3f} m ({maze_w_m/INCH:.1f} x {maze_h_m/INCH:.1f} in)",
                "#2ca02c",
                12,
            )
        )

    legend_h = 99 if actual_xyz is not None else 72
    svg.extend(
        [
            svg_polyline(ref_px, "#1f77b4", width=3),
            svg_polyline(fk_px, "#d62728", width=2, dash="7 5"),
            svg_circle(ref_px[0], "#1f77b4", f"reference start [{ref_xyz[0,0]:+.3f}, {ref_xyz[0,1]:+.3f}] m", r=5),
            svg_text(ref_px[0], f"[{ref_xyz[0,0]/INCH:+.1f}, {ref_xyz[0,1]/INCH:+.1f}] in", "#1f77b4", 12, dx=8, dy=10),
            svg_circle(ref_px[-1], "#1f77b4", f"reference end [{ref_xyz[-1,0]:+.3f}, {ref_xyz[-1,1]:+.3f}] m", r=5),
            svg_text(ref_px[-1], f"[{ref_xyz[-1,0]/INCH:+.1f}, {ref_xyz[-1,1]/INCH:+.1f}] in", "#1f77b4", 12, dx=8, dy=10),
            svg_circle(fk_px[0], "#d62728", "FK start", r=4),
            svg_circle(fk_px[-1], "#d62728", "FK end", r=4),
        ]
    )

    if actual_xyz is not None:
        actual_px = mapper.points(actual_xyz[:, :2])
        svg.append(svg_polyline(actual_px, "#ff7f0e", width=2))
        svg.append(svg_circle(actual_px[0], "#ff7f0e", "actual start", r=4))
        svg.append(svg_circle(actual_px[-1], "#ff7f0e", "actual end", r=4))

    svg.append(f'<rect x="28" y="76" width="380" height="{legend_h}" fill="white" stroke="#ddd"/>')
    svg.append('<line x1="45" y1="98" x2="95" y2="98" stroke="#1f77b4" stroke-width="3"/>')
    svg.append('<text x="105" y="103" font-size="13" font-family="Arial" fill="#333">intended Cartesian waypoint path</text>')
    svg.append('<line x1="45" y1="125" x2="95" y2="125" stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"/>')
    svg.append('<text x="105" y="130" font-size="13" font-family="Arial" fill="#333">FK(planned joint trajectory) path</text>')
    if actual_xyz is not None:
        svg.append('<line x1="45" y1="152" x2="95" y2="152" stroke="#ff7f0e" stroke-width="2"/>')
        svg.append('<text x="105" y="157" font-size="13" font-family="Arial" fill="#333">ACTUAL robot run (FK of measured joints)</text>')

    svg.extend(
        [
            f'<text x="30" y="{height - 52}" font-size="13" font-family="Arial" fill="#444">XY error mean/max: {xy_err.mean():.6f} / {xy_err.max():.6f} m ({xy_err.mean()/INCH:.4f} / {xy_err.max()/INCH:.4f} in)</text>',
            f'<text x="30" y="{height - 32}" font-size="13" font-family="Arial" fill="#444">X range: {x_min:.3f}..{x_max:.3f} m, Y range: {y_min:.3f}..{y_max:.3f} m</text>',
            "</svg>",
        ]
    )

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    print(f"  wrote SVG: {svg_path}")
    print(f"  wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
