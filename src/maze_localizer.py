"""Localize the maze from an overhead image: detect tags -> homography -> crop.

This is the 2D pipeline (no 3D pose, no occupancy grid, no plane rectification).
It does exactly what maze_pipeline.py does -- find the four AprilTag corners,
warp the sheet flat with a homography, and crop tightly to the maze -- and then
writes the artifacts combined_main.py consumes:

    <outdir>/maze_rectified.png   the cropped, tags-removed maze (planning image)
    <outdir>/maze_map.png         same image (kept for the standalone astar.py)
    <outdir>/maze_frame.npz       metadata: T_world_maze, maze_w_m, maze_h_m, ...
    <output>                      annotated copy of the input (tags + maze bbox)

Metric scale (maze_w_m/maze_h_m) is recovered from the known AprilTag side
length measured in the rectified image, so no camera intrinsics are needed.

Usage:
    python maze_localizer.py <input> [output] [--size M] [--outdir DIR] [--anchor ID]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Reuse the exact detection + crop logic from the project-root pipeline.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import maze_pipeline as mp  # noqa: E402

# Default physical AprilTag edge length (meters); overridable via --size.
DEFAULT_MARKER_SIDE_M = 0.0428625


def meters_per_pixel_from_tags(tag_polys, marker_side_m):
    """Estimate metres-per-pixel from the tag squares in the rectified frame.

    Averages each tag's four edge lengths (in rectified pixels) across all tags
    and divides the known physical side length by it. The homography flattens
    the sheet, so this scale is uniform across the crop.
    """
    edge_px = []
    for poly in tag_polys:
        for i in range(4):
            edge_px.append(float(np.linalg.norm(poly[i] - poly[(i + 1) % 4])))
    mean_edge = float(np.mean(edge_px)) if edge_px else 0.0
    if mean_edge <= 1e-6:
        raise RuntimeError("Could not measure tag size in the rectified image.")
    return marker_side_m / mean_edge


def annotate(image, corners, ids, quad):
    """Draw detected tags (ID + corners) and the enclosing quad on the image."""
    out = image.copy()
    cv2.polylines(out, [quad.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    for c, tag_id in zip(corners, ids):
        cv2.polylines(out, [c.astype(np.int32)], True, (0, 165, 255), 2, cv2.LINE_AA)
        center = c.mean(axis=0).astype(int)
        cv2.drawMarker(out, tuple(center), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(out, f"ID {int(tag_id)}", (center[0] - 24, center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?",
                        default=os.path.join(_PROJECT_ROOT, "myphoto.png"),
                        help="input maze image")
    parser.add_argument("output", nargs="?", default=None,
                        help="annotated output image path")
    parser.add_argument("--size", type=float, default=DEFAULT_MARKER_SIDE_M,
                        help="AprilTag edge length in meters")
    parser.add_argument("--outdir", default=os.path.join(_PROJECT_ROOT, "outputs"),
                        help="directory for maze_rectified.png / maze_frame.npz")
    # Accepted for compatibility with combined_main.py's invocation (not needed
    # by the 2D pipeline, but harmless to receive).
    parser.add_argument("--ppm", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--anchor", type=int, default=None,
                        help="tag ID recorded as the maze-frame anchor (metadata only)")
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        root, ext = os.path.splitext(args.input)
        out_path = f"{root}_localized{ext}"
    os.makedirs(args.outdir, exist_ok=True)

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = mp.detect_tags(gray)
    if ids is None or len(ids) < 4:
        n = 0 if ids is None else len(ids)
        found = [] if ids is None else sorted(int(i) for i in ids)
        raise RuntimeError(f"Need 4 corner tags, only detected {n}: {found}")
    print(f"Detected {len(ids)} tags: IDs {sorted(int(i) for i in ids)}")

    # 1) Outer-corner quad -> homography -> flat full crop (tags included).
    outer_quad = mp.outer_quad_from_tags(corners)
    full_crop, H, (fw, fh) = mp.rectify_to_quad(image, outer_quad)
    print(f"Full crop (with tags): {fw} x {fh} px")

    # 2) Crop tightly to the maze (tags masked out).
    tag_polys = mp.warp_tag_polys(corners, H)
    x0, y0, x1, y1 = mp.maze_bbox(cv2.cvtColor(full_crop, cv2.COLOR_BGR2GRAY), tag_polys)
    maze_crop = full_crop[y0:y1 + 1, x0:x1 + 1]
    mh, mw = maze_crop.shape[:2]
    print(f"Maze-only crop: {mw} x {mh} px (bbox [{x0},{y0}]-[{x1},{y1}])")

    # 3) Metric scale from the known tag size (no intrinsics needed).
    mpp = meters_per_pixel_from_tags(tag_polys, args.size)
    maze_w_m = mw * mpp
    maze_h_m = mh * mpp
    print(f"Scale: {mpp*1000:.4f} mm/px -> maze {maze_w_m*1000:.1f} x {maze_h_m*1000:.1f} mm")

    # 4) Maze frame. Origin at the maze top-left corner, axes aligned to the
    # rectified image (x=cols, y=rows, z up). combined_main may override this
    # with a manual tag->maze calibration, so a clean identity-rotation frame is
    # the right neutral default.
    T_world_maze = np.eye(4, dtype=np.float64)
    anchor_id = args.anchor if args.anchor is not None else int(min(int(i) for i in ids))

    # --- Save artifacts -------------------------------------------------------
    maze_rect_path = os.path.join(args.outdir, "maze_rectified.png")
    maze_map_path = os.path.join(args.outdir, "maze_map.png")
    meta_path = os.path.join(args.outdir, "maze_frame.npz")

    cv2.imwrite(maze_rect_path, maze_crop)
    cv2.imwrite(maze_map_path, maze_crop)

    annotated = annotate(image, corners, [int(i) for i in ids], outer_quad)
    cv2.rectangle(annotated, tuple(outer_quad[0].astype(int)),
                  tuple(outer_quad[2].astype(int)), (0, 0, 255), 1)
    cv2.imwrite(out_path, annotated)

    np.savez(
        meta_path,
        T_world_maze=T_world_maze,
        maze_w_m=maze_w_m,
        maze_h_m=maze_h_m,
        meters_per_pixel=mpp,
        marker_side_m=args.size,
        anchor_id=anchor_id,
        maze_bbox=np.array([x0, y0, x1, y1]),
        crop_size=np.array([mw, mh]),
    )

    print("\nSaved:")
    print(f"  Rectified/cropped maze: {maze_rect_path}")
    print(f"  Maze map (for astar):   {maze_map_path}")
    print(f"  Maze-frame metadata:    {meta_path}")
    print(f"  Annotated input:        {out_path}")


if __name__ == "__main__":
    main()
