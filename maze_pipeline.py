"""Simple, robust maze pipeline: 4 tags -> homography -> top-down maze crop.

This is a pure-2D pipeline (no camera intrinsics / 3D pose needed). It:

  1. Detects the 4 AprilTag (36h11) markers at the maze corners.
  2. Takes each tag's OUTER corner (farthest from the tag cluster center) to
     form a quad enclosing the whole sheet -> warps it to a flat rectangle
     (pipeline_crop.png). This still includes the tags in the corners.
  3. Masks the tag corners, finds the maze's wall content, and crops tightly to
     it so only the maze remains (pipeline_maze.png).

Usage:
    python maze_pipeline.py [input_image] [--outdir outputs]

Defaults to myphoto.png next to this script.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

# The maze sheet uses the AprilTag 36h11 family (confirmed by detection probe).
ARUCO_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11


def make_detector_params():
    """Detector parameters tuned to recover low-contrast / blurry corner tags.

    The wide adaptive-threshold sweep (large WinSizeMax, small step) is what lets
    all four corner tags survive uneven lighting / glare across captures.
    """
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 101
    p.adaptiveThreshWinSizeStep = 2
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.aprilTagQuadDecimate = 0.0
    return p


def detect_tags(gray):
    """Detect markers. Returns (corners, ids) where corners is a list of (4,2)."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID), make_detector_params()
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [], None
    corners = [c.reshape(4, 2).astype(np.float64) for c in corners]
    return corners, ids.flatten()


def order_corners_clockwise(pts):
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float64)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def outer_quad_from_tags(corners):
    """Quad from each tag's outermost corner (farthest from the cluster center).

    This encloses the whole sheet so the entire maze is guaranteed to be inside.
    """
    centers = np.array([c.mean(axis=0) for c in corners])
    cluster_center = centers.mean(axis=0)
    pts = []
    for c in corners:
        dists = np.linalg.norm(c - cluster_center, axis=1)
        pts.append(c[np.argmax(dists)])
    return order_corners_clockwise(np.array(pts))


def rectify_to_quad(image, quad):
    """Warp the source quad to an axis-aligned rectangle. Returns (warp, H, size)."""
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(width, 1)
    height = max(height, 1)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(image, H, (width, height))
    return warp, H, (width, height)


def warp_tag_polys(tag_corners, H):
    """Map each tag's 4-corner polygon into the warped image frame."""
    polys = []
    for c in tag_corners:
        p = cv2.perspectiveTransform(c.reshape(1, 4, 2).astype(np.float32), H)
        polys.append(p.reshape(4, 2))
    return polys


def maze_bbox(warp_gray, tag_polys, pad_frac=0.005):
    """Find the maze's tight bounding box inside the full crop.

    The maze is the dense grid of dark lines. We threshold dark pixels, erase
    the tag corners, blur into a density map, and take the bounding box of the
    largest dense connected component (the maze). Returns (x0, y0, x1, y1).
    """
    h, w = warp_gray.shape
    blk = max(3, (int(min(h, w) * 0.04) | 1))  # odd, ~maze-cell scale
    walls = cv2.adaptiveThreshold(
        warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 10
    )
    # Erase the tag squares (slightly padded) so they don't extend the bbox.
    for poly in tag_polys:
        c = poly.mean(axis=0)
        padded = (poly + (poly - c) * 0.25).astype(np.int32)
        cv2.fillConvexPoly(walls, padded, 0)

    win = max(15, (int(min(h, w) * 0.06) | 1))
    density = cv2.blur((walls > 0).astype(np.float32), (win, win))
    dense = (density >= 0.12).astype(np.uint8) * 255
    dense = cv2.morphologyEx(
        dense, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    )

    num, _, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    if num <= 1:
        return 0, 0, w - 1, h - 1
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]
    bh = stats[best, cv2.CC_STAT_HEIGHT]
    px = int(round(pad_frac * bw))
    py = int(round(pad_frac * bh))
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w - 1, x + bw - 1 + px)
    y1 = min(h - 1, y + bh - 1 + py)
    return x0, y0, x1, y1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=os.path.join(here, "myphoto.png"))
    parser.add_argument("--outdir", default=os.path.join(here, "outputs"))
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_tags(gray)
    if ids is None or len(ids) < 4:
        n = 0 if ids is None else len(ids)
        raise RuntimeError(f"Need 4 corner tags, only detected {n}: "
                           f"{[] if ids is None else sorted(int(i) for i in ids)}")
    print(f"Detected {len(ids)} tags: IDs {sorted(int(i) for i in ids)}")

    outer_quad = outer_quad_from_tags(corners)
    full_crop, H, (fw, fh) = rectify_to_quad(image, outer_quad)
    print(f"Full crop (with tags): {fw} x {fh} px")

    tag_polys = warp_tag_polys(corners, H)
    x0, y0, x1, y1 = maze_bbox(cv2.cvtColor(full_crop, cv2.COLOR_BGR2GRAY), tag_polys)
    maze_crop = full_crop[y0:y1 + 1, x0:x1 + 1]
    print(f"Maze-only crop:        {maze_crop.shape[1]} x {maze_crop.shape[0]} px "
          f"(bbox [{x0},{y0}]-[{x1},{y1}])")

    debug = full_crop.copy()
    cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 0, 255), 2)

    paths = {
        "full crop":      os.path.join(args.outdir, "pipeline_crop.png"),
        "maze bbox":      os.path.join(args.outdir, "pipeline_quad.png"),
        "maze-only crop": os.path.join(args.outdir, "pipeline_maze.png"),
        "maze map":       os.path.join(args.outdir, "maze_map.png"),
    }
    cv2.imwrite(paths["full crop"], full_crop)
    cv2.imwrite(paths["maze bbox"], debug)
    cv2.imwrite(paths["maze-only crop"], maze_crop)
    cv2.imwrite(paths["maze map"], maze_crop)

    print("\nSaved:")
    for label, p in paths.items():
        print(f"  {label:16s}: {p}")


if __name__ == "__main__":
    main()
