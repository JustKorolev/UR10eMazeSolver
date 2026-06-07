"""Detect AprilTag/ArUco markers in a RealSense image and localize them in 3D.

This mirrors the lab3 pipeline (detect markers -> estimate pose relative to
camera -> transform into a global frame), adapted for the Intel RealSense D435I.

Key differences vs lab3:
  * Intrinsics come straight from the RealSense factory calibration (queried
    live over USB, with a cached fallback baked in below). The color stream is
    already rectified, so distortion coefficients are zero.
  * There is no robot arm here, so the "global/world" frame is anchored to one
    of the detected tags (default: the lowest detected ID) instead of the robot
    base. The transform chain is otherwise identical:
        lab3:  T_base_marker  = T_base_cam      @ T_cam_marker
        here:  T_world_marker = inv(T_cam_world) @ T_cam_marker

Usage:
    python maze_localizer.py [input_image] [output_image]
    python maze_localizer.py myphoto.png annotated.png --anchor 1

Defaults to ../myphoto.png (the test capture) if no input is given.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import numpy.typing as npt

# --- Tag / camera configuration ------------------------------------------------

# The maze sheet uses the AprilTag 36h11 family (confirmed by detection probe).
ARUCO_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11

# Physical edge length of the black tag square, in meters (1.6875 in).
MARKER_SIDE_M = 0.0428625

# Cached RealSense D435I color intrinsics @ 1280x720 (factory calibration).
# Used as a fallback when the camera is not connected / is busy.
FALLBACK_K = np.array(
    [
        [910.7160034179688, 0.0, 646.0648193359375],
        [0.0, 910.2683715820312, 367.39739990234375],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# Color stream is rectified by the SDK -> no distortion.
FALLBACK_DIST = np.zeros((5, 1), dtype=np.float64)
FALLBACK_RES = (1280, 720)


# --- Intrinsics ----------------------------------------------------------------

def get_realsense_intrinsics(width: int = 1280, height: int = 720):
    """Query live color intrinsics from a connected RealSense.

    Returns (K, dist, (w, h)) or None if no camera is available.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        try:
            intr = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            K = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)
            return K, dist, (intr.width, intr.height)
        finally:
            pipeline.stop()
    except Exception:
        return None


def resolve_intrinsics(image_shape):
    """Pick intrinsics: live RealSense if available, else cached fallback.

    Scales the intrinsics if the image resolution differs from the calibration
    resolution (e.g. a downscaled photo).
    """
    h, w = image_shape[:2]
    live = get_realsense_intrinsics(*FALLBACK_RES)
    if live is not None:
        K, dist, (cw, ch) = live
        source = "RealSense (live)"
    else:
        K, dist, (cw, ch) = FALLBACK_K.copy(), FALLBACK_DIST.copy(), FALLBACK_RES
        source = "RealSense (cached fallback)"

    if (w, h) != (cw, ch):
        sx, sy = w / cw, h / ch
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        source += f" [scaled {cw}x{ch} -> {w}x{h}]"
    return K, dist, source


# --- Pose estimation -----------------------------------------------------------

def marker_object_points(side_m: float) -> npt.NDArray[np.float64]:
    """3D corners of a marker centered at origin, matching OpenCV corner order
    (top-left, top-right, bottom-right, bottom-left), z=0 plane."""
    half = side_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def estimate_pose(corners, K, dist, side_m):
    """Per-marker pose via solvePnP (IPPE_SQUARE) -> 4x4 T_cam_marker."""
    obj = marker_object_points(side_m)
    img = corners.reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T, rvec, tvec


def inv_T(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# --- Planar rectification (world plane z=0 -> top-down metric image) -----------
#
# The maze lies on the z=0 plane of the world frame (the tag plane). Because it
# is planar, the mapping world(X, Y) -> image pixel is an exact homography, so we
# can synthesize a perfectly top-down, metric view of the maze and work in it.

class PlaneRectifier:
    """Maps between world plane (X, Y, meters) and a top-down rectified image.

    Sampling direction is configurable so the output can be oriented upright
    regardless of how the world (tag) frame happens to be aligned:
        flip_x=False: col u increases with world X;  True: decreases
        flip_y=False: row v increases with world Y;  True: decreases
    The mapping is exposed via rect_to_world / world_to_rect so all downstream
    geometry (tag masking, grid->world) stays consistent with the chosen flips.
    """

    def __init__(self, T_cam_world, K, dist, x_range, y_range, ppm,
                 flip_x=False, flip_y=False):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.ppm = float(ppm)
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.width = max(1, int(round((self.x_max - self.x_min) * ppm)))
        self.height = max(1, int(round((self.y_max - self.y_min) * ppm)))

        R = T_cam_world[:3, :3]
        self._rvec, _ = cv2.Rodrigues(R)
        self._tvec = T_cam_world[:3, 3].reshape(3, 1)
        self._K = K
        self._dist = dist

        world_corners = np.array(
            [[self.x_min, self.y_min], [self.x_max, self.y_min],
             [self.x_max, self.y_max], [self.x_min, self.y_max]],
            dtype=np.float64,
        )
        src = self.world_to_image(world_corners).astype(np.float32)
        dst = np.array([self.world_to_rect(x, y) for x, y in world_corners],
                       dtype=np.float32)
        self._H_img_to_rect = cv2.getPerspectiveTransform(src, dst)

    def world_to_image(self, world_xy):
        """World (X, Y) [z=0] -> source image pixels (Nx2)."""
        pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
        pts3 = np.hstack([pts, np.zeros((len(pts), 1))])
        img, _ = cv2.projectPoints(pts3, self._rvec, self._tvec, self._K, self._dist)
        return img.reshape(-1, 2)

    def rectify(self, image):
        return cv2.warpPerspective(image, self._H_img_to_rect, (self.width, self.height))

    def rect_to_world(self, u, v):
        x = (self.x_max - u / self.ppm) if self.flip_x else (self.x_min + u / self.ppm)
        y = (self.y_max - v / self.ppm) if self.flip_y else (self.y_min + v / self.ppm)
        return (x, y)

    def world_to_rect(self, x, y):
        u = (self.x_max - x) * self.ppm if self.flip_x else (x - self.x_min) * self.ppm
        v = (self.y_max - y) * self.ppm if self.flip_y else (y - self.y_min) * self.ppm
        return (u, v)

    def maze_axes_world(self):
        """Unit directions of the rectified +u (col) and +v (row) axes in the
        world frame. Used to build a consistent maze-frame rotation."""
        x_axis = np.array([-1.0 if self.flip_x else 1.0, 0.0, 0.0])
        y_axis = np.array([0.0, -1.0 if self.flip_y else 1.0, 0.0])
        z_axis = np.cross(x_axis, y_axis)  # keep a right-handed frame
        return x_axis, y_axis, z_axis


def overview_region_from_tags(world_coords, margin_frac=0.55):
    """Bounding world region (x_range, y_range) covering all tags + margin."""
    pts = np.array([wc[:2] for wc in world_coords.values()], dtype=np.float64)
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 0.05)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    half = span * (0.5 + margin_frac)
    return (cx - half, cx + half), (cy - half, cy + half)


def mask_out_tags(rect_gray, rectifier, world_coords, side_m, pad=1.8):
    """Paint white squares over the tag locations so they aren't mistaken for
    maze structure during detection."""
    out = rect_gray.copy()
    r = int(round(side_m * pad * rectifier.ppm / 2.0))
    for wc in world_coords.values():
        u, v = rectifier.world_to_rect(wc[0], wc[1])
        u, v = int(round(u)), int(round(v))
        cv2.rectangle(out, (u - r, v - r), (u + r, v + r), 255, -1)
    return out


def _make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def adaptive_wall_mask(gray, block_px, C=10):
    """Extract maze walls (dark lines) as a 0/255 mask using local adaptive
    thresholding. Robust to uneven paper illumination, which defeats a single
    global threshold. block_px is the local neighborhood size in pixels."""
    b = max(3, _make_odd(block_px))
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b, C
    )


def _largest_component_bbox(binary):
    """(u0, v0, u1, v1, mask) of the largest connected component in a 0/255 mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    w = stats[best, cv2.CC_STAT_WIDTH]
    h = stats[best, cv2.CC_STAT_HEIGHT]
    return int(x), int(y), int(x + w - 1), int(y + h - 1), (labels == best)


def detect_maze_bbox(rect_gray, rectifier, world_coords, side_m,
                     block_px=45, adaptive_c=10, density_thresh=0.12,
                     pad_frac=0.01):
    """Locate the maze square in the rectified, tag-masked image.

    The maze is distinguished from the wood grain / tape / text background by
    WALL DENSITY: it is a dense, regular grid of lines, whereas the background
    has only sparse edges. Steps:
      1. Adaptive-threshold to get all wall/edge pixels (illumination robust).
      2. Restrict to the band spanning the two tags (excludes title/copyright).
      3. Blur the wall mask into a local density map and keep dense regions.
      4. The maze = largest dense connected component -> its bounding box.
    Returns (u0, v0, u1, v1) inclusive pixel bounds.
    """
    walls = (adaptive_wall_mask(rect_gray, block_px, adaptive_c) > 0)

    # Band between the tags along their dominant separation axis.
    tag_uv = [rectifier.world_to_rect(wc[0], wc[1]) for wc in world_coords.values()]
    us = [p[0] for p in tag_uv]
    vs = [p[1] for p in tag_uv]
    band = np.zeros(rect_gray.shape, dtype=bool)
    if (max(vs) - min(vs)) >= (max(us) - min(us)):
        band[int(min(vs)):int(max(vs)) + 1, :] = True
    else:
        band[:, int(min(us)):int(max(us)) + 1] = True
    walls = walls & band

    # Local wall density: the maze is consistently dense; wood grain is sparse.
    win = max(15, _make_odd(int(2 * block_px)))
    density = cv2.blur(walls.astype(np.float32), (win, win))
    dense = (density >= density_thresh).astype(np.uint8) * 255
    dense = cv2.morphologyEx(
        dense, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (win, win)),
    )

    comp = _largest_component_bbox(dense)
    if comp is None:
        return None
    u0, v0, u1, v1, _ = comp

    # Pad outward a touch so the maze's outer border isn't clipped, clamped to
    # the rectified image bounds.
    h, w = rect_gray.shape
    pad_u = int(round(pad_frac * (u1 - u0)))
    pad_v = int(round(pad_frac * (v1 - v0)))
    u0 = max(0, u0 - pad_u)
    v0 = max(0, v0 - pad_v)
    u1 = min(w - 1, u1 + pad_u)
    v1 = min(h - 1, v1 + pad_v)
    return u0, v0, u1, v1


# --- Occupancy grid ------------------------------------------------------------

def build_occupancy_grid(maze_gray, grid_n, block_px=45, adaptive_c=10,
                         occ_fraction=0.2, wall_dilate_px=1):
    """Downsample a top-down maze image into an N x N boolean occupancy grid.

    Walls are extracted with adaptive thresholding (handles uneven lighting).
    Because walls are thin (a couple of pixels), naive per-cell averaging leaves
    broken/choppy walls when a wall straddles a cell boundary, so we (1) extract
    walls, (2) lightly dilate them to fully cover the cells they pass through,
    then (3) mark a cell occupied when its wall fraction exceeds occ_fraction.
    """
    h, w = maze_gray.shape
    wall = (adaptive_wall_mask(maze_gray, block_px, adaptive_c) > 0).astype(np.uint8)
    if wall_dilate_px > 0:
        k = 2 * int(wall_dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wall = cv2.dilate(wall, kernel)

    occ = np.zeros((grid_n, grid_n), dtype=bool)
    for row in range(grid_n):
        y0 = int(round(row * h / grid_n))
        y1 = max(int(round((row + 1) * h / grid_n)), y0 + 1)
        for col in range(grid_n):
            x0 = int(round(col * w / grid_n))
            x1 = max(int(round((col + 1) * w / grid_n)), x0 + 1)
            patch = wall[y0:y1, x0:x1]
            occ[row, col] = patch.mean() >= occ_fraction
    return occ


def occupancy_to_image(occ, scale=8):
    """Render a boolean occupancy grid as an image (black=occupied, white=free)."""
    img = np.where(occ, 0, 255).astype(np.uint8)
    return cv2.resize(img, (occ.shape[1] * scale, occ.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def grid_cell_to_world(rectifier, bbox, grid_n, row, col):
    """Map an occupancy-grid cell center to a world-frame (X, Y) point."""
    u0, v0, u1, v1 = bbox
    w = u1 - u0
    h = v1 - v0
    u = u0 + (col + 0.5) * w / grid_n
    v = v0 + (row + 0.5) * h / grid_n
    return rectifier.rect_to_world(u, v)


# --- Annotation ----------------------------------------------------------------

def annotate(image, corners, ids, poses, K, dist, side_m, world_coords):
    out = image.copy()
    cv2.aruco.drawDetectedMarkers(out, corners, ids)
    axis_len = side_m * 0.75
    for i, marker_id in enumerate(ids.flatten()):
        T = poses[i]
        if T is None:
            continue
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape(3, 1)
        cv2.drawFrameAxes(out, K, dist, rvec, tvec, axis_len, 3)

        pts = corners[i].reshape(4, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        wc = world_coords[int(marker_id)]
        label_id = f"ID {int(marker_id)}"
        label_xyz = f"({wc[0]*1000:.0f},{wc[1]*1000:.0f},{wc[2]*1000:.0f})mm"
        cv2.putText(out, label_id, (cx - 50, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label_xyz, (cx - 90, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2, cv2.LINE_AA)
    return out


# --- Main ----------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(os.path.dirname(here), "myphoto.png")

    parser = argparse.ArgumentParser(description="Localize maze AprilTags in 3D.")
    parser.add_argument("input", nargs="?", default=default_in, help="input image")
    parser.add_argument("output", nargs="?", default=None, help="annotated output image")
    parser.add_argument("--anchor", type=int, default=None,
                        help="marker ID to use as world origin (default: lowest ID)")
    parser.add_argument("--size", type=float, default=MARKER_SIDE_M,
                        help="marker edge length in meters")
    parser.add_argument("--grid-size", type=int, default=100,
                        help="occupancy grid resolution (N x N cells)")
    parser.add_argument("--ppm", type=float, default=1500.0,
                        help="rectification resolution in pixels per meter")
    parser.add_argument("--adaptive-block", type=int, default=45,
                        help="adaptive-threshold neighborhood size in pixels "
                             "(odd; ~1.5x a maze cell)")
    parser.add_argument("--adaptive-c", type=int, default=10,
                        help="adaptive-threshold bias (higher = fewer wall pixels)")
    parser.add_argument("--density-threshold", type=float, default=0.12,
                        help="local wall-density to count as maze (vs background)")
    parser.add_argument("--tag-margin", type=float, default=0.0,
                        help="extra margin (m) added around the 4-tag bounding box")
    parser.add_argument("--occ-fraction", type=float, default=0.2,
                        help="wall-pixel fraction for a grid cell to count as occupied")
    parser.add_argument("--wall-dilate", type=int, default=1,
                        help="dilate walls by this many pixels before gridding "
                             "(reduces choppy/broken walls)")
    parser.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True,
                        help="invert the rectified Y axis so the maze is upright")
    parser.add_argument("--flip-x", action=argparse.BooleanOptionalAction, default=False,
                        help="invert the rectified X axis")
    parser.add_argument("--outdir", default=None,
                        help="directory for maze outputs (default: <project>/outputs)")
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        root, ext = os.path.splitext(args.input)
        out_path = f"{root}_localized{ext}"

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    K, dist, source = resolve_intrinsics(image.shape)
    print(f"Intrinsics source: {source}")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")
    print(f"Marker side length: {args.size*1000:.4f} mm\n")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    # More robust detection so all corner tags are found (some are blurrier /
    # at a steeper angle): subpixel corners, wider thresholding sweep, and a
    # smaller minimum marker size.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.05
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID), params,
    )
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("No markers detected.")
        cv2.imwrite(out_path, image)
        return

    ids_flat = ids.flatten()
    print(f"Detected {len(ids_flat)} markers: IDs {sorted(int(i) for i in ids_flat)}\n")

    poses = [estimate_pose(corners[i], K, dist, args.size)[0] for i in range(len(ids_flat))]

    # Choose the world-frame anchor tag.
    anchor_id = args.anchor if args.anchor is not None else int(min(ids_flat))
    id_to_index = {int(mid): i for i, mid in enumerate(ids_flat)}
    if anchor_id not in id_to_index:
        anchor_id = int(min(ids_flat))
    T_cam_world = poses[id_to_index[anchor_id]]
    T_world_cam = inv_T(T_cam_world)

    print(f"World frame anchored at marker ID {anchor_id}.\n")
    print("Per-marker results:")
    print("-" * 70)

    world_coords = {}
    for i, mid in enumerate(ids_flat):
        mid = int(mid)
        T_cam_marker = poses[i]
        T_world_marker = T_world_cam @ T_cam_marker
        cam_xyz = T_cam_marker[:3, 3]
        world_xyz = T_world_marker[:3, 3]
        world_coords[mid] = world_xyz

        print(f"Marker ID {mid}:")
        print(f"  Camera-frame xyz (m): "
              f"[{cam_xyz[0]:+.4f}, {cam_xyz[1]:+.4f}, {cam_xyz[2]:+.4f}]  "
              f"(distance {np.linalg.norm(cam_xyz):.3f} m)")
        print(f"  World-frame  xyz (m): "
              f"[{world_xyz[0]:+.4f}, {world_xyz[1]:+.4f}, {world_xyz[2]:+.4f}]")
        print(f"  T_cam_marker =\n{np.array2string(T_cam_marker, precision=4, suppress_small=True)}")
        print("-" * 70)

    if len(ids_flat) >= 2:
        sorted_ids = sorted(int(i) for i in ids_flat)
        a, b = sorted_ids[0], sorted_ids[1]
        d = np.linalg.norm(world_coords[a] - world_coords[b])
        print(f"\nDistance between tag {a} and tag {b}: {d*1000:.1f} mm "
              f"({d:.4f} m)  <- sanity-check against the printed maze width.")

    annotated = annotate(image, corners, ids, poses, K, dist, args.size, world_coords)
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {out_path}")

    # --- Maze rectification + occupancy grid ----------------------------------
    if len(world_coords) < 2:
        print("\nNeed at least 2 tags to bracket the maze; skipping occupancy grid.")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(os.path.dirname(here), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 70)
    print("MAZE RECTIFICATION + OCCUPANCY GRID")
    print("=" * 70)

    # Decide how to bound the maze. With >=3 tags that span a 2D box (i.e. one
    # near each corner) we can use their bounding box directly. Otherwise fall
    # back to wall-density detection (the 2-tags-on-opposite-edges case).
    #
    # Use the tags' OUTER CORNERS (all four corners of every tag projected into
    # the world plane), not just their centers, so the box reaches the maze's
    # true extent and captures the full map.
    obj_corners = marker_object_points(args.size)  # 4x3, marker frame
    tag_world_corners = []
    for i in range(len(ids_flat)):
        T_world_marker = T_world_cam @ poses[i]
        ch = np.hstack([obj_corners, np.ones((4, 1))])
        cw = (T_world_marker @ ch.T).T[:, :2]
        tag_world_corners.append(cw)
    all_corners = np.vstack(tag_world_corners)

    spread = all_corners.max(axis=0) - all_corners.min(axis=0)
    tags_form_box = len(world_coords) >= 3 and spread.min() > 4.0 * args.size

    if tags_form_box:
        m = args.tag_margin
        x_range = (float(all_corners[:, 0].min() - m), float(all_corners[:, 0].max() + m))
        y_range = (float(all_corners[:, 1].min() - m), float(all_corners[:, 1].max() + m))
        rectifier = PlaneRectifier(T_cam_world, K, dist, x_range, y_range, args.ppm,
                                   flip_x=args.flip_x, flip_y=args.flip_y)
        overview = rectifier.rectify(image)
        overview_gray = cv2.cvtColor(overview, cv2.COLOR_BGR2GRAY)
        # The maze region IS the tag bounding box; mask the corner tags so they
        # don't pollute the occupancy grid.
        maze_gray = mask_out_tags(overview_gray, rectifier, world_coords, args.size)
        u0, v0 = 0, 0
        u1, v1 = overview_gray.shape[1] - 1, overview_gray.shape[0] - 1
        bbox = (u0, v0, u1, v1)
        print(f"Maze bounded by {len(world_coords)} corner tags "
              f"(IDs {sorted(world_coords)}), using outer tag corners.")
    else:
        x_range, y_range = overview_region_from_tags(world_coords)
        rectifier = PlaneRectifier(T_cam_world, K, dist, x_range, y_range, args.ppm,
                                   flip_x=args.flip_x, flip_y=args.flip_y)
        overview = rectifier.rectify(image)
        overview_gray = cv2.cvtColor(overview, cv2.COLOR_BGR2GRAY)

        masked = mask_out_tags(overview_gray, rectifier, world_coords, args.size)
        bbox = detect_maze_bbox(masked, rectifier, world_coords, args.size,
                                block_px=args.adaptive_block, adaptive_c=args.adaptive_c,
                                density_thresh=args.density_threshold)
        if bbox is None:
            print("Could not detect the maze region; saving overview only.")
            cv2.imwrite(os.path.join(outdir, "maze_overview.png"), overview)
            return
        u0, v0, u1, v1 = bbox
        maze_gray = overview_gray[v0:v1 + 1, u0:u1 + 1]
        print("Maze bounded by wall-density detection (fewer than 4 corner tags).")

    # Maze frame: origin at the rectified top-left corner (= A* grid origin),
    # with +X along grid columns and +Y along grid rows. The axis directions in
    # the world frame depend on the rectifier flips, so build the rotation from
    # them to keep grid<->world consistent.
    x0_w, y0_w = rectifier.rect_to_world(u0, v0)
    maze_w_m = (u1 - u0) / args.ppm
    maze_h_m = (v1 - v0) / args.ppm
    x_axis, y_axis, z_axis = rectifier.maze_axes_world()
    T_world_maze = np.eye(4, dtype=np.float64)
    T_world_maze[:3, 0] = x_axis
    T_world_maze[:3, 1] = y_axis
    T_world_maze[:3, 2] = z_axis
    T_world_maze[:3, 3] = [x0_w, y0_w, 0.0]

    print(f"Maze origin (top-left corner) relative to tag {anchor_id} origin:")
    print(f"  x = {x0_w*1000:+.1f} mm, y = {y0_w*1000:+.1f} mm")
    print(f"Maze size: {maze_w_m*1000:.1f} x {maze_h_m*1000:.1f} mm")
    print(f"T_world_maze =\n{np.array2string(T_world_maze, precision=4, suppress_small=True)}")

    occ = build_occupancy_grid(
        maze_gray, args.grid_size,
        block_px=args.adaptive_block, adaptive_c=args.adaptive_c,
        occ_fraction=args.occ_fraction, wall_dilate_px=args.wall_dilate,
    )
    occ_pct = 100.0 * occ.mean()
    cell_w_mm = maze_w_m / args.grid_size * 1000.0
    cell_h_mm = maze_h_m / args.grid_size * 1000.0
    print(f"\nOccupancy grid: {args.grid_size} x {args.grid_size} "
          f"({occ_pct:.1f}% occupied)")
    print(f"Cell size: {cell_w_mm:.2f} x {cell_h_mm:.2f} mm")

    # Save artifacts.
    maze_rect_path = os.path.join(outdir, "maze_rectified.png")
    occ_img_path = os.path.join(outdir, "occupancy_grid.png")
    occ_npy_path = os.path.join(outdir, "occupancy_grid.npy")
    meta_path = os.path.join(outdir, "maze_frame.npz")
    overview_path = os.path.join(outdir, "maze_overview.png")

    cv2.imwrite(maze_rect_path, maze_gray)
    cv2.imwrite(occ_img_path, occupancy_to_image(occ))
    np.save(occ_npy_path, occ)

    # Verification overview: draw detected maze bbox.
    overview_vis = overview.copy()
    cv2.rectangle(overview_vis, (u0, v0), (u1, v1), (0, 0, 255), 3)
    cv2.imwrite(overview_path, overview_vis)

    np.savez(
        meta_path,
        T_world_maze=T_world_maze,
        T_cam_world=T_cam_world,
        anchor_id=anchor_id,
        ppm=args.ppm,
        x_min=rectifier.x_min, y_min=rectifier.y_min,
        x_max=rectifier.x_max, y_max=rectifier.y_max,
        flip_x=rectifier.flip_x, flip_y=rectifier.flip_y,
        bbox=np.array(bbox), grid_size=args.grid_size,
        maze_w_m=maze_w_m, maze_h_m=maze_h_m,
        K=K, dist=dist, marker_side_m=args.size,
    )

    print("\nSaved:")
    print(f"  Rectified maze (for create_nodes): {maze_rect_path}")
    print(f"  Occupancy grid image:              {occ_img_path}")
    print(f"  Occupancy grid array (bool NxN):   {occ_npy_path}")
    print(f"  Maze-frame transform + metadata:   {meta_path}")
    print(f"  Detection overview:                {overview_path}")


if __name__ == "__main__":
    main()
