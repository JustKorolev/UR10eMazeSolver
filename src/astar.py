#!/usr/bin/env python3
#
#   astar.py
#
#   Simple occupancy-grid A* over uniformly sampled pixel-space nodes.

import heapq
import math

import numpy as np


class AStarNode:
    def __init__(self, x, y, row=None, col=None):
        self.x = float(x)
        self.y = float(y)
        self.row = row
        self.col = col
        self.neighbors = set()
        self.blocked = False
        self.reset()

    def reset(self):
        self.done = False
        self.seen = False
        self.parent = None
        self.creach = math.inf
        self.ctogoest = math.inf

    def __lt__(self, other):
        return (self.creach + self.ctogoest) < (other.creach + other.ctogoest)

    def __repr__(self):
        return f"<Point {self.x:5.1f},{self.y:5.1f}>"

    def coordinates(self):
        return (self.x, self.y)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def costToConnect(self, other):
        return self.distance(other)

    def costToGoEst(self, other):
        return self.distance(other)


def _to_grayscale(map_image):
    image = np.asarray(map_image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[:, :, :3].mean(axis=2)
    raise ValueError(f"Expected a 2D or 3D image array, got shape {image.shape}")


def _cell_bounds(row, col, N, height, width):
    y0 = int(round(row * height / N))
    y1 = int(round((row + 1) * height / N))
    x0 = int(round(col * width / N))
    x1 = int(round((col + 1) * width / N))
    return y0, max(y1, y0 + 1), x0, max(x1, x0 + 1)


def _cell_center(row, col, N, height, width):
    y0, y1, x0, x1 = _cell_bounds(row, col, N, height, width)
    return ((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0)


def _cell_is_free(image, row, col, N, free_threshold, free_fraction_threshold):
    height, width = image.shape
    y0, y1, x0, x1 = _cell_bounds(row, col, N, height, width)
    patch = image[y0:y1, x0:x1]
    free_fraction = np.mean(patch >= free_threshold)
    return free_fraction >= free_fraction_threshold


def bresenham_pixels(start, end):
    """Return integer pixel coordinates on the line from start to end."""
    x0, y0 = start
    x1, y1 = end
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def line_is_free(image, start, end, free_threshold=128):
    """Check that every pixel on a candidate edge lies in bright/free space."""
    height, width = image.shape

    for x, y in bresenham_pixels(start, end):
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        if image[y, x] < free_threshold:
            return False

    return True


def inflate_obstacles(image, inflation_radius=1, free_threshold=128):
    """
    Expand dark obstacle pixels by inflation_radius pixels.

    Bright pixels are free. Dark pixels are obstacles. The returned image keeps
    the same bright/free convention, but any pixel near an obstacle is darkened.
    """
    if inflation_radius <= 0:
        return image

    obstacle_mask = image < free_threshold
    inflated_mask = obstacle_mask.copy()

    for dy in range(-inflation_radius, inflation_radius + 1):
        for dx in range(-inflation_radius, inflation_radius + 1):
            if dx * dx + dy * dy > inflation_radius * inflation_radius:
                continue

            src_y0 = max(0, -dy)
            src_y1 = min(image.shape[0], image.shape[0] - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(image.shape[1], image.shape[1] - dx)

            dst_y0 = max(0, dy)
            dst_y1 = min(image.shape[0], image.shape[0] + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(image.shape[1], image.shape[1] + dx)

            inflated_mask[dst_y0:dst_y1, dst_x0:dst_x1] |= obstacle_mask[
                src_y0:src_y1, src_x0:src_x1
            ]

    inflated = image.copy()
    inflated[inflated_mask] = 0
    return inflated


def maze_wall_bbox(map_image, free_threshold=128):
    """Bounding box (x0, y0, x1, y1) of the maze's wall structure.

    The maze walls form one big dark blob (tags are masked white in the crop),
    so the largest dark connected component's bounding box is the maze extent.
    Returns None if no dark pixels are found.
    """
    import cv2

    gray = _to_grayscale(map_image)
    dark = (gray < free_threshold).astype(np.uint8) * 255
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if nlab <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x0 = int(stats[best, cv2.CC_STAT_LEFT])
    y0 = int(stats[best, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[best, cv2.CC_STAT_WIDTH]) - 1
    y1 = y0 + int(stats[best, cv2.CC_STAT_HEIGHT]) - 1
    return x0, y0, x1, y1


def create_nodes(
    N,
    map_image,
    connectivity=8,
    free_threshold=128,
    free_fraction_threshold=0.5,
    allow_corner_cutting=False,
    obstacle_inflation_radius=12,
    confine_to_maze=True,
):
    """
    Create an N x N graph of uniformly spaced pixel-space A* nodes.

    Bright pixels are free space and dark pixels are obstacles. A node is free
    when enough pixels in its image cell are brighter than free_threshold.

    When confine_to_maze is True, nodes whose centers fall outside the maze's
    wall bounding box are blocked. This stops A* from "solving" the maze by
    routing through the free margin AROUND it (the entrance/exit openings
    connect the interior to that margin), forcing a path through the maze.
    """
    if np.isscalar(map_image) and not np.isscalar(connectivity):
        # Backward compatibility for old create_nodes(N, K, map_image) calls.
        map_image = connectivity
        connectivity = 8

    if map_image is None:
        raise ValueError("map_image cannot be None")
    if N <= 0:
        raise ValueError("N must be positive")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    image = _to_grayscale(map_image)

    bbox = maze_wall_bbox(image, free_threshold) if confine_to_maze else None

    image = inflate_obstacles(
        image,
        inflation_radius=obstacle_inflation_radius,
        free_threshold=free_threshold,
    )
    height, width = image.shape
    node_grid = np.empty((N, N), dtype=object)
    nodes = []

    # Small tolerance so openings right on the wall keep an interior node.
    bmargin = 2
    for row in range(N):
        for col in range(N):
            x, y = _cell_center(row, col, N, height, width)
            node = AStarNode(x, y, row=row, col=col)
            node.blocked = not _cell_is_free(
                image, row, col, N, free_threshold, free_fraction_threshold
            )
            if bbox is not None and not node.blocked:
                x0, y0, x1, y1 = bbox
                if (x < x0 - bmargin or x > x1 + bmargin
                        or y < y0 - bmargin or y > y1 + bmargin):
                    node.blocked = True  # exterior margin -> not traversable
            node_grid[row, col] = node
            nodes.append(node)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for node in nodes:
        if node.blocked:
            continue

        for drow, dcol in directions:
            nrow = node.row + drow
            ncol = node.col + dcol

            if nrow < 0 or nrow >= N or ncol < 0 or ncol >= N:
                continue

            neighbor = node_grid[nrow, ncol]
            if neighbor.blocked:
                continue

            is_diagonal = drow != 0 and dcol != 0
            if is_diagonal and not allow_corner_cutting:
                if node_grid[node.row, ncol].blocked or node_grid[nrow, node.col].blocked:
                    continue

            if not line_is_free(
                image, node.coordinates(), neighbor.coordinates(), free_threshold
            ):
                continue

            node.neighbors.add(neighbor)

    return nodes


def _robust_line_fit(t, p, iters=3, keep=2.5):
    """Fit p ~ a*t + b, iteratively rejecting outliers (the wall openings)."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    mask = np.ones(len(t), bool)
    a, b = 0.0, float(np.median(p))
    for _ in range(iters):
        a, b = np.polyfit(t[mask], p[mask], 1)
        resid = p - (a * t + b)
        s = np.std(resid[mask]) or 1.0
        mask = np.abs(resid) < keep * s
    return a, b


def _edge_openings(profile, valid, expect_high, min_len=6, dev=8):
    """Find wall gaps on one edge from its extreme-dark profile.

    profile[t] is the extreme dark coordinate along the edge (e.g. the rightmost
    dark pixel x for each row). A solid wall traces a near-straight (possibly
    tilted) line; an opening is where that wall recedes inward by > dev pixels.
    Fitting the line robustly (ignoring the gaps) makes this tilt-invariant.
    Returns a list of (start, end) along-edge index ranges.
    """
    t = np.array([i for i in range(len(profile)) if valid[i]])
    if len(t) < 10:
        return [], (0.0, 0.0)
    p = np.array([profile[i] for i in t], float)
    a, b = _robust_line_fit(t, p)
    line = a * t + b
    is_gap = (p < line - dev) if expect_high else (p > line + dev)

    runs = []
    start = None
    for i, g in enumerate(list(is_gap) + [False]):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= min_len:
                seg = t[start:i]
                runs.append((int(seg.min()), int(seg.max())))
            start = None
    return runs, (float(a), float(b))


def detect_openings(map_image, free_threshold=128):
    """Detect the maze's entrance/exit gaps in its outer boundary wall.

    Scans all four sides for inward dips in the wall line and returns each gap
    center as an (x, y) pixel point anchored ON the maze's outer wall edge for
    that side (so e.g. a left-edge opening sits at the left wall, not at the
    first interior wall past the gap). One point per detected opening.
    """
    image = _to_grayscale(map_image)
    dark = image < free_threshold
    h, w = dark.shape

    rightmost = [int(np.where(dark[y])[0].max()) if dark[y].any() else -1 for y in range(h)]
    leftmost = [int(np.where(dark[y])[0].min()) if dark[y].any() else 10**9 for y in range(h)]
    bottommost = [int(np.where(dark[:, x])[0].max()) if dark[:, x].any() else -1 for x in range(w)]
    topmost = [int(np.where(dark[:, x])[0].min()) if dark[:, x].any() else 10**9 for x in range(w)]
    valid_rows = [bool(dark[y].any()) for y in range(h)]
    valid_cols = [bool(dark[:, x].any()) for x in range(w)]

    # Each opening is anchored on the (possibly tilted) outer wall, evaluated
    # from that edge's robust line fit at the gap center -> tilt-invariant and
    # always on the true boundary (not the first interior wall past the gap).
    points = []
    runs, (a, b) = _edge_openings(rightmost, valid_rows, expect_high=True)
    for s, e in runs:
        yc = (s + e) // 2
        points.append((int(round(a * yc + b)), yc))
    runs, (a, b) = _edge_openings(leftmost, valid_rows, expect_high=False)
    for s, e in runs:
        yc = (s + e) // 2
        points.append((int(round(a * yc + b)), yc))
    runs, (a, b) = _edge_openings(bottommost, valid_cols, expect_high=True)
    for s, e in runs:
        xc = (s + e) // 2
        points.append((xc, int(round(a * xc + b))))
    runs, (a, b) = _edge_openings(topmost, valid_cols, expect_high=False)
    for s, e in runs:
        xc = (s + e) // 2
        points.append((xc, int(round(a * xc + b))))
    return points


def largest_free_component(nodes):
    """Return the biggest connected set of free nodes (the maze interior).

    The cropped map has small free pockets in the margin that are walled off
    from the maze; planning between arbitrary corners can land in different
    pockets. Restricting start/goal to the largest component avoids that.
    """
    seen = set()
    best = []
    for node in nodes:
        if node.blocked or node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in cur.neighbors:
                if nb not in seen and not nb.blocked:
                    seen.add(nb)
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    return best


def astar(nodes, start, goal):
    for node in nodes:
        node.reset()

    start.blocked = False
    goal.blocked = False

    on_deck = []
    counter = 0

    start.seen = True
    start.parent = None
    start.creach = 0
    start.ctogoest = start.costToGoEst(goal)
    heapq.heappush(on_deck, (start.creach + start.ctogoest, counter, start))

    while on_deck:
        _, _, node = heapq.heappop(on_deck)
        if node.done:
            continue

        node.done = True
        if node is goal:
            break

        for neighbor in node.neighbors:
            if neighbor.done or neighbor.blocked:
                continue

            creach = node.creach + node.costToConnect(neighbor)
            if neighbor.seen and neighbor.creach <= creach:
                continue

            neighbor.seen = True
            neighbor.parent = node
            neighbor.creach = creach
            neighbor.ctogoest = neighbor.costToGoEst(goal)
            counter += 1
            heapq.heappush(on_deck, (neighbor.creach + neighbor.ctogoest, counter, neighbor))

    if goal.parent is None and goal is not start:
        return None

    path = [goal]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    return path


def path_to_pixels(path):
    if path is None:
        return None
    points = []
    for item in path:
        if hasattr(item, "coordinates"):
            points.append(item.coordinates())
        else:
            x, y = item
            points.append((round(float(x),3), round(float(y),3)))
    return points


def spline_path(path, samples_per_segment=10, control_point_stride=10):
    """
    Return a Catmull-Rom spline interpolation of an A* path in pixel space.

    The spline passes through every control_point_stride-th A* point, plus the
    start and goal.
    """
    points = path_to_pixels(path)
    if points is None:
        return None
    if len(points) < 3:
        return points
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be at least 1")
    if control_point_stride < 1:
        raise ValueError("control_point_stride must be at least 1")

    control_points = points[::control_point_stride]
    if control_points[-1] != points[-1]:
        control_points.append(points[-1])
    if len(control_points) < 3:
        control_points = points

    pts = np.array(control_points, dtype=float)
    smoothed = []

    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            smoothed.append((float(point[0]), float(point[1])))

    smoothed.append((float(pts[-1, 0]), float(pts[-1, 1])))
    return smoothed


def build_spline(path, samples_per_segment=10, control_point_stride=10):
    return spline_path(
        path,
        samples_per_segment=samples_per_segment,
        control_point_stride=control_point_stride,
    )


def save_plan_overlay(map_image, path, output_path, color=(0, 0, 255), thickness=2):
    """
    Save an image with the piecewise A* path overlaid on the original map.

    color is in BGR order to match OpenCV conventions.
    """
    import cv2

    image = np.asarray(map_image)
    if image.ndim == 2:
        overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        overlay = image[:, :, :3].astype(np.uint8).copy()
    else:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {image.shape}")

    points = path_to_pixels(path)
    if points is None or len(points) == 0:
        cv2.imwrite(output_path, overlay)
        return overlay

    pts = np.array([[round(x), round(y)] for x, y in points], dtype=np.int32)
    cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)

    radius = max(3, thickness + 1)
    cv2.circle(overlay, tuple(pts[0]), radius, (0, 255, 0), -1)
    cv2.circle(overlay, tuple(pts[-1]), radius, (255, 128, 0), -1)

    cv2.imwrite(output_path, overlay)
    return overlay


def save_spline_overlay(
    map_image,
    path,
    output_path,
    samples_per_segment=10,
    control_point_stride=10,
    raw_color=(255, 0, 0),
    spline_color=(255, 0, 255),
):
    """Save one overlay showing raw A* segments and the fitted spline."""
    import cv2

    image = np.asarray(map_image)
    if image.ndim == 2:
        overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        overlay = image[:, :, :3].astype(np.uint8).copy()
    else:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {image.shape}")

    raw_points = path_to_pixels(path)
    smooth_points = spline_path(
        path,
        samples_per_segment=samples_per_segment,
        control_point_stride=control_point_stride,
    )

    if raw_points is None or len(raw_points) == 0:
        cv2.imwrite(output_path, overlay)
        return overlay

    raw_pts = np.array([[round(x), round(y)] for x, y in raw_points], dtype=np.int32)
    cv2.polylines(overlay, [raw_pts], isClosed=False, color=raw_color, thickness=1)

    if smooth_points is not None and len(smooth_points) > 1:
        spline_pts = np.array(
            [[round(x), round(y)] for x, y in smooth_points], dtype=np.int32
        )
        cv2.polylines(
            overlay, [spline_pts], isClosed=False, color=spline_color, thickness=3
        )

    cv2.circle(overlay, tuple(raw_pts[0]), 4, (0, 255, 0), -1)
    cv2.circle(overlay, tuple(raw_pts[-1]), 4, (255, 128, 0), -1)

    cv2.imwrite(output_path, overlay)
    return overlay


if __name__ == "__main__":
    import cv2
    import os

    # Resolve paths relative to the project root (parent of this src/ dir) so the
    # script works regardless of the current working directory.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(project_root, "outputs")
    map_path = os.path.join(outputs_dir, "maze_map.png")

    arr = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(
            f"Could not read maze map: {map_path}. "
            "Run maze_pipeline.py first to generate outputs/maze_map.png."
        )
    print(f"Loaded maze map {map_path} ({arr.shape[1]}x{arr.shape[0]} px)")

    nodes = create_nodes(100, arr, obstacle_inflation_radius=12)
    free_nodes = [node for node in nodes if not node.blocked]
    if not free_nodes:
        raise RuntimeError("No free cells found in the maze map; check thresholds.")

    # create_nodes already confines planning to the maze (blocks the exterior
    # margin); restrict further to the largest connected free region (interior).
    bbox = maze_wall_bbox(arr)
    if bbox is not None:
        print(f"Maze wall bbox: x[{bbox[0]},{bbox[2]}] y[{bbox[1]},{bbox[3]}]")
    interior = largest_free_component(nodes)
    print(f"Free nodes: {len(free_nodes)}; maze interior component: {len(interior)}")

    # Start/goal are the maze's two boundary openings (entrance + exit). Snap
    # each detected opening point to the nearest free interior node.
    openings = detect_openings(arr)
    print(f"Detected {len(openings)} boundary openings: {openings}")
    if len(openings) >= 2:
        def nearest_interior(pt):
            px, py = pt
            return min(interior, key=lambda n: (n.x - px) ** 2 + (n.y - py) ** 2)
        start = nearest_interior(openings[0])
        goal = nearest_interior(openings[1])
    else:
        print("WARNING: expected 2 openings; falling back to corner-to-corner.")
        start = min(interior, key=lambda n: n.row + n.col)
        goal = max(interior, key=lambda n: n.row + n.col)
    print(f"Start node (row,col)=({start.row},{start.col}) px=({start.x:.0f},{start.y:.0f})")
    print(f"Goal  node (row,col)=({goal.row},{goal.col}) px=({goal.x:.0f},{goal.y:.0f})")
    path = astar(nodes, start, goal)

    if path is None:
        print("No path found")
    else:
        print(f"Path found with {len(path)} nodes")

        # Extend the path endpoints out to the actual boundary openings, which
        # are already anchored on the maze's outer wall, so start/goal sit right
        # at the entrance/exit gaps.
        if len(openings) >= 2:
            path = [openings[0]] + list(path) + [openings[1]]
            print(f"Endpoints set to openings: {path[0]} -> {path[-1]}")

        os.makedirs(outputs_dir, exist_ok=True)
        save_plan_overlay(arr, path, os.path.join(outputs_dir, "astar_overlay.png"))
        save_spline_overlay(
            arr,
            path,
            os.path.join(outputs_dir, "astar_spline_overlay.png"),
            samples_per_segment=1,
            control_point_stride=3,
        )
        print(f"Saved overlays to {outputs_dir}")
