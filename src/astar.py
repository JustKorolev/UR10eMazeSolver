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


def create_nodes(
    N,
    map_image,
    connectivity=8,
    free_threshold=128,
    free_fraction_threshold=0.5,
    allow_corner_cutting=False,
):
    """
    Create an N x N graph of uniformly spaced pixel-space A* nodes.

    Bright pixels are free space and dark pixels are obstacles. A node is free
    when enough pixels in its image cell are brighter than free_threshold.
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
    height, width = image.shape
    node_grid = np.empty((N, N), dtype=object)
    nodes = []

    for row in range(N):
        for col in range(N):
            x, y = _cell_center(row, col, N, height, width)
            node = AStarNode(x, y, row=row, col=col)
            node.blocked = not _cell_is_free(
                image, row, col, N, free_threshold, free_fraction_threshold
            )
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
            points.append((float(x), float(y)))
    return points


def spline_path(path, samples_per_segment=10):
    """
    Return a Catmull-Rom spline interpolation of an A* path in pixel space.

    The spline passes through the A* points. It does not re-check collision
    against the map image, so use this as a geometric smoothing step after A*.
    """
    points = path_to_pixels(path)
    if points is None:
        return None
    if len(points) < 3:
        return points
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be at least 1")

    pts = np.array(points, dtype=float)
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
    smooth_points = spline_path(path, samples_per_segment=samples_per_segment)

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

    arr = cv2.imread("mazes/maze_1.png", cv2.IMREAD_GRAYSCALE)
    nodes = create_nodes(100, arr)
    free_nodes = [node for node in nodes if not node.blocked]

    start = free_nodes[0]
    goal = free_nodes[-1]
    path = astar(nodes, start, goal)

    if path is None:
        print("No path found")
    else:
        print(f"Path found with {len(path)} nodes")
        print(path_to_pixels(path))
        os.makedirs("outputs", exist_ok=True)
        save_plan_overlay(arr, path, "outputs/astar_overlay.png")
        save_plan_overlay(
            arr,
            spline_path(path, samples_per_segment=1000),
            "outputs/astar_spline_overlay.png",
            color=(255, 0, 255),
            thickness=3,
        )
        save_spline_overlay(arr, path, "outputs/astar_raw_vs_spline_overlay.png")
