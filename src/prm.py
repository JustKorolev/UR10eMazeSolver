#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import random
import imutils
import time

from math               import pi, sin, cos, sqrt, ceil, dist
from scipy.spatial      import KDTree

from alpha.astar import AStarNode, astar


#   Node Definition
#
class Node(AStarNode):
    # Static variables
    MAP_WIDTH = 320 # FIXME: UPDATE THIS IN PLANNER!!!
    MAP_HEIGHT = 220 #
    OBSTACLE_THRESHOLD = 100

    def __init__(self, x, y, dir=np.array([0, 0])):
        # Setup the basic A* node.
        super().__init__()

        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = int(x)
        self.y = int(y)
        self.dir = dir

        ROB = ['                 ',  #  8
               '     #######     ',  #  7
               '   ###########   ',  #  6
               ' ############### ',  #  3
               ' ############### ',  #  5
               ' ############### ',  #  4
               ' ############### ',  #  2
               ' ############### ',  #  1
               ' #######@####### ',  #  0
               ' ############### ',  # -1
               ' ############### ',  # -2
               ' ############### ',  # -3
               ' ############### ',  # -4
               ' ############### ',  # -5
               '   ###########   ',  # -6
               '     ######      ',  # -7
               '                 ',] # -8
        self.robot_mask = np.array([[(cell != ' ') for cell in row] for row in ROB],
                                   dtype="uint8")
        self.delta = self.robot_mask.shape[0] // 2
        # Check for proper shape
        assert np.size(self.robot_mask, 0) == np.size(self.robot_mask, 1), "Mask must be square"
        assert np.size(self.robot_mask, 0) % 2 == 1,                       "Mask must be odd-sized"

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    def bresenham(self, start, end):
        # Extract the coordinates
        (xs, ys) = start
        (xe, ye) = end

        if (xs == xe):
            return []

        # Move along ray (excluding endpoint).
        if (np.abs(xe-xs) >= np.abs(ye-ys)):
            return[(u, int(ys + (ye-ys)/(xe-xs) * (u+0.5-xs)))
                   for u in range(int(xs), int(xe), int(np.sign(xe-xs)))]
        else:
            return[(int(xs + (xe-xs)/(ye-ys) * (v+0.5-ys)), v)
                   for v in range(int(ys), int(ye), np.sign(ye-ys))]

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    # Compute the Euclidean distance to another node.
    def distance(self, other):
        return np.linalg.norm(np.subtract(self.coordinates(), other.coordinates()))

    ###############
    # A* functions:
    # Actual and Estimated costs.
    def costToConnect(self, other):
        return self.distance(other) + self.penalty

    def costToGoEst(self, other):
        return self.distance(other) + self.penalty

    ################
    # Collision functions:
    # Check whether in free space.
    def is_free(self, map_array, theta, override_coords=False, coords=None):

        if override_coords:
            x, y = coords[0], coords[1]
        else:
            x, y = self.x, self.y

        min_x = x - self.delta
        max_x = x + self.delta
        min_y = y - self.delta
        max_y = y + self.delta

        map_slice = map_array[min_y:max_y + 1, min_x:max_x + 1]
        mask = imutils.rotate(self.robot_mask, angle = 0)

        try:
            masked_values = np.logical_and(map_slice, mask) #map_slice[mask]
        except ValueError:
            return False

        if np.any(masked_values == True):
            return False
        return True

    # Check the local planner - whether this connects to another node.
    def connects_to(self, other, map_array): # FIXME
        intermediates = np.array(self.bresenham(self.coordinates(),
                                 other.coordinates()))

        for x_interm, y_interm in intermediates:
            if not self.is_free(map_array, 0, True, (x_interm, y_interm)):
                return False
        return True


######################################################################
#
#   PRM Functions
#
# Create the list of nodes.
def create_nodes(N, map_array):
    # Add nodes sampled uniformly across the space.
    nodes = []
    node_positions = set()

    while len(nodes) < N:
        x, y = random.uniform(0, Node.MAP_WIDTH), random.uniform(0, Node.MAP_HEIGHT)
        node = Node(x, y)
        if not node.is_free(map_array, 0, False) or (x, y) in node_positions:
            continue
        nodes.append(node)
        node_positions.add((x,y))

    return nodes

# Connect the nearest neighbors
def connect_nearest_neighbors(nodes, K, map_array):
    # Clear any existing neighbors.  Use a set to add below.
    for node in nodes:
        node.neighbors = set()

    X = np.array([node.coordinates() for node in nodes])
    [dist, idx] = KDTree(X).query(X, k=(K+1))

    # Add the edges.  Ignore the first neighbor (being itself).
    for i, nbrs in enumerate(idx):
        for n in nbrs[1:]:
            if nodes[i].connects_to(nodes[n], map_array):
                nodes[i].neighbors.add(nodes[n])
                nodes[n].neighbors.add(nodes[i])

def add_query_nodes(nodes, start_node, goal_node, K, map_array):
    start_node.query = True
    goal_node.query = True
    nodes.append(start_node)
    nodes.append(goal_node)

    coords = np.array([n.coordinates() for n in nodes[:-2]])
    tree   = KDTree(coords)

    for node in (start_node, goal_node):
        dists, idxs = tree.query(node.coordinates(), k=K)
        for i in idxs:
            neighbor = nodes[i]
            # only connect if there's a collision-free edge
            # if node.connects_to(neighbor, map_array):
            node.neighbors.add(neighbor)
            neighbor.neighbors.add(node)

def update_blocked_nodes(nodes, map_array, blocked_radius=2):

    # Get blocked map ids
    rows, cols = np.where(map_array >= Node.OBSTACLE_THRESHOLD)
    blocked_idxs = np.column_stack((cols, rows))

    # Get unblocked map ids
    rows, cols = np.where(map_array < Node.OBSTACLE_THRESHOLD)
    unblocked_idxs = np.column_stack((cols, rows))

    # Get prm node and prm blocked node inidices
    prm_node_coords = np.array([list(node.coordinates()) for node in nodes], dtype=float)
    blocked_prm_node_hits = KDTree(prm_node_coords).query_ball_point(blocked_idxs, blocked_radius)
    blocked_prm_node_idxs = set(idx for sub in blocked_prm_node_hits for idx in sub)
    unblocked_prm_node_hits = KDTree(prm_node_coords).query_ball_point(unblocked_idxs, 1)
    unblocked_prm_node_idxs = set(idx for sub in unblocked_prm_node_hits for idx in sub)

    for idx in unblocked_prm_node_idxs:
        node = nodes[idx]
        node.blocked = False

    for idx in blocked_prm_node_idxs:
        node = nodes[idx]
        if not node.query:
            node.blocked = True

        # Plot PRM nodes, blocked nodes, and obstacles
    all_coords = np.array([node.coordinates() for node in nodes])
    blocked_coords = np.array([node.coordinates() for node in nodes if node.blocked])

    # Convert blocked_idxs (grid coordinates) to match PRM node coordinates
    obstacle_coords = blocked_idxs  # Already in grid coordinates

    # Plot using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.scatter(all_coords[:, 0], all_coords[:, 1], label="All Nodes", color="blue", s=10)
    if len(blocked_coords) > 0:
        plt.scatter(blocked_coords[:, 0], blocked_coords[:, 1], label="Blocked Nodes", color="red", s=20)
    plt.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], label="Obstacles", color="black", s=5)

    plt.legend()
    plt.xlabel("X (Grid Coordinates)")
    plt.ylabel("Y (Grid Coordinates)")
    plt.title("PRM Nodes and Obstacles")
    plt.grid(True)
    plt.savefig("prm_graph.png")
    plt.close()


def update_node_penalties(nodes, penalty_map):
    H, W = penalty_map.shape
    for node in nodes:
        ix = int(round(node.x))
        iy = int(round(node.y))
        if 0 <= ix < W and 0 <= iy < H:
            node.penalty = float(penalty_map[iy, ix])
        else:
            node.penalty = 0.0


    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_subplot(111)
    im  = ax.imshow(penalty_map,
                    cmap="coolwarm",
                    origin="lower",
                    interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="penalty")
    ax.set_title("Coverage penalty map")
    ax.set_xlabel("grid-x")
    ax.set_ylabel("grid-y")
    fig.tight_layout()

    fig.savefig("penalty_map.png", dpi=200)
    plt.close(fig)

# Post Process the Path
def post_process(path, map_array):
    i = 0
    while i < len(path) - 2:
        if path[i].connects_to(path[i+2], map_array):
            path.remove(path[i+1])
            i -= 1
        i += 1