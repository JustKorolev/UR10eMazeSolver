"""
Generate an overlay image comparing:
1. Reference trajectory (from maze path planning)
2. FK actual trajectory (forward kinematics of achieved positions)
3. MPC predicted trajectory (from q_ref in MPC trace)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.ur10e import UR10e
from src.utils import pose6_to_T

# Load data
output_dir = "outputs"

# Initialize robot for FK
WORKSPACE_OFFSET = pose6_to_T([0, -0.8, 0.1, np.pi, 0.01, 0.01])
robot = UR10e(workspace_offset=WORKSPACE_OFFSET)

# Load reference trajectory (target maze path)
ref_xyz = np.load(os.path.join(output_dir, "base_waypoints_xyz_m.npy"))
ref_xy = ref_xyz[:, :2]

# Load FK actual trajectory
fk_data = np.genfromtxt(os.path.join(output_dir, "fk_xy_diagnostic.csv"), 
                         delimiter=',', skip_header=1, usecols=[3, 4])  # fk_x, fk_y
fk_xy = fk_data

# Load MPC trace to get reference joint trajectory
mpc_trace = np.load(os.path.join(output_dir, "mpc_trace.npz"))
q_ref = mpc_trace["q_ref"]  # Reference joint trajectory
q_meas = mpc_trace["q_meas"]  # Measured joint trajectory

# Convert q_ref and q_meas to XY positions using FK
def joints_to_xy(joints_array, robot):
    """Convert joint trajectories to XY positions."""
    xy_positions = []
    for q_modified in joints_array:
        try:
            T = robot.FK(q_modified)
            xy_positions.append(T[:2, 3])  # Extract XY from transformation matrix
        except:
            # If FK fails, use last valid position
            if xy_positions:
                xy_positions.append(xy_positions[-1])
            else:
                xy_positions.append([0, 0])
    return np.array(xy_positions)

print("Computing FK positions from reference joint trajectory...")
mpc_xy_ref = joints_to_xy(q_ref, robot)

print("Computing FK positions from measured joint trajectory...")
mpc_xy_meas = joints_to_xy(q_meas, robot)

# Trim all trajectories to the shortest common length
min_len = min(len(ref_xy), len(fk_xy), len(mpc_xy_ref), len(mpc_xy_meas))
ref_xy_trim = ref_xy[:min_len]
fk_xy_trim = fk_xy[:min_len]
mpc_xy_ref_trim = mpc_xy_ref[:min_len]
mpc_xy_meas_trim = mpc_xy_meas[:min_len]

print(f"\nTrajectory lengths:")
print(f"  Reference trajectory: {len(ref_xy_trim)} points")
print(f"  FK actual trajectory: {len(fk_xy_trim)} points")
print(f"  MPC ref (predicted): {len(mpc_xy_ref_trim)} points")
print(f"  MPC meas (executed): {len(mpc_xy_meas_trim)} points")

# Calculate errors
ref_to_fk = np.linalg.norm(ref_xy_trim - fk_xy_trim, axis=1)
ref_to_mpc_ref = np.linalg.norm(ref_xy_trim - mpc_xy_ref_trim, axis=1)
ref_to_mpc_meas = np.linalg.norm(ref_xy_trim - mpc_xy_meas_trim, axis=1)
fk_to_mpc_ref = np.linalg.norm(fk_xy_trim - mpc_xy_ref_trim, axis=1)

print(f"\nError statistics:")
print(f"Reference to FK: mean={ref_to_fk.mean():.6f}m, max={ref_to_fk.max():.6f}m, std={ref_to_fk.std():.6f}m")
print(f"Reference to MPC (ref): mean={ref_to_mpc_ref.mean():.6f}m, max={ref_to_mpc_ref.max():.6f}m, std={ref_to_mpc_ref.std():.6f}m")
print(f"Reference to MPC (meas): mean={ref_to_mpc_meas.mean():.6f}m, max={ref_to_mpc_meas.max():.6f}m, std={ref_to_mpc_meas.std():.6f}m")
print(f"FK to MPC (ref): mean={fk_to_mpc_ref.mean():.6f}m, max={fk_to_mpc_ref.max():.6f}m, std={fk_to_mpc_ref.std():.6f}m")

# Create overlay visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: All trajectories overlaid
ax = axes[0]
ax.plot(ref_xy_trim[:, 0], ref_xy_trim[:, 1], 'g-', linewidth=2.5, label='Reference (Target Maze Path)', alpha=0.8)
ax.plot(fk_xy_trim[:, 0], fk_xy_trim[:, 1], 'r--', linewidth=1.5, label='FK Actual', alpha=0.7)
ax.plot(mpc_xy_ref_trim[:, 0], mpc_xy_ref_trim[:, 1], 'b:', linewidth=2, label='MPC Reference (q_ref)', alpha=0.7)
ax.plot(mpc_xy_meas_trim[:, 0], mpc_xy_meas_trim[:, 1], 'orange', linestyle='-.', linewidth=1.5, label='MPC Measured (q_meas)', alpha=0.7)

# Mark start and end
ax.plot(ref_xy_trim[0, 0], ref_xy_trim[0, 1], 'go', markersize=12, label='Start', zorder=5)
ax.plot(ref_xy_trim[-1, 0], ref_xy_trim[-1, 1], 'r*', markersize=18, label='End', zorder=5)

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_title('Trajectory Comparison: Reference vs FK vs MPC', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Error over path progression
ax = axes[1]
path_distance = np.cumsum(np.linalg.norm(np.diff(ref_xy_trim, axis=0), axis=1))
path_distance = np.concatenate(([0], path_distance))

ax.plot(path_distance, ref_to_fk, 'r-', linewidth=2, label='Ref to FK Error', marker='o', markersize=1, markevery=50)
ax.plot(path_distance, ref_to_mpc_ref, 'b-', linewidth=2, label='Ref to MPC (ref)', marker='s', markersize=1, markevery=50)
ax.plot(path_distance, ref_to_mpc_meas, 'orange', linewidth=2, linestyle='--', label='Ref to MPC (meas)', marker='^', markersize=1, markevery=50)

ax.set_xlabel('Path Distance (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
ax.set_title('Tracking Error Along Path', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, "mpc_path_overlay.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved trajectory overlay plot to: {output_path}")
plt.close()

# Create a detailed spatial comparison plot with error heatmap
fig, ax = plt.subplots(figsize=(13, 11))

# Create a scatter plot with color based on error to reference
scatter = ax.scatter(fk_xy_trim[:, 0], fk_xy_trim[:, 1], c=ref_to_fk, 
                     cmap='RdYlGn_r', s=25, alpha=0.7, label='FK Actual (colored by error to Ref)')

# Plot reference path
ax.plot(ref_xy_trim[:, 0], ref_xy_trim[:, 1], 'g-', linewidth=2.5, label='Reference Path', alpha=0.8, zorder=3)

# Plot MPC predicted paths
ax.plot(mpc_xy_ref_trim[:, 0], mpc_xy_ref_trim[:, 1], 'b:', linewidth=2, label='MPC Ref (q_ref)', alpha=0.6, zorder=2)
ax.plot(mpc_xy_meas_trim[:, 0], mpc_xy_meas_trim[:, 1], 'orange', linestyle='-.', linewidth=1.5, label='MPC Meas (q_meas)', alpha=0.6, zorder=2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Error from Reference (m)')
cbar.ax.tick_params(labelsize=10)

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_title('FK Trajectory with Error Magnitude Heatmap', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

output_path2 = os.path.join(output_dir, "mpc_error_heatmap.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved error heatmap to: {output_path2}")
plt.close()

print("\nVisualization complete!")
