"""
Generate overlay comparing:
1. Reference trajectory (from maze path planning)
2. FK actual trajectory (from fk_xy_diagnostic - the real robot positions)
3. MPC integrated trajectory (q_ref integrated using velocity commands from mpc_command_diagnostic)
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

# Load FK actual trajectory (THIS is what the robot actually did)
fk_data = np.genfromtxt(os.path.join(output_dir, "fk_xy_diagnostic.csv"), 
                         delimiter=',', skip_header=1)
fk_xy = fk_data[:, [3, 4]]  # fk_x, fk_y columns - actual robot position

# Load MPC diagnostic data (velocity commands)
mpc_command_data = np.genfromtxt(os.path.join(output_dir, "mpc_command_diagnostic.csv"),
                                 delimiter=',', skip_header=1)
time_col = mpc_command_data[:, 0]      # Column 0: time
u_cols = mpc_command_data[:, 1:7]      # Columns 1-6: u1-u6 (joint velocities)

# Load MPC trace (has q_ref and q_meas that were used)
mpc_trace = np.load(os.path.join(output_dir, "mpc_trace.npz"))
q_meas_trace = mpc_trace["q_meas"]  # Measured joint states
q_ref_trace = mpc_trace["q_ref"]    # Reference joint states from MPC

print(f"MPC command data shape: {mpc_command_data.shape}")
print(f"Time steps: {len(time_col)}")
print(f"dt: {time_col[1] - time_col[0]:.6f}s")
print(f"MPC trace q_meas shape: {q_meas_trace.shape}")
print(f"MPC trace q_ref shape: {q_ref_trace.shape}")

# Integrate MPC velocity commands starting from the correct initial position
# Start from q_ref[0] (the MPC's initial reference) and integrate velocity commands
dt = time_col[1] - time_col[0] if len(time_col) > 1 else (1/75)

print(f"\nInitial conditions:")
print(f"q_ref[0] from trace: {q_ref_trace[0]}")
print(f"q_meas[0] from trace: {q_meas_trace[0]}")

# Integrate velocities starting from q_ref[0]
q_integrated = np.zeros_like(q_ref_trace)
q_integrated[0] = q_ref_trace[0]  # Start from MPC's reference position

print(f"\nIntegrating MPC velocity commands...")
for i in range(1, min(len(u_cols), len(q_integrated))):
    # Integrate: q[i] = q[i-1] + u[i-1] * dt
    q_integrated[i] = q_integrated[i-1] + u_cols[i-1] * dt

# Trim to common length
min_len = min(len(ref_xy), len(fk_xy), len(q_meas_trace), len(q_integrated))
ref_xy_trim = ref_xy[:min_len]
fk_xy_trim = fk_xy[:min_len]
q_meas_trim = q_meas_trace[:min_len]
q_int_trim = q_integrated[:min_len]

print(f"\nTrajectory lengths: {min_len} points")

# Convert joint trajectories to XY using FK
def joints_to_xy(joints_array):
    """Convert joint trajectories to XY positions."""
    xy_positions = []
    for q_val in joints_array:
        try:
            T = robot.FK(q_val)
            xy_positions.append(T[:2, 3])
        except:
            if xy_positions:
                xy_positions.append(xy_positions[-1])
            else:
                xy_positions.append([0, 0])
    return np.array(xy_positions)

print("Computing XY from q_meas...")
mpc_meas_xy = joints_to_xy(q_meas_trim)

print("Computing XY from q_integrated...")
mpc_int_xy = joints_to_xy(q_int_trim)

# Calculate errors
ref_to_fk = np.linalg.norm(ref_xy_trim - fk_xy_trim, axis=1)
ref_to_mpc_meas = np.linalg.norm(ref_xy_trim - mpc_meas_xy, axis=1)
ref_to_mpc_int = np.linalg.norm(ref_xy_trim - mpc_int_xy, axis=1)
mpc_int_to_fk = np.linalg.norm(fk_xy_trim - mpc_int_xy, axis=1)

# The FK model doesn't match the real robot. Fix this by aligning MPC paths to start at FK actual position
# This is a coordinate frame offset correction
offset = fk_xy_trim[0] - mpc_int_xy[0]
mpc_int_xy_corrected = mpc_int_xy + offset

offset_meas = fk_xy_trim[0] - mpc_meas_xy[0]
mpc_meas_xy_corrected = mpc_meas_xy + offset_meas

print(f"\nApplying coordinate frame correction:")
print(f"  MPC integrated offset: {offset}")
print(f"  MPC measured offset: {offset_meas}")

# Recalculate errors with corrected positions
ref_to_mpc_int_corrected = np.linalg.norm(ref_xy_trim - mpc_int_xy_corrected, axis=1)
ref_to_mpc_meas_corrected = np.linalg.norm(ref_xy_trim - mpc_meas_xy_corrected, axis=1)
mpc_int_to_fk_corrected = np.linalg.norm(fk_xy_trim - mpc_int_xy_corrected, axis=1)

print(f"\nError statistics:")
print(f"  Ref to FK: mean={ref_to_fk.mean():.6f}m, max={ref_to_fk.max():.6f}m")
print(f"  Ref to MPC (measured q, RAW): mean={ref_to_mpc_meas.mean():.6f}m, max={ref_to_mpc_meas.max():.6f}m")
print(f"  Ref to MPC (integrated, RAW): mean={ref_to_mpc_int.mean():.6f}m, max={ref_to_mpc_int.max():.6f}m")
print(f"\nAfter coordinate frame alignment:")
print(f"  Ref to MPC (integrated, ALIGNED): mean={ref_to_mpc_int_corrected.mean():.6f}m, max={ref_to_mpc_int_corrected.max():.6f}m")
print(f"  Ref to MPC (measured, ALIGNED): mean={ref_to_mpc_meas_corrected.mean():.6f}m, max={ref_to_mpc_meas_corrected.max():.6f}m")
print(f"  MPC (integrated, ALIGNED) to FK: mean={mpc_int_to_fk_corrected.mean():.6f}m, max={mpc_int_to_fk_corrected.max():.6f}m")

# Create overlay visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: All trajectories overlaid
ax = axes[0]
ax.plot(ref_xy_trim[:, 0], ref_xy_trim[:, 1], 'g-', linewidth=2.5, label='Reference (Target Maze)', alpha=0.85, zorder=3)
ax.plot(fk_xy_trim[:, 0], fk_xy_trim[:, 1], 'r-', linewidth=2, label='FK Actual (Real Robot)', alpha=0.8, zorder=4)
ax.plot(mpc_int_xy_corrected[:, 0], mpc_int_xy_corrected[:, 1], 'b--', linewidth=2, label='MPC Predicted (integrated u)', alpha=0.75, zorder=2)
ax.plot(mpc_meas_xy_corrected[:, 0], mpc_meas_xy_corrected[:, 1], 'orange', linestyle=':', linewidth=1.5, label='MPC Measured (q_meas)', alpha=0.7, zorder=1)

# Mark start and end
ax.plot(ref_xy_trim[0, 0], ref_xy_trim[0, 1], 'go', markersize=14, label='Start', zorder=5)
ax.plot(ref_xy_trim[-1, 0], ref_xy_trim[-1, 1], 'r*', markersize=20, label='Goal', zorder=5)

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_title('MPC Trajectory Comparison: Reference vs Actual vs MPC Prediction', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Error along path
ax = axes[1]
path_distance = np.cumsum(np.linalg.norm(np.diff(ref_xy_trim, axis=0), axis=1))
path_distance = np.concatenate(([0], path_distance))

ax.plot(path_distance, ref_to_fk, 'r-', linewidth=2.5, label='Ref to FK', marker='o', markersize=2, markevery=100, alpha=0.8)
ax.plot(path_distance, ref_to_mpc_int_corrected, 'b--', linewidth=2.5, label='Ref to MPC (integrated)', marker='s', markersize=2, markevery=100, alpha=0.75)
ax.plot(path_distance, ref_to_mpc_meas_corrected, 'orange', linewidth=2, linestyle=':', label='Ref to MPC (measured)', marker='^', markersize=1.5, markevery=100, alpha=0.7)
ax.plot(path_distance, mpc_int_to_fk_corrected, 'purple', linewidth=2, linestyle='-.', label='MPC (int) to FK', marker='D', markersize=1.5, markevery=100, alpha=0.7)

ax.set_xlabel('Path Distance (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
ax.set_title('Tracking Error Along Path', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, "mpc_path_overlay.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved trajectory overlay to: {output_path}")
plt.close()

# Create error heatmap
fig, ax = plt.subplots(figsize=(13, 11))

# Color by MPC integrated vs FK error
scatter = ax.scatter(fk_xy_trim[:, 0], fk_xy_trim[:, 1], c=mpc_int_to_fk_corrected, 
                     cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)

# Plot paths
ax.plot(ref_xy_trim[:, 0], ref_xy_trim[:, 1], 'g-', linewidth=2.5, label='Reference', alpha=0.85, zorder=3)
ax.plot(fk_xy_trim[:, 0], fk_xy_trim[:, 1], 'r-', linewidth=1.5, label='FK Actual', alpha=0.7, zorder=4)
ax.plot(mpc_int_xy_corrected[:, 0], mpc_int_xy_corrected[:, 1], 'b--', linewidth=2, label='MPC Integrated', alpha=0.6, zorder=2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Error: MPC Integrated vs FK Actual (m)', fontsize=11, fontweight='bold')

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_title('MPC Integration Error Heatmap: Predicted vs Actual FK Path', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

output_path2 = os.path.join(output_dir, "mpc_error_heatmap.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved error heatmap to: {output_path2}")
plt.close()

print("\nVisualization complete!")
