"""
Generate overlay comparing:
1. Reference trajectory (from maze path planning)
2. FK actual trajectory (from fk_xy_diagnostic)
3. MPC predicted trajectory (integrated from joint velocity commands u1-u6)
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
                         delimiter=',', skip_header=1)
fk_xy = fk_data[:, [3, 4]]  # fk_x, fk_y columns

# Load MPC diagnostic data (velocity commands and joint angles)
mpc_data = np.genfromtxt(os.path.join(output_dir, "mpc_command_diagnostic.csv"),
                          delimiter=',', skip_header=1)
time_col = mpc_data[:, 0]      # Column 0: time
u_cols = mpc_data[:, 1:7]      # Columns 1-6: u1-u6 (joint velocities)
q_cols = mpc_data[:, 15:21]    # Columns 15-20: q1-q6 (joint angles)

print(f"\nMPC data shape: {mpc_data.shape}")
print(f"Time steps: {len(time_col)}")
print(f"dt: {time_col[1] - time_col[0]:.6f}s")
print(f"u_cols shape: {u_cols.shape}, q_cols shape: {q_cols.shape}")

# Debug: print first few velocity and joint commands
print(f"\nFirst 5 velocity commands (u1-u6):")
for i in range(min(5, len(u_cols))):
    print(f"  t={time_col[i]:.4f}s: u={u_cols[i]}")

print(f"\nFirst 5 joint angles (q1-q6):")
for i in range(min(5, len(q_cols))):
    print(f"  t={time_col[i]:.4f}s: q={q_cols[i]}")

# Get the starting FK position (in XYZ from reference)
fk_start_xyz = np.array([fk_xy[0, 0], fk_xy[0, 1], ref_xyz[0, 2]])  # Use reference Z
print(f"\nFK/Reference starting position (XYZ): {fk_start_xyz}")

# For MPC integration, we need to start from the correct joint configuration
# Use the first measured joint state as the starting point (it's already at the right place)
start_joints = q_cols[0].copy()
print(f"Starting joint position from q_cols[0]: {start_joints}")

# Verify this matches the FK actual starting position
T_check = robot.FK(start_joints)
check_xy = T_check[:2, 3]
print(f"FK check from start_joints: {check_xy} (should match FK actual start)")
print(f"FK actual start: {fk_xy[0]}")
# The problem might be that q_cols are in classical joint frame, not DH-modified
# Try converting from classical to modified
print(f"\nAttempting classical to modified joint conversion...")
try:
    q_start_modified = robot.ClassicalToDHModified(start_joints)
    print(f"q_start (classical): {start_joints}")
    print(f"q_start (modified): {q_start_modified}")
    T_modified = robot.FK(q_start_modified)
    check_xy_modified = T_modified[:2, 3]
    print(f"FK from modified joints: {check_xy_modified}")
except Exception as e:
    print(f"Conversion failed: {e}")
    q_start_modified = start_joints
# Integrate MPC velocity commands to predict joint trajectory
dt = time_col[1] - time_col[0] if len(time_col) > 1 else (1/75)

q_predicted = np.zeros_like(q_cols)
q_predicted[0] = start_joints  # Start from measured position

print(f"\nIntegrating MPC joint velocity commands from correct starting position...")
print(f"dt = {dt:.6f}s, time steps = {len(u_cols)}")
for i in range(1, len(u_cols)):
    # Integrate: q[i] = q[i-1] + u[i-1] * dt
    q_predicted[i] = q_predicted[i-1] + u_cols[i-1] * dt

# Debug: check if prediction is different from measured
print(f"\nIntegration check:")
print(f"q_predicted[0]: {q_predicted[0]}")
print(f"q_predicted[100]: {q_predicted[100]}")
print(f"q_cols[100]: {q_cols[100]}")
pred_diff = np.linalg.norm(q_predicted - q_cols)
print(f"Total difference (predicted vs measured): {pred_diff:.6f}")
print(f"First 5 predicted joints at t=5:")
for i in [0, 1, 2, 10, 50]:
    print(f"  t={time_col[i]:.4f}s: q_pred={q_predicted[i]}, q_meas={q_cols[i]}, diff={np.linalg.norm(q_predicted[i]-q_cols[i]):.6f}")

# Convert predicted joint trajectory to XY positions using FK
def joints_to_xy(joints_array, robot):
    """Convert joint trajectories to XY positions."""
    xy_positions = []
    for q_modified in joints_array:
        try:
            T = robot.FK(q_modified)
            xy_positions.append(T[:2, 3])  # Extract XY from transformation matrix
        except Exception as e:
            # If FK fails, use last valid position
            if xy_positions:
                xy_positions.append(xy_positions[-1])
            else:
                xy_positions.append([0, 0])
    return np.array(xy_positions)

print("Computing FK positions from MPC-integrated joint trajectory...")
mpc_xy_integrated = joints_to_xy(q_predicted, robot)

print("Computing FK positions from measured joint angles...")
mpc_xy_measured = joints_to_xy(q_cols, robot)


# Trim all trajectories to the shortest common length
min_len = min(len(ref_xy), len(fk_xy), len(mpc_xy_integrated))
ref_xy_trim = ref_xy[:min_len]
fk_xy_trim = fk_xy[:min_len]
mpc_xy_int_trim = mpc_xy_integrated[:min_len]
mpc_xy_meas_trim = mpc_xy_measured[:min_len]

print(f"\nTrajectory lengths:")
print(f"  Reference: {len(ref_xy_trim)} points")
print(f"  FK actual: {len(fk_xy_trim)} points")
print(f"  MPC integrated (predicted): {len(mpc_xy_int_trim)} points")
print(f"  MPC measured: {len(mpc_xy_meas_trim)} points")

# Calculate errors
ref_to_fk = np.linalg.norm(ref_xy_trim - fk_xy_trim, axis=1)
ref_to_mpc_int = np.linalg.norm(ref_xy_trim - mpc_xy_int_trim, axis=1)
ref_to_mpc_meas = np.linalg.norm(ref_xy_trim - mpc_xy_meas_trim, axis=1)
mpc_int_to_fk = np.linalg.norm(fk_xy_trim - mpc_xy_int_trim, axis=1)

print(f"\nError statistics:")
print(f"  Ref to FK: mean={ref_to_fk.mean():.6f}m, max={ref_to_fk.max():.6f}m, std={ref_to_fk.std():.6f}m")
print(f"  Ref to MPC (integrated): mean={ref_to_mpc_int.mean():.6f}m, max={ref_to_mpc_int.max():.6f}m, std={ref_to_mpc_int.std():.6f}m")
print(f"  Ref to MPC (measured): mean={ref_to_mpc_meas.mean():.6f}m, max={ref_to_mpc_meas.max():.6f}m, std={ref_to_mpc_meas.std():.6f}m")
print(f"  MPC (integrated) to FK: mean={mpc_int_to_fk.mean():.6f}m, max={mpc_int_to_fk.max():.6f}m, std={mpc_int_to_fk.std():.6f}m")

# Create overlay visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: All trajectories overlaid
ax = axes[0]
ax.plot(ref_xy_trim[:, 0], ref_xy_trim[:, 1], 'g-', linewidth=2.5, label='Reference (Target)', alpha=0.85, zorder=3)
ax.plot(fk_xy_trim[:, 0], fk_xy_trim[:, 1], 'r-', linewidth=2, label='FK Actual', alpha=0.8, zorder=4)
ax.plot(mpc_xy_int_trim[:, 0], mpc_xy_int_trim[:, 1], 'b--', linewidth=2, label='MPC Predicted (integrated u)', alpha=0.75, zorder=2)
ax.plot(mpc_xy_meas_trim[:, 0], mpc_xy_meas_trim[:, 1], 'orange', linestyle=':', linewidth=1.5, label='MPC Measured (q)', alpha=0.7, zorder=1)

# Mark start and end
ax.plot(ref_xy_trim[0, 0], ref_xy_trim[0, 1], 'go', markersize=14, label='Start', zorder=5)
ax.plot(ref_xy_trim[-1, 0], ref_xy_trim[-1, 1], 'r*', markersize=20, label='Goal', zorder=5)

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_title('MPC Trajectory Comparison: Reference vs Actual vs Integrated Velocity Commands', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Error along path
ax = axes[1]
path_distance = np.cumsum(np.linalg.norm(np.diff(ref_xy_trim, axis=0), axis=1))
path_distance = np.concatenate(([0], path_distance))

ax.plot(path_distance, ref_to_fk, 'r-', linewidth=2.5, label='Ref to FK', marker='o', markersize=2, markevery=100, alpha=0.8)
ax.plot(path_distance, ref_to_mpc_int, 'b--', linewidth=2.5, label='Ref to MPC (integrated)', marker='s', markersize=2, markevery=100, alpha=0.75)
ax.plot(path_distance, ref_to_mpc_meas, 'orange', linewidth=2, linestyle=':', label='Ref to MPC (measured)', marker='^', markersize=1.5, markevery=100, alpha=0.7)
ax.plot(path_distance, mpc_int_to_fk, 'purple', linewidth=2, linestyle='-.', label='MPC (int) to FK', marker='D', markersize=1.5, markevery=100, alpha=0.7)

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
scatter = ax.scatter(fk_xy_trim[:, 0], fk_xy_trim[:, 1], c=mpc_int_to_fk, 
                     cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)

# Plot paths
ax.plot(ref_xy_trim[:, 0], ref_xy_trim[:, 1], 'g-', linewidth=2.5, label='Reference', alpha=0.85, zorder=3)
ax.plot(fk_xy_trim[:, 0], fk_xy_trim[:, 1], 'r-', linewidth=1.5, label='FK Actual', alpha=0.7, zorder=4)
ax.plot(mpc_xy_int_trim[:, 0], mpc_xy_int_trim[:, 1], 'b--', linewidth=2, label='MPC Integrated', alpha=0.6, zorder=2)

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
