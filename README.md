# UR10e Maze Solver

Autonomous maze solving and drawing with a UR10e robot arm.

This repo captures an overhead image of a physical maze, localizes and rectifies it with AprilTags, plans a valid path through the maze, converts that path into robot joint targets, and executes the path on a UR10e with MPC velocity control.

## Demo Video

Autoplaying preview:

<p align="center">
  <img src="./outputs/demo_preview.gif" alt="UR10e maze solver demo preview" width="720">
</p>

Full MP4:

[outputs/235b_final_demo_maze_solver (1).mp4](outputs/235b_final_demo_maze_solver%20(1).mp4)

## What This Repo Can Do

- Move the UR10e to a fixed overhead camera pose.
- Capture a RealSense color image of the maze.
- Detect AprilTags around the maze.
- Rectify the maze with a homography.
- Detect the maze entrance and exit.
- Run A* through the maze without routing through walls or outside margins.
- Smooth the A* result with a spline.
- Convert the pixel path into metric robot-base waypoints.
- Convert Cartesian waypoints into UR10e joint targets with inverse kinematics.
- Resample the joint path for smooth, feasible playback.
- Track the trajectory with an MPC controller.
- Stream robot commands to the UR10e and record telemetry for debugging.
- Save overlays, diagnostics, traces, and generated trajectories in `outputs/`.

## Pipeline Overview

The full pipeline is:

```text
RealSense capture
  -> AprilTag detection
  -> homography rectification
  -> maze opening detection
  -> A* planning
  -> spline smoothing
  -> pixel-to-metric transform
  -> robot-base transform
  -> inverse kinematics
  -> joint-space resampling
  -> MPC tracking
  -> UR10e drawing
```

The figures below are the actual saved outputs from the final pipeline run.

## 1. Capture and AprilTag Localization

The robot first moves to a fixed overhead pose. A wrist-mounted Intel RealSense camera captures a color image at 1280 by 720. Four AprilTags around the maze are detected and used to locate the sheet.

<p align="center">
  <img src="./outputs/localized_input.png" alt="Localized input with AprilTags" width="650">
</p>
<p align="center"><em>Captured image with AprilTag detections and the localization overlay.</em></p>

The tag corners define a quadrilateral around the maze sheet. Since the maze is planar, the perspective projection can be represented by a homography. For image point `p` and rectified point `p'`:

```text
p' ~ H p
```

The homography `H` is estimated from four point correspondences. This flattens the maze without needing to separately calibrate the camera intrinsics and extrinsics.

## 2. Rectified Maze

The homography produces a flat, cropped maze image. This is the image used by the planner.

<p align="center">
  <img src="./outputs/maze_rectified.png" alt="Rectified maze" width="520">
</p>
<p align="center"><em>Rectified and cropped maze image used for planning.</em></p>

The metric scale is computed from the known AprilTag side length:

```text
meters_per_pixel = tag_side_meters / average_tag_edge_pixels
```

For the final run, the scale was about `0.69 mm/px`.

## 3. A* Maze Planning

The rectified maze is converted into a grid graph. Bright cells are free space and dark cells are walls. Walls are inflated before graph creation to keep the pen path away from wall pixels.

The planner detects boundary openings and validates them with an inward free-space probe. This avoids false openings caused by tags, shadows, glare, or crop margins. The final path starts at the right opening and ends at the left opening.

<p align="center">
  <img src="./outputs/astar_overlay.png" alt="A* path overlay" width="520">
</p>
<p align="center"><em>A* path overlay. Green is the start opening and blue is the goal opening.</em></p>

The planner also restricts the search to the maze wall bounding box so A* cannot route around the outside of the maze. The bounding box is based on the outer wall span, not just the largest dark connected component, because interior wall blobs can otherwise be mistaken for the full maze.

## 4. Spline Smoothing

The raw A* path is valid but jagged. A spline is fit through the path so the pen has a smoother reference to follow.

<p align="center">
  <img src="./outputs/astar_spline_overlay.png" alt="Spline-smoothed A* path overlay" width="520">
</p>
<p align="center"><em>Spline-smoothed path sent into the robot trajectory pipeline.</em></p>

The spline is saved as pixel coordinates in:

```text
outputs/spline_pixels.npy
```

## 5. Coordinate Transforms

The spline points are converted from pixels to robot-base coordinates.

First, rectified pixels are converted to metric maze coordinates:

```text
x_maze = u * meters_per_pixel
y_maze = v * meters_per_pixel
z_maze = 0
```

Then a homogeneous transform chain maps the maze point into the UR10e base frame:

```text
p_base = T_base_tag * T_tag_maze * p_maze
```

where:

- `T_base_tag` is the measured pose of the anchor AprilTag in the robot base frame.
- `T_tag_maze` is the measured offset from the anchor tag to the maze origin.
- A fixed axis calibration handles the maze image x-axis direction relative to the robot base x-axis.

The output waypoints are saved in:

```text
outputs/base_waypoints_xyz_m.npy
outputs/T_base_maze.npy
```

## 6. Inverse Kinematics

Each Cartesian waypoint is converted into a UR10e joint target. The robot is controlled at the pen tip, not at the flange, so the tool transform is included:

```text
T_base_pen(q) = FK(q) * T_tcp_pen
```

For each desired pen pose, the IK solves:

```text
FK(q) = T_base_pen_desired * inv(T_tcp_pen)
```

The implementation uses a consistent elbow-up UR10e solution branch so the arm does not switch configurations during the path.

The joint trajectory is saved in:

```text
outputs/joint_trajectory.npy
```

## 7. Joint-Space Resampling

The spline is evenly spaced in image space, but after IK the joint-space spacing is not uniform. Since the MPC consumes one point per tick, uneven joint spacing creates uneven speed. To fix this, the joint trajectory is resampled at uniform joint-space arc length.

For joint path length `L`, cruise speed `v_c`, and controller period `dt`:

```text
N_points = ceil(L / (v_c * dt))
```

This makes the reference velocity smooth and avoids acceleration spikes.

## 8. MPC Tracking

The drawing phase uses an MPC controller with a single-integrator joint model:

```text
q[k+1] = q[k] + dt * u[k]
```

where:

- `q` is the 6D joint state.
- `u` is the 6D joint velocity command.
- `dt = 1 / 75 s`.

The MPC tracks the next window of joint targets while enforcing:

- Joint velocity limits.
- Joint acceleration limits.
- Smooth command changes.

The first velocity command from each MPC solve is sent to the robot.

## 9. Robot Execution

The execution layer uses both `movej` and `speedj`:

- `movej` is used for static moves, such as moving to the overhead camera pose, moving above the maze start, lowering the pen to contact, and returning after the run.
- `speedj` is used during the drawing phase, where the MPC streams joint velocity commands.

Joint feedback is read from the UR real-time monitor instead of the slow default `getj()` interface. This gives high-rate feedback for the MPC loop.

## 10. Safety and CBFs

The repo includes runtime safety checks and optional CBF filtering.

Available safety layers:

- Joint limit checks.
- Link self-collision checks.
- Endpoint height CBF to keep the pen/tool above a minimum height.
- Optional post-MPC CBF-QP safety filter.

The relevant files are:

```text
src/cbf.py
src/simulation.py
src/urx_control_thread.py
src/combined_main.py
```

To enable runtime collision and joint-limit checks before `speedj` commands, set this in `src/urx_control_thread.py`:

```python
RUN_RUNTIME_SAFETY_CHECKS = True
```

To enable the CBF filter during MPC execution, set this in `src/simulation.py`:

```python
ENABLE_CBF = True
```

The CBF filter is constructed in `SharedTrajectoryState` in `src/combined_main.py`. By default it includes `EndpointHeightCBF`, which enforces a minimum endpoint height relative to the workspace plane. Extra constraints can be added in `src/cbf.py`, such as `JointLimitCBF`.

For final maze drawing, these filters may be disabled during debugging to inspect raw MPC tracking. For safer hardware testing, enable them and verify the resulting commands with telemetry.

## Hardware

- UR10e robot arm.
- Intel RealSense camera for overhead image capture.
- AprilTags around the maze sheet.
- Pen tool mounted to the robot.
- Printed maze on a flat drawing surface.
- Robot and control computer on the same network.

## Python Packages

Install the repo dependencies with:

```bash
pip install -r requirements.txt
```

The main packages are:

- `numpy` for numerical work.
- `opencv-contrib-python` for image processing and AprilTag/ArUco support.
- `pyrealsense2` for RealSense camera capture.
- `urx==0.11.0` for UR robot communication.
- `casadi` for the MPC optimization problem.
- `PyYAML` for parameter loading.
- `matplotlib` for plots and diagnostics.
- `cvxpy` for optimization utilities used by safety/filtering code.

## Main Files

```text
src/combined_main.py        Main pipeline and execution entry point
src/maze_localizer.py       AprilTag detection, homography, rectification
src/astar.py                Maze graph creation, opening detection, A*, spline overlays
src/mpc.py                  CasADi MPC controller
src/simulation.py           MPC execution loop and telemetry saving
src/trajectory_tracking.py  MPC thread wrapper
src/urx_control_thread.py   UR10e communication, movej and speedj commands
src/ur10e.py                UR10e kinematics and IK
src/utils.py                Geometry, transforms, safety utilities
tools/diagnose_trajectory_fk.py
tools/diagnose_mpc_commands.py
```

## How To Run

### Full robot run

Run from the repo root:

```bash
python3 -m src.combined_main --execute
```

This will:

1. Connect to the UR10e.
2. Move to the overhead camera pose.
3. Capture the maze image.
4. Run localization.
5. Plan the A* and spline path.
6. Move to the start.
7. Lower the pen.
8. Run MPC and draw the maze solution.

### Run using an existing image

```bash
python3 -m src.combined_main --no-capture --image outputs/captured_maze.png --execute
```

### Run planning only

```bash
python3 -m src.combined_main --no-capture --image outputs/captured_maze.png
```

This produces the planning and localization outputs without commanding the robot.

## Useful Outputs

After a run, the main outputs are:

```text
outputs/captured_maze.png          Raw RealSense capture
outputs/localized_input.png        AprilTag detections and localization overlay
outputs/maze_rectified.png         Rectified maze image
outputs/maze_map.png               Planner map
outputs/astar_overlay.png          Raw A* path
outputs/astar_spline_overlay.png   Smoothed spline path
outputs/spline_pixels.npy          Spline path in pixels
outputs/base_waypoints_xyz_m.npy   Cartesian robot-base waypoints
outputs/joint_trajectory.npy       IK joint trajectory
outputs/mpc_trace.npz              MPC and robot telemetry
outputs/mpc_telemetry.txt          Text telemetry summary
outputs/actual_run_overlay.png     Actual run overlay from measured robot motion
```

## Diagnostics

Forward-kinematics check:

```bash
python tools/diagnose_trajectory_fk.py outputs
```

MPC command diagnostic:

```bash
python tools/diagnose_mpc_commands.py outputs
```

These tools help check:

- Whether the planned joint trajectory maps back to the expected Cartesian path.
- Whether the actual robot motion matches the planned path.
- Whether MPC commands are saturating velocity or acceleration limits.
- Whether the path is being physically under-executed by the robot.

## Report

The project report is included here:

```text
report/UR10e_Maze_Solver.pdf
```

If the report PDF is not present, the LaTeX source is in:

```text
report/report.tex
```

## Notes

- The maze must stay fixed between image capture and drawing.
- The AprilTag-to-robot-base calibration must match the physical setup.
- The paper height and pen tool transform must be correct for clean contact.
- The system assumes the maze has two valid outer boundary openings.
- If the robot draws a scaled-down version of the path, check speed scaling, `speedj` streaming behavior, and execution telemetry.
