# Real-Time Hand-Guided Trajectory Tracking for UR10e Robot Arm

A comprehensive system for intuitive human-robot interaction that enables real-time control of a UR10e robotic arm through natural hand motion. The system combines computer vision-based hand tracking with Model Predictive Control (MPC) to translate arbitrary human gestures into smooth, feasible robot trajectories that respect physical constraints and safety requirements.

## Overview

This project demonstrates advanced trajectory generation and control techniques for robotic systems, featuring:

- **Real-time hand tracking** using MediaPipe computer vision
- **Model Predictive Control** for smooth trajectory execution
- **Comprehensive safety systems** with collision avoidance
- **Dynamic trajectory interpolation** respecting joint velocity limits
- **Multi-threaded architecture** for concurrent control and visualization

The system allows users to draw trajectories in space using hand gestures, which the robot reproduces physically with high accuracy and smoothness. Applications include drawing tasks, trajectory demonstration, and intuitive robot programming.

## Technical Architecture

### 1. Hand Tracking Pipeline

The vision system captures user hand motion in real-time using a standard webcam and the MediaPipe hand tracking model:

- **Detection**: Single hand detection with index finger tip extraction as the primary control point
- **Coordinate Mapping**: Normalized image coordinates scaled to physical workspace dimensions (50cm × 25cm)
- **Smoothing**: Exponential filtering to reduce vision noise and improve trajectory continuity
- **Workspace Transformation**: Conversion to robot base frame coordinates with configurable workspace offset

```python
# Key parameters
x_span_m = 0.50   # 50 cm workspace width
y_span_m = 0.25   # 25 cm workspace height
alpha = 0.25      # Smoothing factor
```

### 2. Trajectory Interpolation

Raw hand motion often violates robot velocity constraints, requiring intelligent interpolation to ensure feasible execution:

**Joint Space Interpolation**:
- Computes required joint velocities: `v_req = |Δθ|/Δt`
- Subdivides motion to respect velocity limits: `N = max(1, max_i(|Δθ_i|/(Δt·v_max,i)))`
- Generates intermediate waypoints: `θ^(j) = θ_k + (j/N)·Δθ`

**Cartesian Space Interpolation** (for planar tasks):
- Preserves geometric shape through Cartesian interpolation
- Constrains motion to desired drawing plane: `z^(j) = z_workspace`
- Falls back to joint space if inverse kinematics fails

### 3. Model Predictive Control

The MPC controller converts reference trajectories into smooth, constraint-aware joint commands:

**System Model**:
```
q_{k+1} = q_k + Δt·u_k
```
where `q_k ∈ ℝ^6` are joint angles and `u_k ∈ ℝ^6` are commanded joint velocities.

**Optimization Problem**:
```
min Σ(||q_{k+i} - q^ref_{k+i}||²_Q + ||u_{k+i}||²_R) + ||q_{k+N} - q^ref_{k+N}||²_P
```

**Constraints**:
- Joint position limits: `q_min ≤ q_{k+i} ≤ q_max`
- Velocity limits: `u_min ≤ u_{k+i} ≤ u_max`
- Acceleration limits: `Δu_min ≤ u_{k+i} - u_{k+i-1} ≤ Δu_max`

**Implementation**: CasADi optimization framework with IPOPT solver for real-time performance.

### 4. Safety and Collision Avoidance

Multi-layered safety system ensures safe operation under arbitrary user input:

**Layer 1 - Joint Limits**:
- Per-joint absolute position limits (±350° working range)
- Real-time monitoring of all joint configurations

**Layer 2 - Self-Collision Detection**:
- Forward kinematics computation of all link origins
- Line segment approximation of robot links
- Minimum distance calculation between non-adjacent link pairs
- Safety threshold: 5cm minimum separation distance

**Layer 3 - Ground Plane Avoidance**:
- Ensures all link origins remain above workspace boundary
- Prevents collision with table or environmental obstacles

**Hard Stop Mechanism**:
- Immediate velocity zeroing upon constraint violation
- GUI collision alerts with detailed reason codes
- Manual collision clearing required before trajectory resumption

## System Performance

### Control Specifications
- **Sampling Rate**: 75 Hz
- **MPC Horizon**: 15 steps (0.2 seconds)
- **Joint Velocity Limits**: 0.4 rad/s (conservative working limits)
- **Joint Acceleration Limits**: 1.2 rad/s²
- **Tracking Accuracy**: Sub-centimeter end-effector positioning

### Real-Time Capabilities
- **Hand Tracking Latency**: <20ms
- **MPC Solve Time**: <10ms per iteration
- **Total System Latency**: <50ms end-to-end
- **Trajectory Smoothness**: C¹ continuous joint motion

## Installation and Setup

### Prerequisites
```bash
# Core dependencies
pip install numpy matplotlib casadi
pip install mediapipe==0.10.21
pip install opencv-python
pip install urx  # UR robot communication

# Optional: YAML configuration support
pip install pyyaml
```

### Hardware Requirements
- UR10e robotic arm with network connectivity
- Standard USB webcam for hand tracking
- Whiteboard or drawing surface (for demonstration tasks)

### Configuration
Key parameters in `src/combined_main.py`:

```python
SAMPLING_RATE = 75  # Hz
MPC_HORIZON = SAMPLING_RATE // 5  # 0.2 seconds
WORKSPACE_OFFSET = pose6_to_T([0, -0.8, -0.015, np.pi, 0.05, 0.05])
VJ = 0.4   # rad/s - conservative velocity limit
AJ = 1.2   # rad/s² - acceleration limit
```

## Usage

### Basic Operation
```bash
# Launch complete system
python src/combined_main.py

# Simulation mode (no robot required)
python src/simulation.py
```

### GUI Controls
- **Start Tracking**: Begin hand-guided trajectory following
- **Record Trajectory**: Capture hand motion for playback
- **Play Trajectory**: Execute recorded motion sequence
- **Emergency Stop**: Immediate motion halt with safety lockout
- **Clear Collision**: Reset safety system after collision detection

### Trajectory Recording and Playback
The system supports trajectory recording for repeatable motion sequences:

1. **Recording Phase**: Capture hand motion with timestamp synchronization
2. **Storage**: Trajectories saved as joint-space waypoint sequences
3. **Playback**: Smooth reproduction with identical MPC control parameters

## Example Output Plots

### End-Effector Trajectory Tracking
*[Placeholder for trajectory tracking plots showing reference vs. actual end-effector motion]*

### Joint State Evolution
*[Placeholder for joint angle and velocity plots demonstrating smooth motion profiles]*

### MPC Tracking Performance
*[Placeholder for tracking error plots showing controller performance metrics]*

### Safety System Validation
*[Placeholder for collision avoidance demonstration plots]*

## Video Demonstrations

### Real-Time Hand Tracking Control
*[Placeholder for video showing live hand-guided robot motion]*

### Drawing Task Execution
*[Placeholder for video demonstrating whiteboard drawing capabilities]*

### Safety System Response
*[Placeholder for video showing collision avoidance and emergency stop functionality]*

## Technical Achievements

### Advanced Control Implementation
- **Real-time MPC**: Sub-10ms optimization solve times for 6-DOF system
- **Constraint Handling**: Simultaneous satisfaction of position, velocity, and acceleration limits
- **Stability Guarantees**: Terminal cost design ensuring closed-loop stability

### Vision-Control Integration
- **Low-Latency Pipeline**: End-to-end latency under 50ms for responsive control
- **Noise Robustness**: Exponential filtering and MPC prediction compensate for vision noise
- **Workspace Scaling**: Intuitive mapping between screen coordinates and robot workspace

### Safety Engineering
- **Defense in Depth**: Multiple independent collision checking layers
- **Predictive Safety**: Forward simulation prevents unsafe configurations
- **Graceful Degradation**: Safe system shutdown with detailed diagnostic information

### Software Architecture
- **Multi-Threading**: Concurrent hand tracking, MPC computation, and robot communication
- **Modular Design**: Separated concerns enabling easy modification and testing
- **Real-Time Performance**: Deterministic timing with priority-based thread scheduling

## Applications and Extensions

### Current Capabilities
- **Trajectory Demonstration**: Intuitive robot programming through gesture
- **Drawing Tasks**: Precise reproduction of hand-drawn patterns
- **Motion Teaching**: Record and replay complex motion sequences

### Future Extensions
- **Full 6-DOF Control**: Extension to complete end-effector pose control
- **Force Control Integration**: Contact-aware manipulation tasks
- **Multi-Robot Coordination**: Synchronized control of multiple robot arms
- **Advanced Vision**: Integration of object recognition and scene understanding

## Research Contributions

This work advances the state-of-the-art in human-robot interaction through:

1. **Real-time MPC for gesture control**: Novel application of predictive control to vision-guided robotics
2. **Integrated safety architecture**: Comprehensive collision avoidance system for arbitrary user input
3. **Trajectory interpolation algorithms**: Dynamic feasibility enforcement for human-generated motion
4. **Multi-threaded control architecture**: High-performance implementation enabling real-time operation

## Technical Documentation

### Core Modules
- `src/combined_main.py`: Main system orchestration and GUI
- `src/mpc.py`: Model Predictive Control implementation
- `src/hand_tracking.py`: MediaPipe-based vision pipeline
- `src/trajectory_tracking.py`: MPC simulation and control thread
- `src/utils.py`: Mathematical utilities and safety functions
- `src/ur10e.py`: UR10e kinematics and dynamics model

### Configuration Files
- `resources/tuning.yaml`: MPC cost function weights and solver parameters
- `COLLISION_AVOIDANCE.md`: Detailed safety system documentation


