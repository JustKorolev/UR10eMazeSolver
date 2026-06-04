import threading
import time
import os
import glob
from collections import deque
import src.utils as utils

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mediapipe as mp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

from src.ur10e import UR10e


class MediaPipeHandTracker:
    def __init__(self, cam_index=1, x_span_m=0.7, y_span_m=0.5, alpha=0.25):
        self.cam_index = cam_index

        # Screen-space spans mapped into robot base-frame motion:
        # screen horizontal -> robot y
        # screen vertical   -> robot z
        self.x_span_m = x_span_m
        self.y_span_m = y_span_m
        self.alpha = alpha

        self.smoothed_pos = None
        self.cap = None

        self.trail_points = deque(maxlen=60)   # recent wrist pixel positions

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.6
        )

    def start(self):
        if self.cap is None:
            candidates = [self.cam_index] + [i for i in range(5) if i != self.cam_index]
            for idx in candidates:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    self.cap = cap
                    self.cam_index = idx
                    print(f"[CAMERA] Opened camera index {idx}")
                    return
                cap.release()
            raise RuntimeError(
                f"Could not open any webcam (tried indices {candidates})."
            )

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.trail_points.clear()

        try:
            self.hands.close()
        except Exception:
            pass

    def reset_tracking_state(self):
        self.smoothed_pos = None
        self.trail_points.clear()

    def normalized_to_scaled_xy(self, x_norm, y_norm):
        x_scaled = (x_norm - 0.5) * self.x_span_m
        y_scaled = (0.5 - y_norm) * self.y_span_m
        # print(x_scaled, y_scaled)
        return np.array([x_scaled, y_scaled], dtype=float)

    def smooth(self, pos):
        if self.smoothed_pos is None:
            self.smoothed_pos = pos.copy()
        else:
            self.smoothed_pos = self.alpha * pos + (1.0 - self.alpha) * self.smoothed_pos
        return self.smoothed_pos.copy()

    def get_hand_position(self, draw=True):
        """
        Returns:
            pos_xy: np.array([x_scaled, y_scaled]) or None
            frame
        """
        if self.cap is None:
            self.start()

        ok, frame = self.cap.read()
        if not ok:
            return None, None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        smooth_screen_pos_xy = None
        current_px = None
        current_py = None

        if draw:
            cx, cy = w // 2, h // 2
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            raw_pos = self.normalized_to_scaled_xy(finger.x, finger.y)
            smooth_screen_pos_xy = self.smooth(raw_pos)

            current_px = int(finger.x * w)
            current_py = int(finger.y * h)
            self.trail_points.append((current_px, current_py))

            if draw:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Draw trail
                pts = list(self.trail_points)
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

                # Draw current wrist point
                cv2.circle(frame, (current_px, current_py), 8, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    f"y={smooth_screen_pos_xy[0]:+.3f} m, z_off={smooth_screen_pos_xy[1]:+.3f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        else:
            if draw:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # Still draw the old trail even if the hand is briefly lost
                pts = list(self.trail_points)
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

        if draw:
            cv2.putText(
                frame,
                f"X span = {self.x_span_m:.2f} m | Y span = {self.y_span_m:.2f} m",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )

        return smooth_screen_pos_xy, frame


class GUI:
    def __init__(self, root, shared_state, sampling_rate=50, mpc_horizon=1, workspace_offset=np.eye(4, 4)):
        self.root = root
        self.root.title("MediaPipe Hand Trajectory Controller")
        self.shared_state = shared_state
        self._workspace_offset = workspace_offset

        self.sampling_rate = sampling_rate
        self.mpc_horizon = mpc_horizon

        self.running = True
        self.state_lock = threading.Lock()

        self.mode = "idle"
        self.status_text = "Idle"

        self.request_record = False
        self.request_replay = False
        self.request_stop = False

        self.trajectory_ready = False
        self.replay_enabled = False
        self.replay_start_wall = 0.0

        self.record_time = 10.0

        self.recorded_positions = np.zeros((1, 3))
        self.recorded_times = np.zeros(1)
        self.recorded_orientations = np.zeros((1, 3))

        self.trajectories_dir = r".\trajectories"
        self.current_trajectory_name = ""

        self.MAX_LIVE_SAMPLES = 500
        self.live_t = deque(maxlen=self.MAX_LIVE_SAMPLES)
        self.live_x = deque(maxlen=self.MAX_LIVE_SAMPLES)
        self.live_y = deque(maxlen=self.MAX_LIVE_SAMPLES)

        self._stream_thread_running = False
        self._stream_thread = None

        self.hand_tracker = None

        self.local_x_min = -0.35
        self.local_x_max = 0.35
        self.local_y_min = -0.2
        self.local_y_max = 0.2

        self._last_valid_hand_pos_xy = None

        self.build_gui()

        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.ani = FuncAnimation(self.fig, self.update_plots, interval=30, cache_frame_data=False)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_gui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Record Time (s)").grid(row=0, column=0, sticky="w")
        self.record_var = tk.StringVar(value="10.0")
        self.record_entry = ttk.Entry(top, textvariable=self.record_var, width=10)
        self.record_entry.grid(row=0, column=1, padx=5, sticky="w")

        ttk.Label(top, text="Trajectory Name").grid(row=0, column=2, sticky="w")
        self.traj_name_var = tk.StringVar(value=self.get_next_trajectory_name())
        self.traj_name_entry = ttk.Entry(top, textvariable=self.traj_name_var, width=16)
        self.traj_name_entry.grid(row=0, column=3, padx=5, sticky="w")

        self.record_btn = ttk.Button(top, text="Record", command=self.start_recording)
        self.record_btn.grid(row=0, column=4, padx=5, sticky="w")

        ttk.Label(top, text="Load Trajectory").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.traj_list_var = tk.StringVar()
        self.traj_combo = ttk.Combobox(top, textvariable=self.traj_list_var, width=16, state="readonly")
        self.traj_combo.grid(row=1, column=1, padx=5, pady=(10, 0), sticky="w")

        self.load_btn = ttk.Button(top, text="Load", command=self.load_trajectory)
        self.load_btn.grid(row=1, column=2, padx=5, pady=(10, 0), sticky="w")

        self.replay_btn = ttk.Button(top, text="Replay", command=self.start_replay)
        self.replay_btn.grid(row=2, column=0, padx=5, pady=(10, 0), sticky="w")

        self.stop_btn = ttk.Button(top, text="Stop Replay", command=self.stop_replay)
        self.stop_btn.grid(row=2, column=1, padx=5, pady=(10, 0), sticky="w")

        self.follow_btn = ttk.Button(top, text="Start Following", command=self.shared_state.start_following)
        self.follow_btn.grid(row=2, column=2, padx=5, pady=(10, 0), sticky="w")

        self.stop_follow_btn = ttk.Button(top, text="Stop Following", command=self.stop_following_callback)
        self.stop_follow_btn.grid(row=2, column=3, padx=5, pady=(10, 0), sticky="w")

        self.home_btn = tk.Button(
            top,
            text="Home Arm",
            command=self.shared_state.request_home,
            bg="#4CAF50",
            fg="white",
            font=("TkDefaultFont", 9, "bold"),
            activebackground="#388E3C",
            activeforeground="white"
        )
        self.home_btn.grid(row=2, column=4, padx=5, pady=(10, 0), sticky="w")

        self.prerecorded_btn = tk.Button(
            top,
            text="Pre-recorded",
            command=self.shared_state.set_prerecorded_flag,
            bg="#FF00EA",
            fg="white",
            font=("TkDefaultFont", 9, "bold"),
            activebackground="#B91989",
            activeforeground="white"
        )
        self.prerecorded_btn.grid(row=2, column=5, padx=5, pady=(10, 0), sticky="w")

        ttk.Label(top, text="Plot Trajectory").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.plot_traj_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(top, textvariable=self.plot_traj_var, width=16, state="readonly")
        self.plot_combo.grid(row=3, column=1, padx=5, pady=(10, 0), sticky="w")

        self.plot_save_btn = ttk.Button(top, text="Plot and Save", command=self.plot_and_save_trajectory)
        self.plot_save_btn.grid(row=3, column=2, padx=5, pady=(10, 0), sticky="w")

        ttk.Label(top, text="Local Y Span Clamp (m)").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.local_y_limit_var = tk.StringVar(value="0.35")
        self.local_y_limit_entry = ttk.Entry(top, textvariable=self.local_y_limit_var, width=10)
        self.local_y_limit_entry.grid(row=4, column=1, padx=5, pady=(10, 0), sticky="w")

        ttk.Label(top, text="Local Z Span Clamp (m)").grid(row=4, column=2, sticky="w", pady=(10, 0))
        self.local_z_limit_var = tk.StringVar(value="0.2")
        self.local_z_limit_entry = ttk.Entry(top, textvariable=self.local_z_limit_var, width=10)
        self.local_z_limit_entry.grid(row=4, column=3, padx=5, pady=(10, 0), sticky="w")

        self.status_var = tk.StringVar(value="Status: Idle")
        self.status_label = ttk.Label(top, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=5, column=0, columnspan=6, sticky="w", pady=(12, 0))

        self.collision_var = tk.StringVar(value="")
        self.collision_label = tk.Label(
            top,
            textvariable=self.collision_var,
            fg="red",
            font=("TkDefaultFont", 10, "bold")
        )
        self.collision_label.grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 0))

        self.clear_collision_btn = tk.Button(
            top,
            text="Clear Collision",
            command=self._clear_collision,
            bg="#F44336",
            fg="white",
            font=("TkDefaultFont", 9, "bold"),
            activebackground="#D32F2F",
            activeforeground="white"
        )
        self.clear_collision_btn.grid(row=6, column=4, padx=5, pady=(4, 0), sticky="w")
        self.clear_collision_btn.grid_remove()

        self._poll_collision_status()

        self.fig = plt.Figure(figsize=(10, 7), dpi=100)
        self.ax_live = self.fig.add_subplot(2, 1, 1)
        self.ax_traj = self.fig.add_subplot(2, 1, 2, projection="3d")

        self.line_live_x, = self.ax_live.plot([], [], label="hand x")
        self.line_live_y, = self.ax_live.plot([], [], label="hand y")

        self.ax_live.set_title("Live Hand-Mapped Robot Coordinates")
        self.ax_live.set_xlabel("Time (s)")
        self.ax_live.set_ylabel("Position (m)")
        self.ax_live.grid(True)
        self.ax_live.legend(loc="upper right")

        self.traj_line, = self.ax_traj.plot([], [], [], linewidth=2)
        self.traj_point, = self.ax_traj.plot([], [], [], marker="o", markersize=7)

        self.ax_traj.set_title("Trajectory (Loaded / Replay)", pad=5)
        self.ax_traj.set_xlabel("X (m)")
        self.ax_traj.set_ylabel("Y (m)")
        self.ax_traj.set_zlabel("Z (m)")

        canvas_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.refresh_trajectory_list()

    def stop_following_callback(self):
        self._last_valid_hand_pos_xy = None

        if self.hand_tracker is not None:
            self.hand_tracker.reset_tracking_state()

        self.shared_state.stop_following()

    def get_next_trajectory_name(self):
        if not os.path.exists(self.trajectories_dir):
            os.makedirs(self.trajectories_dir)

        pattern = os.path.join(self.trajectories_dir, "traj_*.txt")
        existing_files = glob.glob(pattern)

        if not existing_files:
            return "traj_1"

        numbers = []
        for file in existing_files:
            basename = os.path.basename(file)
            if basename.startswith("traj_") and basename.endswith(".txt"):
                try:
                    num_str = basename[5:-4]
                    numbers.append(int(num_str))
                except ValueError:
                    continue

        if numbers:
            return f"traj_{max(numbers) + 1}"
        return "traj_1"

    def refresh_trajectory_list(self):
        if not os.path.exists(self.trajectories_dir):
            os.makedirs(self.trajectories_dir)

        pattern = os.path.join(self.trajectories_dir, "*.txt")
        files = glob.glob(pattern)
        trajectory_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        trajectory_names.sort()

        self.traj_combo["values"] = trajectory_names
        self.plot_combo["values"] = trajectory_names
        if trajectory_names:
            self.traj_combo.set(trajectory_names[0])
            self.plot_combo.set(trajectory_names[0])

    def save_trajectory(self, times, positions, orientations, name):
        if not os.path.exists(self.trajectories_dir):
            os.makedirs(self.trajectories_dir)

        filepath = os.path.join(self.trajectories_dir, f"{name}.txt")
        stats = self.analyze_trajectory_stats(times, positions, orientations)

        with open(filepath, "w") as f:
            f.write("# MediaPipe Hand Trajectory Data\n")
            f.write("# Format: time(s) x(m) y(m) z(m) roll(rad) pitch(rad) yaw(rad)\n")
            f.write(f"# Duration: {stats.get('duration', 0):.3f}s, Samples: {stats.get('samples', 0)}\n")
            f.write(f"# Max Linear Vel: {stats.get('max_linear_vel', 0):.3f} m/s, Max Angular Vel: {stats.get('max_angular_vel', 0):.3f} rad/s\n")
            f.write(f"# Max Linear Accel: {stats.get('max_linear_accel', 0):.3f} m/s^2, Max Angular Accel: {stats.get('max_angular_accel', 0):.3f} rad/s^2\n")
            f.write(f"# Total Distance: {stats.get('total_distance', 0):.3f}m, Total Rotation: {stats.get('total_rotation', 0):.3f}rad\n")
            f.write("# Robot base frame: +x forward, +y left, +z up\n")
            f.write("# Recorded in ZY plane with fixed X\n")

            for i in range(len(times)):
                f.write(
                    f"{times[i]:.6f} "
                    f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f} "
                    f"{orientations[i,0]:.6f} {orientations[i,1]:.6f} {orientations[i,2]:.6f}\n"
                )

    def load_trajectory(self):
        selected = self.traj_list_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a trajectory to load.")
            return

        filepath = os.path.join(self.trajectories_dir, f"{selected}.txt")
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Trajectory file {selected}.txt not found.")
            return

        try:
            times, positions, orientations = self.load_trajectory_data(filepath)

            if len(times) < 2:
                raise ValueError("Not enough data points in trajectory file.")

            with self.state_lock:
                self.recorded_times = np.array(times)
                self.recorded_positions = np.array(positions)
                self.recorded_orientations = np.array(orientations)
                self.trajectory_ready = True
                self.replay_enabled = False
                self.current_trajectory_name = selected

            self.set_status(f"Loaded trajectory: {selected}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load trajectory: {str(e)}")

    def load_trajectory_data(self, filepath):
        times = []
        positions = []
        orientations = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                parts = line.split()
                if len(parts) >= 7:
                    times.append(float(parts[0]))
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    orientations.append([float(parts[4]), float(parts[5]), float(parts[6])])

        return np.array(times), np.array(positions), np.array(orientations)

    def analyze_trajectory_stats(self, times, positions, orientations):
        if len(times) < 2:
            return {}

        dt = np.diff(times)
        pos_diff = np.diff(positions, axis=0)
        ori_diff = np.diff(orientations, axis=0)

        valid = dt > 0
        dt = dt[valid]
        pos_diff = pos_diff[valid]
        ori_diff = ori_diff[valid]

        if len(dt) < 1:
            return {"duration": 0, "samples": len(times)}

        linear_velocities = np.linalg.norm(pos_diff, axis=1) / dt
        angular_velocities = np.linalg.norm(ori_diff, axis=1) / dt

        linear_accelerations = np.diff(linear_velocities) / dt[:-1] if len(dt) > 1 else np.array([])
        angular_accelerations = np.diff(angular_velocities) / dt[:-1] if len(dt) > 1 else np.array([])

        return {
            "duration": times[-1] - times[0],
            "samples": len(times),
            "avg_sample_rate": len(times) / (times[-1] - times[0]) if times[-1] > times[0] else 0,
            "max_linear_vel": np.max(linear_velocities) if len(linear_velocities) > 0 else 0,
            "avg_linear_vel": np.mean(linear_velocities) if len(linear_velocities) > 0 else 0,
            "max_angular_vel": np.max(angular_velocities) if len(angular_velocities) > 0 else 0,
            "avg_angular_vel": np.mean(angular_velocities) if len(angular_velocities) > 0 else 0,
            "max_linear_accel": np.max(np.abs(linear_accelerations)) if len(linear_accelerations) > 0 else 0,
            "max_angular_accel": np.max(np.abs(angular_accelerations)) if len(angular_accelerations) > 0 else 0,
            "total_distance": np.sum(np.linalg.norm(pos_diff, axis=1)) if len(pos_diff) > 0 else 0,
            "total_rotation": np.sum(np.linalg.norm(ori_diff, axis=1)) if len(ori_diff) > 0 else 0
        }

    def plot_and_save_trajectory(self):
        selected = self.plot_traj_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a trajectory to plot.")
            return

        filepath = os.path.join(self.trajectories_dir, f"{selected}.txt")
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Trajectory file {selected}.txt not found.")
            return

        try:
            times, positions, orientations = self.load_trajectory_data(filepath)

            if len(times) < 2:
                raise ValueError("Not enough data points in trajectory file.")

            fig1 = plt.figure(figsize=(15, 10))

            ax1 = fig1.add_subplot(2, 3, 1, projection="3d")
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=2, alpha=0.8)
            ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="green", s=100, label="Start")
            ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="red", s=100, label="End")
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.set_zlabel("Z (m)")
            ax1.set_title("3D Trajectory")
            ax1.legend()
            ax1.grid(True)

            ax2 = fig1.add_subplot(2, 3, 2)
            ax2.plot(times, positions[:, 0], "r-", label="X", linewidth=2)
            ax2.plot(times, positions[:, 1], "g-", label="Y", linewidth=2)
            ax2.plot(times, positions[:, 2], "b-", label="Z", linewidth=2)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Position (m)")
            ax2.set_title("XYZ Position vs Time")
            ax2.legend()
            ax2.grid(True)

            ax3 = fig1.add_subplot(2, 3, 3)
            ax3.plot(times, np.degrees(orientations[:, 0]), "r-", label="Roll", linewidth=2)
            ax3.plot(times, np.degrees(orientations[:, 1]), "g-", label="Pitch", linewidth=2)
            ax3.plot(times, np.degrees(orientations[:, 2]), "b-", label="Yaw", linewidth=2)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Angle (deg)")
            ax3.set_title("Orientation vs Time")
            ax3.legend()
            ax3.grid(True)

            ax4 = fig1.add_subplot(2, 3, 4)
            ax4.plot(positions[:, 1], positions[:, 2], "b-", linewidth=2)
            ax4.scatter(positions[0, 1], positions[0, 2], color="green", s=100, label="Start")
            ax4.scatter(positions[-1, 1], positions[-1, 2], color="red", s=100, label="End")
            ax4.set_xlabel("Y (m)")
            ax4.set_ylabel("Z (m)")
            ax4.set_title("ZY Plane")
            ax4.legend()
            ax4.grid(True)
            ax4.axis("equal")

            ax5 = fig1.add_subplot(2, 3, 5)
            ax5.plot(positions[:, 0], positions[:, 2], "b-", linewidth=2)
            ax5.scatter(positions[0, 0], positions[0, 2], color="green", s=100, label="Start")
            ax5.scatter(positions[-1, 0], positions[-1, 2], color="red", s=100, label="End")
            ax5.set_xlabel("X (m)")
            ax5.set_ylabel("Z (m)")
            ax5.set_title("XZ View")
            ax5.legend()
            ax5.grid(True)
            ax5.axis("equal")

            ax6 = fig1.add_subplot(2, 3, 6)
            stats = self.analyze_trajectory_stats(times, positions, orientations)
            stats_text = (
                f"Trajectory Statistics:\n"
                f"Duration: {stats.get('duration', 0):.2f}s\n"
                f"Samples: {stats.get('samples', 0)}\n"
                f"Max Vel: {stats.get('max_linear_vel', 0):.3f} m/s\n"
                f"Avg Vel: {stats.get('avg_linear_vel', 0):.3f} m/s\n"
                f"Distance: {stats.get('total_distance', 0):.3f}m\n"
                f"Max Ang Vel: {stats.get('max_angular_vel', 0):.3f} rad/s\n"
                f"Max Accel: {stats.get('max_linear_accel', 0):.3f} m/s^2\n"
            )

            ax6.text(
                0.05, 0.95, stats_text,
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8)
            )
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.axis("off")
            ax6.set_title("Trajectory Statistics")

            plt.tight_layout()

            plot_filename = os.path.join(self.trajectories_dir, f"{selected}_plot.png")
            fig1.savefig(plot_filename, dpi=300, bbox_inches="tight")

            plt.show()

            self.set_status(f"Plot saved as {selected}_plot.png")

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot trajectory: {str(e)}")

    def set_status(self, text):
        with self.state_lock:
            self.status_text = text
        self.root.after(0, lambda: self.status_var.set(f"Status: {text}"))

    def start_recording(self):
        try:
            self.record_time = float(self.record_var.get())
            if self.record_time <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Record time must be a positive number.")
            return

        with self.state_lock:
            self.request_record = True

    def start_replay(self):
        with self.state_lock:
            if not self.trajectory_ready:
                messagebox.showwarning("No trajectory", "Load a trajectory first.")
                return
            self.request_replay = True

    def stop_replay(self):
        with self.state_lock:
            self.request_stop = True

    def set_axes_equal_3d(self, ax, x, y, z):
        if len(x) < 2:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            return

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        xmid = 0.5 * (x.min() + x.max())
        ymid = 0.5 * (y.min() + y.max())
        zmid = 0.5 * (z.min() + z.max())

        xr = max(x.max() - x.min(), 0.1)
        yr = max(y.max() - y.min(), 0.1)
        zr = max(z.max() - z.min(), 0.1)

        r = 0.5 * max(xr, yr, zr)

        ax.set_xlim(xmid - r, xmid + r)
        ax.set_ylim(ymid - r, ymid + r)
        ax.set_zlim(zmid - r, zmid + r)

    def _pose_to_joint_angles(self, position):

        try:
            # Return previous IK angles if new position is close enough to previous
            if self._last_valid_joints is not None:
                pos_change = np.linalg.norm(position - self._last_ik_pos)
                if pos_change < 0.001:
                    return self._last_valid_joints.copy()

            self._last_ik_pos = position.copy()
            self._last_ik_ori = np.zeros(3, dtype=float)

            T = self.local_pose_to_base_transform(position)
            joints = self._ik_robot.IK("elbow_up_2", T)
            return joints
        except:
            return None

    def _ensure_tracker(self):
        if self.hand_tracker is None:
            self.hand_tracker = MediaPipeHandTracker(
                cam_index=0,
                alpha=0.25
            )
        self.hand_tracker.start()

    def _read_mapping_settings(self):
        try:
            y_lim = abs(float(self.local_y_limit_var.get()))
            if y_lim < 1e-6:
                raise ValueError
            self.local_x_min = -y_lim
            self.local_x_max = y_lim
        except ValueError:
            self.local_x_min = -0.25
            self.local_x_max = 0.25

        try:
            z_lim = abs(float(self.local_z_limit_var.get()))
            if z_lim < 1e-6:
                raise ValueError
            self.local_y_min = -z_lim
            self.local_y_max = z_lim
        except ValueError:
            self.local_y_min = -0.125
            self.local_y_max = 0.125

    def local_hand_pose_from_tracking(self, pos_xy):
        """
        Convert hand tracker output into a LOCAL workspace-frame pose.

        Screen center -> [0, 0, 0]
        Motion only in local YZ plane.
        local x stays zero.
        """
        dx = np.clip(pos_xy[0], self.local_x_min, self.local_x_max)
        dy = np.clip(pos_xy[1], self.local_y_min, self.local_y_max)

        position_local = np.array([dx, dy, 0.0], dtype=float)
        orientation_local = np.zeros(3, dtype=float)
        return position_local, orientation_local

    def local_pose_to_base_transform(self, position_local):
        """
        Ignore workspace orientation for drawing motion.

        Screen center -> workspace_offset translation.
        Local motion affects only base-frame y and z.
        Base-frame x stays fixed at workspace_offset x.
        """
        position_local_list = position_local.tolist()
        T_local = utils.pose6_to_T([*position_local_list, 0, 0, 0])
        T_local_rotated = T_local
        pose_local_rotated = utils.T_to_pose6(T_local_rotated)
        pose_local_rotated[3:] = np.zeros(3)

        pose_home = utils.T_to_pose6(self._workspace_offset)
        pose_target = pose_home + pose_local_rotated
        T_target = utils.pose6_to_T(pose_target)

        # print(pose_home)
        # print(pose_local_rotated)
        # print(pose_target)

        return T_target

    def local_pose_to_base_position(self, position_local):
        T_base = self.local_pose_to_base_transform(position_local)
        return T_base[:3, 3].copy()

    def record_trajectory(self):
        self._ensure_tracker()
        self._read_mapping_settings()

        self.set_status(
            f"Recording hand trajectory for {self.record_time:.2f} s "
            f"as LOCAL YZ motion around workspace offset..."
        )

        raw_times = []
        positions = []
        orientations = []

        start_wall = time.time()

        with self.state_lock:
            self.live_t.clear()
            self.live_x.clear()
            self.live_y.clear()

        while self.running:
            elapsed = time.time() - start_wall
            if elapsed > self.record_time:
                break

            pos_zy, frame = self.hand_tracker.get_hand_position(draw=True)

            if frame is not None:
                remaining = max(0.0, self.record_time - elapsed)
                cv2.putText(
                    frame,
                    f"RECORDING  {remaining:.1f}s left",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )
                cv2.imshow("MediaPipe Hand Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if pos_zy is None:
                continue

            position_local, orientation_local = self.local_hand_pose_from_tracking(pos_zy)
            position_base = self.local_pose_to_base_position(position_local)

            raw_times.append(elapsed)
            positions.append(position_base)
            orientations.append(orientation_local)

            with self.state_lock:
                self.live_t.append(elapsed)
                self.live_x.append(position_local[0])
                self.live_y.append(position_local[0])

        try:
            cv2.destroyWindow("MediaPipe Hand Tracker")
        except Exception:
            pass

        if len(raw_times) < 10:
            raise RuntimeError("Not enough hand samples recorded.")

        return np.array(raw_times), np.array(positions), np.array(orientations)

    def stream_trajectory_to_window(self):
        self.set_status("Starting MediaPipe hand stream to MPC window...")
        print("[STREAM] MediaPipe trajectory streaming started")

        self._ik_robot = UR10e(dt=1.0 / self.sampling_rate)
        self._last_valid_joints = None
        self._last_ik_pos = np.zeros(3)
        self._last_ik_ori = np.zeros(3)
        self._ik_warmup_samples = 0

        try:
            self._ensure_tracker()
        except Exception as e:
            self.set_status(f"Camera error: {e}")
            print(f"[STREAM] Camera error: {e}")
            self._stream_thread_running = False
            self._stream_thread = None
            return

        self._last_valid_hand_pos_xy = None
        self.hand_tracker.reset_tracking_state()

        self._read_mapping_settings()

        dt_stream = 1.0 / float(self.sampling_rate)
        stream_count = 0
        stream_time = 0.0
        lost_hand_count = 0

        position = np.zeros(3, dtype=float)   # local workspace-frame pose
        orientation = np.zeros(3, dtype=float)

        with self.state_lock:
            self.live_t.clear()
            self.live_x.clear()
            self.live_y.clear()

        start_wall = time.time()

        try:
            while self.running:
                with self.shared_state.lock:
                    following = self.shared_state.following_trajectory

                if not following:
                    self.set_status("MediaPipe trajectory streaming stopped")
                    break

                dt = dt_stream

                pos_zy, frame = self.hand_tracker.get_hand_position(draw=True)

                if frame is not None:
                    cv2.imshow("MediaPipe Hand Tracker", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        with self.shared_state.lock:
                            self.shared_state.following_trajectory = False
                        self.set_status("Stopped by user")
                        break

                if pos_zy is None:
                    lost_hand_count += 1

                    if self._last_valid_hand_pos_xy is not None:
                        pos_zy = self._last_valid_hand_pos_xy.copy()
                    else:
                        if self._last_valid_joints is not None:
                            self.shared_state.append_joint_target(self._last_valid_joints.copy(), dt)

                        if lost_hand_count % 20 == 0:
                            print(f"[STREAM] Hand lost for {lost_hand_count} frames (no prior hand position)")
                        continue
                else:
                    self._last_valid_hand_pos_xy = pos_zy.copy()
                    lost_hand_count = 0


                t_live = time.time() - start_wall

                position, orientation = self.local_hand_pose_from_tracking(pos_zy)
                position_base = self.local_pose_to_base_position(position)

                with self.state_lock:
                    self.live_t.append(t_live)
                    self.live_x.append(position[0])
                    self.live_y.append(position[1])

                if stream_count == 0:
                    joints = self._pose_to_joint_angles(position)
                    if joints is None:
                        joints = self._ik_robot.get_initial_pose()
                    self._last_valid_joints = joints.copy()
                    self.shared_state.append_joint_target(joints, dt)
                    stream_count += 1
                    print(f"[STREAM] Initial joints (deg): {np.degrees(joints).round(1)}")
                    continue

                if stream_count < self._ik_warmup_samples:
                    joints = self._last_valid_joints.copy()
                else:
                    joints = self._pose_to_joint_angles(position)
                    if joints is not None:
                        self._last_valid_joints = joints.copy()
                    else:
                        joints = self._last_valid_joints.copy()

                prev_joints = self.shared_state._last_joint_target
                self.shared_state.append_joint_target(joints, dt)

                stream_count += 1
                stream_time += dt

                if stream_count % 20 == 0:
                    wlen = len(self.shared_state.trajectory_window)
                    elapsed_s = stream_time

                    if prev_joints is not None and dt > 1e-6:
                        v_req = np.abs(joints - prev_joints) / dt
                        worst_j = np.argmax(v_req)
                        worst_v = v_req[worst_j]
                        feasible = "FEASIBLE" if worst_v <= 0.5 else f"FAST j{worst_j}:{worst_v:.2f}rad/s"
                    else:
                        feasible = "INIT"

                    print(
                        f"[STREAM] t={elapsed_s:.2f}s | {stream_count} pts | "
                        f"window={wlen} | {feasible} | "
                        f"local_yz=[{position[1]:.3f},{position[2]:.3f}] | "
                        f"base_xyz=[{position_base[0]:.3f},{position_base[1]:.3f},{position_base[2]:.3f}]"
                    )

        except Exception as e:
            self.set_status(f"Stream error: {e}")
            print(f"[STREAM] Error: {e}")
        finally:
            self._last_valid_hand_pos_xy = None

            if self.hand_tracker is not None:
                self.hand_tracker.reset_tracking_state()

            try:
                cv2.destroyWindow("MediaPipe Hand Tracker")
            except Exception:
                pass

            print(f"[STREAM] Stopped after {stream_count} points")
            self._stream_thread_running = False
            self._stream_thread = None

    def worker_loop(self):
        while self.running:
            do_record = False
            do_replay = False
            do_stop = False

            with self.state_lock:
                if self.request_record:
                    self.request_record = False
                    do_record = True
                if self.request_replay:
                    self.request_replay = False
                    do_replay = True
                if self.request_stop:
                    self.request_stop = False
                    do_stop = True

            with self.shared_state.lock:
                following = self.shared_state.following_trajectory

            if following:
                if self._stream_thread is None or not self._stream_thread.is_alive():
                    self._stream_thread_running = True
                    self._stream_thread = threading.Thread(
                        target=self.stream_trajectory_to_window,
                        daemon=True
                    )
                    self._stream_thread.start()

            try:
                if do_record:
                    trajectory_name = self.traj_name_var.get().strip()
                    if not trajectory_name:
                        trajectory_name = self.get_next_trajectory_name()

                    times, positions, orientations = self.record_trajectory()
                    self.save_trajectory(times, positions, orientations, trajectory_name)

                    with self.state_lock:
                        self.recorded_times = times
                        self.recorded_positions = positions
                        self.recorded_orientations = orientations
                        self.trajectory_ready = True
                        self.replay_enabled = False
                        self.current_trajectory_name = trajectory_name

                    self.root.after(0, self.refresh_trajectory_list)
                    self.root.after(0, lambda: self.traj_name_var.set(self.get_next_trajectory_name()))

                    self.set_status(f"Recording complete. Saved as {trajectory_name}.")

                elif do_replay:
                    with self.state_lock:
                        if self.trajectory_ready:
                            self.replay_enabled = True
                            self.mode = "replaying"
                    self.replay_start_wall = time.time()
                    self.set_status("Replay enabled.")

                elif do_stop:
                    with self.state_lock:
                        self.replay_enabled = False
                        self.mode = "idle"
                    self.set_status("Replay stopped.")

            except Exception as e:
                self.set_status(f"Error: {e}")

            time.sleep(0.02)

    def update_plots(self, frame):
        with self.state_lock:
            lt = list(self.live_t)
            lx = list(self.live_x)
            ly = list(self.live_y)

            traj_ready = self.trajectory_ready
            replay_enabled = self.replay_enabled
            traj_t = self.recorded_times.copy()
            traj_p = self.recorded_positions.copy()
            replay_start = self.replay_start_wall

        if len(lt) > 0:
            t0 = lt[0]
            tp = [x - t0 for x in lt]

            self.line_live_x.set_data(tp, lx)
            self.line_live_y.set_data(tp, ly)

            xmin = tp[0]
            xmax = tp[-1] if tp[-1] > 1.0 else 1.0
            self.ax_live.set_xlim(xmin, xmax)

            vals = lx + ly
            ymin = min(vals)
            ymax = max(vals)
            if ymin == ymax:
                ymin -= 0.1
                ymax += 0.1
            self.ax_live.set_ylim(ymin - 0.05, ymax + 0.05)
        else:
            self.line_live_x.set_data([], [])
            self.line_live_y.set_data([], [])

        if traj_ready and len(traj_t) >= 2 and len(traj_p) >= 2:
            if replay_enabled:
                duration = traj_t[-1]
                if duration <= 0:
                    duration = 1.0
                tplay = (time.time() - replay_start) % duration
                idx = np.searchsorted(traj_t, tplay, side="right")
                idx = max(1, min(idx, len(traj_t)))
                seg = traj_p[:idx]
            else:
                seg = traj_p

            if len(seg) > 0:
                self.traj_line.set_data(seg[:, 0], seg[:, 1])
                self.traj_line.set_3d_properties(seg[:, 2])

                self.traj_point.set_data([seg[-1, 0]], [seg[-1, 1]])
                self.traj_point.set_3d_properties([seg[-1, 2]])

                self.set_axes_equal_3d(self.ax_traj, traj_p[:, 0], traj_p[:, 1], traj_p[:, 2])
        else:
            self.traj_line.set_data([], [])
            self.traj_line.set_3d_properties([])
            self.traj_point.set_data([], [])
            self.traj_point.set_3d_properties([])

        self.canvas.draw_idle()

        return (
            self.line_live_x,
            self.line_live_y,
            self.traj_line,
            self.traj_point
        )

    def _poll_collision_status(self):
        with self.shared_state.lock:
            detected = self.shared_state.collision_detected
            reason = self.shared_state.collision_reason

        if detected:
            self.collision_var.set(f"COLLISION: {reason}")
            self.clear_collision_btn.grid()
        else:
            self.collision_var.set("")
            self.clear_collision_btn.grid_remove()

        self.root.after(200, self._poll_collision_status)

    def _clear_collision(self):
        self.shared_state.clear_collision()

    def on_close(self):
        self.running = False
        with self.shared_state.lock:
            self.shared_state.shutdown = True
            self.shared_state.robot_enabled = False
            self.shared_state.u_curr = np.zeros((6, 1))

        try:
            if self.hand_tracker is not None:
                self.hand_tracker.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.root.destroy()


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = IMUGUI(root, shared_state)
#     root.mainloop()