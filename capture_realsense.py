"""Capture a single color image from an Intel RealSense camera (e.g. D435I) and save it.

Usage:
    python capture_realsense.py [output_path]

Defaults to a timestamped PNG in the current directory if no path is given.
The script also saves the aligned depth frame as a colorized PNG alongside the
color image (skipped automatically if depth is unavailable).
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import cv2
import pyrealsense2 as rs

# Stream config. The D435I color sensor supports 1280x720 @ 30fps.
COLOR_W, COLOR_H, FPS = 1280, 720, 30
WARMUP_FRAMES = 30  # let auto-exposure settle before grabbing the shot


def main():
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.getcwd(), f"realsense_{stamp}.png")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, FPS)

    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)

    try:
        # Warm up so auto-exposure/white-balance can converge.
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received from camera.")

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(out_path, color_image)
        print(f"Saved color image to: {out_path}")

        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colored = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            root, ext = os.path.splitext(out_path)
            depth_path = f"{root}_depth{ext}"
            cv2.imwrite(depth_path, depth_colored)
            print(f"Saved colorized depth to: {depth_path}")
    finally:
        pipeline.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
