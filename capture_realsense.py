"""Capture a single color (and depth) image from any Intel RealSense camera.

Device-agnostic: it scans for the connected RealSense (any model / OS / USB
port), selects the best supported color profile, warms up auto-exposure, and
saves the shot. Works the same on Windows and Ubuntu.

Usage:
    python capture_realsense.py [output_path]
    python capture_realsense.py myphoto.png --serial 1234567890
    python capture_realsense.py shot.png --width 1920 --height 1080 --no-depth

Defaults to a timestamped PNG in the current directory if no path is given.
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np

import realsense_utils as rsu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default=None, help="output image path")
    parser.add_argument("--serial", default=None, help="select camera by serial number")
    parser.add_argument("--width", type=int, default=rsu.PREFERRED_COLOR[0])
    parser.add_argument("--height", type=int, default=rsu.PREFERRED_COLOR[1])
    parser.add_argument("--fps", type=int, default=rsu.PREFERRED_COLOR[2])
    parser.add_argument("--warmup", type=int, default=rsu.WARMUP_FRAMES,
                        help="frames to discard so auto-exposure can settle")
    parser.add_argument("--no-depth", dest="depth", action="store_false",
                        help="do not capture/save the depth frame")
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.getcwd(), f"realsense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    print("Starting RealSense pipeline...")
    pipeline, _, info = rsu.start_pipeline(
        serial=args.serial, want_depth=args.depth,
        preferred_color=(args.width, args.height, args.fps), warmup=args.warmup,
    )
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received from camera.")

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(out_path, color_image)
        print(f"Saved color image to: {out_path}")

        if args.depth:
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
