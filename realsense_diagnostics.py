"""Diagnose RealSense capture quality and settings.

Prints a full report so you can compare setups (e.g. why Ubuntu shots look
worse than Windows): connected devices + USB link speed, the active color
sensor settings (auto-exposure, exposure, gain, white balance, sharpness...),
image-quality metrics of a live frame (brightness, contrast, sharpness,
clipping), and an actual AprilTag detection check using the pipeline's detector.
Ends with PASS/WARN findings and concrete recommendations.

Usage:
    python realsense_diagnostics.py [--serial S] [--width W --height H --fps F]
    python realsense_diagnostics.py --save outputs/diag_capture.png
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

import realsense_utils as rsu
import maze_pipeline as mp

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(str(exc))


# Color-sensor options worth reporting (skipped automatically if unsupported).
REPORT_OPTIONS = [
    ("enable_auto_exposure", rs.option.enable_auto_exposure),
    ("exposure", rs.option.exposure),
    ("gain", rs.option.gain),
    ("enable_auto_white_balance", rs.option.enable_auto_white_balance),
    ("white_balance", rs.option.white_balance),
    ("brightness", rs.option.brightness),
    ("contrast", rs.option.contrast),
    ("gamma", rs.option.gamma),
    ("hue", rs.option.hue),
    ("saturation", rs.option.saturation),
    ("sharpness", rs.option.sharpness),
    ("backlight_compensation", rs.option.backlight_compensation),
    ("power_line_frequency", rs.option.power_line_frequency),
]

# Quality thresholds (heuristic, tuned for tag/maze capture).
MIN_SHARPNESS = 120.0     # Laplacian variance; below this looks soft/blurry
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 60.0, 200.0
MAX_CLIP_FRACTION = 0.05  # >5% blown-out highlights = overexposed


def image_metrics(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "resolution": (bgr.shape[1], bgr.shape[0]),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "sharpness_lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "clip_low_frac": float((gray < 5).mean()),
        "clip_high_frac": float((gray > 250).mean()),
        "gray": gray,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=rsu.PREFERRED_COLOR[0])
    parser.add_argument("--height", type=int, default=rsu.PREFERRED_COLOR[1])
    parser.add_argument("--fps", type=int, default=rsu.PREFERRED_COLOR[2])
    parser.add_argument("--warmup", type=int, default=rsu.WARMUP_FRAMES)
    parser.add_argument("--save", default=os.path.join("outputs", "diag_capture.png"),
                        help="where to save the diagnostic frame")
    args = parser.parse_args()

    findings = []  # (level, message)

    print("=" * 72)
    print("REALSENSE DIAGNOSTICS")
    print("=" * 72)

    # --- 1. Devices --------------------------------------------------------
    devices = rsu.list_devices()
    if not devices:
        raise SystemExit("No RealSense devices found. Check connection / udev rules / "
                         "run `rs-enumerate-devices`.")
    print(f"\n[Devices] {len(devices)} found:")
    for d in devices:
        print(f"  - {d.name} | serial {d.serial} | USB {d.usb_type or '?'} | FW {d.firmware} "
              f"| {d.product_line}")
        if d.usb_type and not d.is_usb3:
            findings.append(("WARN", f"{d.name} is on USB {d.usb_type} (not USB3). This "
                             "throttles bandwidth and is a common cause of lower image "
                             "quality / forced low resolution. Use a USB3 port + cable."))

    # --- 2. Supported color profiles --------------------------------------
    dev = rsu.select_device(args.serial)
    color_profs = rsu._video_profiles(dev.handle, rs.stream.color, rs.format.bgr8)
    print(f"\n[Color profiles] {dev.name} supports (BGR8):")
    print("  " + ", ".join(f"{w}x{h}@{f}" for w, h, f in color_profs[-12:]))
    best = max(color_profs, key=lambda p: (p[0] * p[1], p[2])) if color_profs else None
    if best:
        print(f"  highest available: {best[0]}x{best[1]}@{best[2]}")

    # --- 3. Start pipeline & read sensor settings --------------------------
    print("\n[Streaming] starting pipeline + warm-up...")
    pipeline, profile, info = rsu.start_pipeline(
        serial=args.serial, want_depth=False,
        preferred_color=(args.width, args.height, args.fps), warmup=args.warmup,
        verbose=True,
    )
    try:
        chosen = info["color"]
        if best and (chosen[0] * chosen[1]) < (best[0] * best[1]):
            findings.append(("WARN", f"Capturing at {chosen[0]}x{chosen[1]} but the camera "
                             f"supports up to {best[0]}x{best[1]}. Higher resolution gives "
                             "sharper tags; pass --width/--height to use it."))

        sensor = rsu.get_color_sensor(profile)
        print("\n[Color sensor settings]")
        auto_exp = None
        exposure_val = None
        if sensor is not None:
            for label, opt in REPORT_OPTIONS:
                if not sensor.supports(opt):
                    continue
                val = sensor.get_option(opt)
                try:
                    rng = sensor.get_option_range(opt)
                    rng_s = f"  [range {rng.min:g}..{rng.max:g}, default {rng.default:g}]"
                except Exception:
                    rng_s = ""
                print(f"  {label:28s} = {val:g}{rng_s}")
                if label == "enable_auto_exposure":
                    auto_exp = val
                if label == "exposure":
                    exposure_val = val
        else:
            print("  (could not locate the RGB sensor)")

        # --- 4. Capture a frame and measure quality ------------------------
        bgr = rsu.grab_color(pipeline)
        m = image_metrics(bgr)
        print("\n[Frame quality]")
        print(f"  resolution        = {m['resolution'][0]}x{m['resolution'][1]}")
        print(f"  brightness (mean) = {m['brightness_mean']:.1f}   (good ~{MIN_BRIGHTNESS:.0f}-{MAX_BRIGHTNESS:.0f})")
        print(f"  contrast (std)    = {m['contrast_std']:.1f}")
        print(f"  sharpness (LapVar)= {m['sharpness_lapvar']:.1f}   (good > {MIN_SHARPNESS:.0f})")
        print(f"  clipped dark      = {100*m['clip_low_frac']:.2f}%")
        print(f"  clipped bright    = {100*m['clip_high_frac']:.2f}%")

        # --- 5. Tag detection check ---------------------------------------
        corners, ids = mp.detect_tags(m["gray"])
        n_tags = 0 if ids is None else len(ids)
        found = [] if ids is None else sorted(int(i) for i in ids)
        print("\n[Tag detection] (AprilTag 36h11, pipeline detector)")
        print(f"  detected {n_tags} tags: {found}")

        # --- 6. Findings & recommendations --------------------------------
        if m["sharpness_lapvar"] < MIN_SHARPNESS:
            findings.append(("WARN", f"Low sharpness ({m['sharpness_lapvar']:.0f}). Causes: "
                             "USB2 link, motion/defocus, low resolution, or too few warm-up "
                             "frames. Steady the camera, use USB3, raise resolution."))
        if m["brightness_mean"] < MIN_BRIGHTNESS:
            findings.append(("WARN", "Image is underexposed (dark). Increase exposure/gain "
                             "or lighting; let auto-exposure settle (more --warmup frames)."))
        if m["brightness_mean"] > MAX_BRIGHTNESS:
            findings.append(("WARN", "Image is overexposed (bright). Lower exposure/gain."))
        if m["clip_high_frac"] > MAX_CLIP_FRACTION:
            findings.append(("WARN", f"{100*m['clip_high_frac']:.1f}% blown-out highlights "
                             "(glare). Reduce exposure or avoid reflections/plastic."))
        if auto_exp is not None and auto_exp == 0:
            findings.append(("INFO", f"Auto-exposure is OFF (manual exposure={exposure_val}). "
                             "For consistent shots across machines this is fine, but make "
                             "sure the fixed exposure suits the lighting."))
        if n_tags < 4:
            findings.append(("WARN", f"Only {n_tags}/4 corner tags detected. Improve "
                             "lighting/sharpness/resolution; the maze pipeline needs all 4."))
        else:
            findings.append(("PASS", "All 4 corner tags detected with current settings."))

        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, bgr)
            print(f"\nSaved diagnostic frame to: {args.save}")
    finally:
        pipeline.stop()

    print("\n" + "=" * 72)
    print("FINDINGS")
    print("=" * 72)
    order = {"WARN": 0, "INFO": 1, "PASS": 2}
    for level, msg in sorted(findings, key=lambda f: order.get(f[0], 9)):
        print(f"  [{level}] {msg}")
    if not any(level == "WARN" for level, _ in findings):
        print("\n  No warnings - capture settings look good.")


if __name__ == "__main__":
    main()
