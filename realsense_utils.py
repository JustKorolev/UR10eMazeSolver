"""Cross-platform helpers for finding and streaming Intel RealSense cameras.

Works regardless of which machine / OS / USB port / camera model is in use:
it enumerates connected devices, picks one (by serial or the first found),
selects the best supported color (and depth) stream profile, and starts a
pipeline with an auto-exposure warm-up.

Used by capture_realsense.py and realsense_diagnostics.py so both share the same
device-selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "pyrealsense2 is not installed. Install the RealSense SDK + Python "
        "wrapper (pip install pyrealsense2, or build librealsense with "
        "-DBUILD_PYTHON_BINDINGS=ON on Ubuntu)."
    ) from exc


# Preferred color stream; selection falls back gracefully if unsupported.
PREFERRED_COLOR = (1280, 720, 30)
WARMUP_FRAMES = 30


def _info(dev, key):
    """Safely read a camera_info field, returning '' if unsupported."""
    return dev.get_info(key) if dev.supports(key) else ""


@dataclass
class DeviceInfo:
    name: str
    serial: str
    firmware: str
    product_line: str
    usb_type: str
    handle: object = field(repr=False, default=None)

    @property
    def is_usb3(self) -> bool:
        # usb_type strings look like "3.2", "2.1", etc.
        return self.usb_type.strip().startswith("3")


def list_devices() -> list[DeviceInfo]:
    """Enumerate all connected RealSense devices."""
    ctx = rs.context()
    out = []
    for dev in ctx.query_devices():
        out.append(
            DeviceInfo(
                name=_info(dev, rs.camera_info.name) or "RealSense device",
                serial=_info(dev, rs.camera_info.serial_number),
                firmware=_info(dev, rs.camera_info.firmware_version),
                product_line=_info(dev, rs.camera_info.product_line),
                usb_type=_info(dev, rs.camera_info.usb_type_descriptor),
                handle=dev,
            )
        )
    return out


def select_device(serial: Optional[str] = None) -> DeviceInfo:
    """Pick a device: by serial if given, else the first one found."""
    devices = list_devices()
    if not devices:
        raise RuntimeError(
            "No RealSense camera detected. Check the USB connection, and on "
            "Ubuntu that udev rules are installed and the SDK can see the device "
            "(try `rs-enumerate-devices`)."
        )
    if serial is not None:
        for d in devices:
            if d.serial == serial:
                return d
        raise RuntimeError(f"No RealSense with serial {serial}. Found: "
                           f"{[d.serial for d in devices]}")
    return devices[0]


def _video_profiles(device, stream_type, want_format):
    """All (w, h, fps) profiles of a stream type matching want_format."""
    profs = []
    for sensor in device.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() != stream_type or not p.is_video_stream_profile():
                continue
            if want_format is not None and p.format() != want_format:
                continue
            vp = p.as_video_stream_profile()
            profs.append((vp.width(), vp.height(), vp.fps()))
    return sorted(set(profs))


def choose_color_profile(device, preferred=PREFERRED_COLOR):
    """Choose a color (w, h, fps). Prefer `preferred`; else best available.

    'Best' = highest resolution, then highest fps (capped at 30 to avoid
    bandwidth issues), among BGR8 profiles.
    """
    profs = _video_profiles(device, rs.stream.color, rs.format.bgr8)
    if not profs:
        raise RuntimeError("Camera exposes no BGR8 color profiles.")
    if tuple(preferred) in profs:
        return tuple(preferred)
    capped = [p for p in profs if p[2] <= 30] or profs
    # max by (area, fps)
    return max(capped, key=lambda p: (p[0] * p[1], p[2]))


def choose_depth_profile(device, preferred=PREFERRED_COLOR):
    """Choose a depth (w, h, fps), or None if the device has no depth stream."""
    profs = _video_profiles(device, rs.stream.depth, rs.format.z16)
    if not profs:
        return None
    if tuple(preferred) in profs:
        return tuple(preferred)
    capped = [p for p in profs if p[2] <= 30] or profs
    return max(capped, key=lambda p: (p[0] * p[1], p[2]))


def start_pipeline(serial: Optional[str] = None, want_depth: bool = True,
                   preferred_color=PREFERRED_COLOR, warmup: int = WARMUP_FRAMES,
                   verbose: bool = True):
    """Start a streaming pipeline on the selected device.

    Returns (pipeline, profile, info_dict) where info_dict records the device
    and the chosen color/depth profiles. Remember to call pipeline.stop().
    """
    dev = select_device(serial)
    color = choose_color_profile(dev.handle, preferred_color)
    depth = choose_depth_profile(dev.handle, preferred_color) if want_depth else None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(dev.serial)
    config.enable_stream(rs.stream.color, color[0], color[1], rs.format.bgr8, color[2])
    if depth is not None:
        config.enable_stream(rs.stream.depth, depth[0], depth[1], rs.format.z16, depth[2])

    if verbose:
        print(f"Using {dev.name} (serial {dev.serial}, USB {dev.usb_type or '?'}, "
              f"FW {dev.firmware})")
        print(f"Color profile: {color[0]}x{color[1]} @ {color[2]}fps"
              + (f" | Depth: {depth[0]}x{depth[1]} @ {depth[2]}fps" if depth else ""))
        if dev.usb_type and not dev.is_usb3:
            print("WARNING: camera is on a USB 2.x link -> limited bandwidth and "
                  "image quality. Use a USB 3 port/cable for best results.")

    profile = pipeline.start(config)
    for _ in range(max(0, warmup)):  # let auto-exposure / white-balance settle
        pipeline.wait_for_frames()
    return pipeline, profile, {"device": dev, "color": color, "depth": depth}


def get_color_sensor(profile):
    """Return the RGB/color sensor from a started pipeline profile, or None."""
    dev = profile.get_device()
    for s in dev.query_sensors():
        name = s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else ""
        if "RGB" in name or "Color" in name:
            return s
    # Fallback: a sensor that supports exposure but is not the stereo module.
    for s in dev.query_sensors():
        if s.supports(rs.option.exposure):
            return s
    return None


def grab_color(pipeline) -> np.ndarray:
    """Grab one color frame as a BGR numpy array."""
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError("No color frame received from camera.")
    return np.asanyarray(color.get_data())
