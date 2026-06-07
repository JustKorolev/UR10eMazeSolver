"""Detect and annotate all ArUco tags in an image.

Single-file utility: loads an image, searches across the common predefined
ArUco dictionaries (since the tag family is unknown), draws the detected marker
borders + IDs, and writes an annotated copy next to the input.

Usage:
    python detect_aruco.py [input_image] [output_image]

Defaults to resources/Aruco_listsjpg.jpg if no arguments are given.
"""

import os
import sys

import cv2
import numpy as np

# Candidate dictionaries to probe. The image's tag family is unknown, so we try
# the standard families and keep whichever finds the most markers.
CANDIDATE_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def make_detector_params():
    """Detector parameters tuned to recover low-contrast / blurry tags.

    The default OpenCV parameters miss tags whose borders are soft or whose
    local contrast is low (e.g. a corner of a printed sheet under uneven
    lighting). Widening the adaptive-threshold sweep, loosening the polygon
    approximation, and enabling sub-pixel corner refinement recovers them.
    """
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.aprilTagQuadDecimate = 0.0
    return params


def detect_with_dict(gray, dict_name, params):
    """Run detection with a single named dictionary. Returns (corners, ids)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def best_detection(gray):
    """Probe all candidate dictionaries and return the family detecting the most tags.

    Returns (dict_name, corners, ids, count). Detection runs with the tuned
    parameters from :func:`make_detector_params`, and the dictionary that
    decodes the most markers wins.
    """
    params = make_detector_params()
    best = (None, [], None, 0)  # (dict_name, corners, ids, count)
    for dict_name in CANDIDATE_DICTS:
        corners, ids = detect_with_dict(gray, dict_name, params)
        count = 0 if ids is None else len(ids)
        if count > best[3]:
            best = (dict_name, corners, ids, count)
    return best


def annotate(image, corners, ids):
    """Draw marker borders, corner dots, and an ID label on each detected tag."""
    out = image.copy()
    cv2.aruco.drawDetectedMarkers(out, corners, ids)
    if ids is None:
        return out
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape(4, 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(
            out, f"ID {int(marker_id)}", (center[0] - 40, center[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA,
        )
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(here, "resources", "Aruco_listsjpg.jpg")

    in_path = sys.argv[1] if len(sys.argv) > 1 else default_in
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_annotated{ext}"

    image = cv2.imread(in_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dict_name, corners, ids, count = best_detection(gray)

    if count == 0:
        print("No ArUco markers detected with any candidate dictionary.")
        cv2.imwrite(out_path, image)
        return

    detected_ids = sorted(int(i) for i in ids.flatten())
    print(f"Best dictionary: {dict_name}")
    print(f"Detected {count} markers: IDs {detected_ids}")

    annotated = annotate(image, corners, ids)
    cv2.imwrite(out_path, annotated)
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
