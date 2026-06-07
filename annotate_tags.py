"""Annotate detected ArUco/AprilTag markers with their exact pixel locations.

Uses the same detector as the pipeline (maze_pipeline.make_detector_params) so
the reported coordinates match what the pipeline actually sees. For each tag it
draws the border, the 4 ordered corners (TL/TR/BR/BL) with their pixel (x, y),
and the center, and prints a coordinate table you can use to build a static
image->world transform.

Usage:
    python annotate_tags.py [input_image] [output_image]
Defaults to photo2.png -> outputs/aruco_detected.png
"""

import os
import sys

import cv2
import numpy as np

import maze_pipeline as mp

CORNER_NAMES = ["TL", "TR", "BR", "BL"]  # OpenCV ArUco corner order
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "photo2.png")
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else os.path.join(here, "outputs", "aruco_detected.png"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    image = cv2.imread(in_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = mp.detect_tags(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No tags detected.")

    order = np.argsort(ids)
    out = image.copy()

    print(f"Image: {in_path}  ({image.shape[1]}x{image.shape[0]} px)")
    print(f"Detected {len(ids)} tags (AprilTag 36h11).\n")
    header = f"{'ID':>3} | {'center (x,y)':>16} | " + " | ".join(
        f"{n} (x,y)".rjust(16) for n in CORNER_NAMES)
    print(header)
    print("-" * len(header))

    for i in order:
        c = corners[i]
        tag_id = int(ids[i])
        center = c.mean(axis=0)

        cv2.polylines(out, [c.astype(np.int32)], True, (0, 165, 255), 2, cv2.LINE_AA)
        for (cx, cy), name, col in zip(c, CORNER_NAMES, CORNER_COLORS):
            cv2.circle(out, (int(round(cx)), int(round(cy))), 5, col, -1)
            cv2.putText(out, f"{name}({cx:.0f},{cy:.0f})",
                        (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        ctr = (int(round(center[0])), int(round(center[1])))
        cv2.drawMarker(out, ctr, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(out, f"ID {tag_id}", (ctr[0] - 26, ctr[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"({center[0]:.1f},{center[1]:.1f})", (ctr[0] - 40, ctr[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        row = f"{tag_id:>3} | ({center[0]:7.1f},{center[1]:7.1f}) | " + " | ".join(
            f"({cx:6.1f},{cy:6.1f})" for cx, cy in c)
        print(row)

    cv2.imwrite(out_path, out)
    print(f"\nAnnotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
