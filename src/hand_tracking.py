import cv2
import mediapipe as mp # python -m pip install mediapipe==0.10.21
import numpy as np


class MediaPipeHandTracker:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index

        # Desired physical span represented by the full screen width/height
        self.x_span_m = 0.50   # 50 cm across the image width
        self.y_span_m = 0.25   # 25 cm across the image height

        # Simple smoothing
        self.alpha = 0.25
        self.smoothed_pos = None

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def normalized_to_scaled_position(self, x_norm, y_norm):
        """
        MediaPipe gives normalized coordinates in image space:
            x_norm in [0, 1], left -> right
            y_norm in [0, 1], top -> bottom

        We map:
            center of image -> (0, 0)
            full width -> 0.50 m
            full height -> 0.25 m
            positive y -> upward
        """
        x_m = (x_norm - 0.5) * self.x_span_m
        y_m = (0.5 - y_norm) * self.y_span_m
        return np.array([x_m, y_m], dtype=float)

    def smooth(self, pos):
        if self.smoothed_pos is None:
            self.smoothed_pos = pos.copy()
        else:
            self.smoothed_pos = self.alpha * pos + (1.0 - self.alpha) * self.smoothed_pos
        return self.smoothed_pos.copy()

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        print("Press 'q' to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror for more natural interaction
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            tracked_text = "No hand detected"

            # Draw center crosshair
            cx, cy = w // 2, h // 2
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Wrist landmark
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                # Convert normalized image coords -> scaled meters
                raw_pos = self.normalized_to_scaled_position(wrist.x, wrist.y)
                pos = self.smooth(raw_pos)

                x_m, y_m = pos
                tracked_text = f"x = {x_m:+.3f} m, y = {y_m:+.3f} m"

                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Draw wrist point
                px = int(wrist.x * w)
                py = int(wrist.y * h)
                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)

                # Draw scaled coordinate text
                cv2.putText(
                    frame,
                    tracked_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Optional: print to terminal for downstream use
                print(f"\r{tracked_text}", end="")

            else:
                cv2.putText(
                    frame,
                    tracked_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Show scale info
            cv2.putText(
                frame,
                "Width mapped to 0.50 m | Height mapped to 0.25 m",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )

            cv2.imshow("MediaPipe Hand Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        print()
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    tracker = MediaPipeHandTracker(cam_index=0)
    tracker.run()