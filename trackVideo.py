import argparse
from pathlib import Path

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO


def make_kalman_filter(x0: float, y0: float) -> KalmanFilter:
    """Constant-velocity Kalman filter over 2D position.
    State: [x, y, vx, vy]. Measurement: [x, y]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x0, y0, 0.0, 0.0])

    dt = 1.0  # one frame per step
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    kf.R *= 5.0        # measurement noise -- how much to trust detections
    kf.P *= 50.0        # initial state uncertainty
    kf.Q *= 0.5          # process noise -- how much the ball's motion can vary
    return kf


def detect_ball_center(model: YOLO, frame, conf: float):
    """Run detection on a frame, return the highest-confidence ball
    center (x, y) or None if nothing was detected above threshold."""
    result = model.predict(frame, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    # Take the highest-confidence detection if there are multiple
    confs = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[best_idx]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=str)
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to your fine-tuned best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=str, default=None,
                         help="Output video path (default: <input>_tracked.mp4)")
    parser.add_argument("--max-missed-frames", type=int, default=8,
                         help="How many consecutive missed detections before "
                              "we stop trusting the Kalman prediction and "
                              "consider tracking lost")
    parser.add_argument("--trail-length", type=int, default=20,
                         help="How many past positions to draw as a trail")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    out_path = Path(args.out) if args.out else video_path.with_name(f"{video_path.stem}_tracked.mp4")

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    kf = None
    missed_count = 0
    trail = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detect_ball_center(model, frame, args.conf)

        if kf is None and detection is not None:
            # First detection -- initialize the filter
            kf = make_kalman_filter(*detection)
            missed_count = 0
        elif kf is not None:
            kf.predict()
            if detection is not None:
                kf.update(np.array(detection))
                missed_count = 0
            else:
                missed_count += 1
                if missed_count > args.max_missed_frames:
                    # Lost the ball for too long -- reset tracking, wait
                    # for a fresh detection rather than keep drifting on
                    # pure prediction.
                    kf = None
                    trail.clear()

        if kf is not None:
            x, y = kf.x[0], kf.x[1]
            trail.append((int(x), int(y)))
            if len(trail) > args.trail_length:
                trail.pop(0)

            # Draw current position: filled circle if we have a real
            # detection this frame, hollow if this is a predicted
            # (bridged) position with no matching detection.
            color = (0, 0, 255) if detection is not None else (0, 165, 255)
            thickness = -1 if detection is not None else 2
            cv2.circle(frame, (int(x), int(y)), 8, color, thickness)

            # Draw trajectory trail
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Processed {frame_idx} frames -> {out_path}")


if __name__ == "__main__":
    main()