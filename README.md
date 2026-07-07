# PitchVision — Pitch Ball Detection & Trajectory Tracking

Detects and tracks a baseball through its flight path from broadcast pitch footage, using a fine-tuned YOLO object detector and frame-to-frame tracking.

## Overview

This project takes broadcast pitch clips and outputs an annotated video showing the ball's detected position and tracked trajectory frame-by-frame, with an estimated speed/arc overlay. It was built to explore small, fast-moving object detection — a genuinely hard case in computer vision compared to detecting larger, slower objects.

## Motivation


## Demo


## How It Works

1. **Data collection**: Pitch clips downloaded from [Baseball Savant](https://baseballsavant.mlb.com/), extracted into individual frames.
2. **Frame sampling**: A stratified sampling script selects a diverse, representative subset of frames per pitch (covering release → midflight → plate) rather than labeling every frame.
3. **Annotation**: Frames labeled in [Roboflow](https://roboflow.com/) with bounding boxes around the ball.
4. **Detection**: A YOLO model fine-tuned on the labeled dataset to detect the ball in a single frame.
5. **Tracking**: Per-frame detections linked across a video into a smooth trajectory using a Kalman filter, handling brief occlusion/missed detections.
6. **Output**: Annotated video with the tracked ball position and trajectory arc overlaid.

## Results

| Metric | Value |
|---|---|
| mAP50 | 0.94 |
| mAP50-95 | 0.66 |

<!-- Add a sentence or two interpreting these -- e.g. "The model reliably
detects the ball's presence and rough location, with some looseness in
exact box precision, likely due to [dataset size / label consistency /
motion blur]." -->

## Tech Stack

- **Detection**: YOLO (Ultralytics)
- **Annotation**: Roboflow
- **Tracking**: Kalman filter
- **Video/image processing**: OpenCV
- **Data source**: Baseball Savant (MLB Statcast broadcast clips)

## Project Structure

```
pitcharc/
├── data/
│   ├── frames/              # extracted video frames
│   └── selected_frames/     # curated subset chosen for labeling
├── scripts/
│   ├── download_clips.py        # pull pitch clips from Baseball Savant
│   ├── extract_frames.py        # video -> frame images
│   ├── select_frames_for_labeling.py  # stratified frame sampling
│   ├── train_ball_detector.py   # YOLO fine-tuning
│   ├── visualize_predictions.py # GT vs. predicted box comparison
│   └── track_video.py           # detection + Kalman tracking on video
├── runs/                    # training outputs, weights, metrics
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd pitcharc
pip install -r requirements.txt
```

## Usage

**1. Download pitch clips**
```bash
python scripts/download_clips.py --play-ids-csv path/to/statcast_export.csv
```

**2. Extract and select frames for labeling**
```bash
python scripts/extract_frames.py ...
python scripts/select_frames_for_labeling.py data/frames data/selected_frames --per-video 8
```

**3. Label in Roboflow, export in YOLO format**

**4. Train the detector**
```bash
python scripts/train_ball_detector.py path/to/data.yaml --model-size s
```

**5. Run tracking on a new video**
```bash
python scripts/track_video.py path/to/clip.mp4 --weights runs/ball_detector/ball_v1/weights/best.pt
```

## Challenges & Learnings

<!--
This section is often what a reviewer actually reads. Be specific and honest, e.g.:
- Small/fast object detection required higher training resolution than YOLO defaults
- Labeling was the primary bottleneck -- X hours for Y labeled frames
- mAP50 vs mAP50-95 gap indicated box precision issues rather than detection failures
- [Whatever you actually ran into with tracking / occlusion / etc.]
-->

## Future Work

- [ ] Real-time inference on live camera feed
- [ ] Pitch type classification (fastball, curveball, etc.) from trajectory shape
- [ ] Speed estimation calibrated against real Statcast velocity data
- [ ] Extend to full player + ball tracking

## Acknowledgments

- Pitch footage sourced from [Baseball Savant](https://baseballsavant.mlb.com/) (MLB Advanced Media). Used for personal/educational purposes; clips are not redistributed in this repo.
- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Roboflow](https://roboflow.com/).
