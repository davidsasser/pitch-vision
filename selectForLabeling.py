import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

FRAME_PATTERN = re.compile(r"^pitch_(?P<play_id>.+)_frame_(?P<frame_num>\d+)\.jpg$")


def group_frames_by_play(frames_dir: Path) -> dict[str, list[Path]]:
    """Group flat-folder frame files by play_id, sorted by frame number."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    unmatched = 0
    for p in frames_dir.glob("*.jpg"):
        m = FRAME_PATTERN.match(p.name)
        if not m:
            unmatched += 1
            continue
        groups[m.group("play_id")].append((int(m.group("frame_num")), p))

    if unmatched:
        print(f"Warning: {unmatched} files didn't match the expected naming pattern and were skipped")

    return {play_id: [p for _, p in sorted(frames)] for play_id, frames in groups.items()}


def motion_scores(frame_paths: list[Path]) -> list[float]:
    """Return a motion score per frame based on diff with previous frame."""
    scores = [0.0]
    prev_gray = None
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            scores.append(0.0)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            scores.append(float(np.mean(diff)))
        prev_gray = gray
    return scores[: len(frame_paths)]


def select_from_video(frame_paths: list[Path], n_select: int) -> list[Path]:
    """Pick n_select frames from one video's frames: evenly spaced
    positions, with motion score used as a tiebreaker/bias."""
    if len(frame_paths) <= n_select:
        return frame_paths

    scores = motion_scores(frame_paths)

    # Split the sequence into n_select buckets, pick the highest-motion
    # frame within each bucket. This guarantees coverage across the
    # whole clip (release -> plate) while still favoring likely-ball frames.
    bucket_edges = np.linspace(0, len(frame_paths), n_select + 1, dtype=int)
    selected = []
    for i in range(n_select):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        if hi <= lo:
            continue
        bucket_scores = scores[lo:hi]
        best_idx = lo + int(np.argmax(bucket_scores))
        selected.append(frame_paths[best_idx])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=str,
                         help="Flat directory containing all pitch_<play_id>_frame_<n>.jpg files")
    parser.add_argument("out_dir", type=str,
                         help="Where to copy the selected frames")
    parser.add_argument("--per-video", type=int, default=8,
                         help="How many frames to select per pitch clip")
    parser.add_argument("--max-videos", type=int, default=None,
                         help="Optional cap on number of pitch clips to process")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_frames_by_play(frames_dir)
    play_ids = sorted(groups.keys())
    if args.max_videos:
        play_ids = play_ids[: args.max_videos]

    print(f"Found {len(groups)} distinct pitch clips in {frames_dir}")

    total_selected = 0
    for play_id in play_ids:
        frame_paths = groups[play_id]
        if not frame_paths:
            continue
        selected = select_from_video(frame_paths, args.per_video)
        for p in selected:
            dest = out_dir / p.name
            shutil.copy(p, dest)
            total_selected += 1
        print(f"{play_id}: selected {len(selected)}/{len(frame_paths)} frames")

    print(f"\nDone. Selected {total_selected} frames total -> {out_dir}")
    print("Next: upload this folder to Roboflow/CVAT and start annotating.")


if __name__ == "__main__":
    main()