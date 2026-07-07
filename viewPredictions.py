import argparse
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Read a YOLO-format label file and convert normalized cx,cy,w,h
    into pixel-space x1,y1,x2,y2 boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        _, cx, cy, w, h = (float(x) for x in parts[:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_iou_for_gt(gt_boxes, pred_boxes) -> float:
    """Return worst-case (minimum) best-match IoU across all GT boxes
    in this image -- used to rank images from worst to best."""
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0  # no ball, no prediction = fine
    worst = 1.0
    for gt in gt_boxes:
        best_match = max((iou(gt, pred) for pred in pred_boxes), default=0.0)
        worst = min(worst, best_match)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=str)
    parser.add_argument("weights", type=str, help="Path to best.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--out-dir", type=str, default="prediction_review")
    parser.add_argument("--max-images", type=int, default=None,
                         help="Optional cap, useful if your val set is large")
    args = parser.parse_args()

    data_yaml = yaml.safe_load(Path(args.data_yaml).read_text())
    dataset_root = Path(args.data_yaml).parent

    split_key = {"train": "train", "val": "val", "test": "test"}[args.split]
    images_dir = (dataset_root / data_yaml[split_key]).resolve()
    # Roboflow's YOLO export usually mirrors images/ and labels/ side by side
    labels_dir = Path(str(images_dir).replace("images", "labels"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    image_paths = sorted(images_dir.glob("*.jpg"))
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    results_summary = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        gt_boxes = load_yolo_labels(labels_dir / f"{img_path.stem}.txt", w, h)

        pred = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        pred_boxes = [tuple(b) for b in pred.boxes.xyxy.cpu().numpy().tolist()] if pred.boxes is not None else []

        score = best_iou_for_gt(gt_boxes, pred_boxes)

        for (x1, y1, x2, y2) in gt_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for (x1, y1, x2, y2) in pred_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.putText(img, f"worst-IoU: {score:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        results_summary.append((score, img_path.name))

    # Rename files with a rank prefix so worst cases sort first in a file browser
    results_summary.sort(key=lambda x: x[0])
    for rank, (score, name) in enumerate(results_summary):
        src = out_dir / name
        dst = out_dir / f"{rank:04d}_iou{score:.2f}_{name}"
        src.rename(dst)

    print(f"Saved {len(results_summary)} annotated images to {out_dir}")
    print("Sorted worst-first by filename -- start with 0000_... to see the biggest problem cases.")
    print("GREEN = ground truth, RED = prediction")


if __name__ == "__main__":
    main()