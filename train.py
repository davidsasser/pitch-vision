import argparse
from pathlib import Path

from ultralytics import YOLO


def train(data_yaml: str, epochs: int, imgsz: int, model_size: str, run_name: str):
    # Start from a pretrained checkpoint (transfer learning) rather than
    # training from scratch -- this matters a lot with a small dataset.
    # "n" (nano) is fastest to iterate with; move up to "s" or "m" later
    # if you want more accuracy and have the GPU time to spare.
    model = YOLO(f"yolo11{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,           # lower this (e.g. 8) if you run out of GPU memory
        patience=20,        # stop early if validation loss stalls
        name=run_name,
        project="runs/ball_detector",
    )

    print(f"\nTraining complete. Best weights saved to: {results.save_dir}/weights/best.pt")
    return results


def validate(weights_path: str, data_yaml: str):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print("\nValidation metrics:")
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=str,
                         help="Path to data.yaml from the Roboflow export")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960,
                         help="Training image size -- keep high since the ball is small")
    parser.add_argument("--model-size", type=str, default="n",
                         choices=["n", "s", "m", "l", "x"],
                         help="YOLO11 model size: n=fastest, x=most accurate/slowest")
    parser.add_argument("--name", type=str, default="ball_v1",
                         help="Name for this training run (used for output folder)")
    args = parser.parse_args()

    if not Path(args.data_yaml).exists():
        raise FileNotFoundError(
            f"Couldn't find {args.data_yaml} -- check the path to your Roboflow export"
        )

    results = train(args.data_yaml, args.epochs, args.imgsz, args.model_size, args.name)
    best_weights = f"{results.save_dir}/weights/best.pt"
    validate(best_weights, args.data_yaml)


if __name__ == "__main__":
    main()