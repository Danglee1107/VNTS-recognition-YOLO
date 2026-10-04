"""Evaluate trained YOLO weights on the held-out VNTS test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.path import (  # noqa: E402
    SAVED_MODEL_WEIGHTS,
    TEST_RUNS_DIR,
    TRAIN_RUNS_DIR,
    load_dataset_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Trained best.pt file (defaults to the newest VNTS training run)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--project", type=Path, default=TEST_RUNS_DIR)
    parser.add_argument("--name", default="vnts-yolo26n-test")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the run directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights
    if weights is None:
        candidates = list(TRAIN_RUNS_DIR.glob("vnts-yolo26n*/weights/best.pt"))
        weights = (
            max(candidates, key=lambda path: path.stat().st_mtime)
            if candidates
            else SAVED_MODEL_WEIGHTS
        )

    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {weights}\n"
            "Run src/train/train.py first or pass --weights with a valid best.pt path."
        )

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(load_dataset_config()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        plots=True,
    )
    print(
        "Test metrics: "
        f"precision={metrics.box.mp:.4f}, recall={metrics.box.mr:.4f}, "
        f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}"
    )
    print(f"Test results saved to: {args.project / args.name}")


if __name__ == "__main__":
    main()