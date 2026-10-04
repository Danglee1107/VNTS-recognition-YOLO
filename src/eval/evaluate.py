"""Evaluate the saved VNTS YOLO model on a dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.path import (  # noqa: E402
    EVAL_RUNS_DIR,
    SAVED_MODEL_WEIGHTS,
    load_dataset_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=SAVED_MODEL_WEIGHTS,
        help=f"Trained YOLO checkpoint (default: {SAVED_MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to evaluate; use test for final held-out results.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--project", type=Path, default=EVAL_RUNS_DIR)
    parser.add_argument("--name", default=None, help="Evaluation output directory name")
    parser.add_argument(
        "--exist-ok", action="store_true", help="Allow reusing the run directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Pass --weights with a valid trained checkpoint."
        )

    run_name = args.name or f"vnts-yolo26n-{args.split}-evaluation"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(load_dataset_config()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=run_name,
        exist_ok=args.exist_ok,
        plots=True,
    )

    print(
        f"{args.split.upper()} metrics: "
        f"precision={metrics.box.mp:.4f}, recall={metrics.box.mr:.4f}, "
        f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}"
    )
    print(f"Evaluation artifacts saved to: {args.project / run_name}")


if __name__ == "__main__":
    main()