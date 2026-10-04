"""Train and validate an Ultralytics YOLO model on the VNTS dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.path import (  # noqa: E402
    STARTING_MODEL,
    TRAIN_RUNS_DIR,
    load_dataset_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=str(STARTING_MODEL), help="YOLO model or weights file"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--project", type=Path, default=TRAIN_RUNS_DIR)
    parser.add_argument("--name", default="vnts-yolo26n")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the run directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset_config()

    model = YOLO(args.model)
    model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        patience=args.patience,
        plots=True,
    )

    trainer = getattr(model, "trainer", None)
    run_dir = Path(getattr(trainer, "save_dir", args.project / args.name))
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished without best weights at {best_weights}")

    print(f"\nBest weights: {best_weights}")
    print("Running final validation on the separate valid split...")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(dataset),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run_dir.parent),
        name=f"{run_dir.name}-validation",
        plots=True,
    )
    print(
        "Validation metrics: "
        f"precision={metrics.box.mp:.4f}, recall={metrics.box.mr:.4f}, "
        f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}"
    )
    print(
        "\nTo evaluate the held-out test split, run: "
        f"python src/train/test.py --weights {best_weights}"
    )


if __name__ == "__main__":
    main()