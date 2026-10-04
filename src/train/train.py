"""Train and validate an Ultralytics YOLO model on the VNTS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "vnts"
DATASET_YAML = DATASET_ROOT / "data.yaml"


def load_dataset_config() -> Path:
    """Write and return an Ultralytics YAML with paths matching this repo layout."""
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_YAML}")

    with DATASET_YAML.open(encoding="utf-8") as file:
        source_config = yaml.safe_load(file) or {}

    raw_names = source_config.get("names")
    if isinstance(raw_names, dict):
        names: list[str] | dict[int, str] = {
            int(class_id): str(name) for class_id, name in raw_names.items()
        }
        class_count = len(names)
    elif isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
        class_count = len(names)
    else:
        raise ValueError(f"Expected a class-name list or mapping in {DATASET_YAML}")

    configured_class_count = int(source_config.get("nc", class_count))
    if configured_class_count != class_count:
        raise ValueError(
            f"Dataset config declares nc={configured_class_count}, "
            f"but provides {class_count} class names."
        )

    splits = {"train": "train", "val": "valid", "test": "test"}
    missing = [
        DATASET_ROOT / folder / "images"
        for folder in splits.values()
        if not (DATASET_ROOT / folder / "images").is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing dataset image directories: " + ", ".join(map(str, missing))
        )

    # Ultralytics expects a YAML filename, not a dict, for detection training.
    # Write the corrected split paths without changing the downloaded source YAML.
    resolved_config = {
        "path": str(DATASET_ROOT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": configured_class_count,
        "names": names,
    }
    resolved_yaml = PROJECT_ROOT / "runs" / ".vnts-dataset.yaml"
    resolved_yaml.parent.mkdir(parents=True, exist_ok=True)
    resolved_yaml.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    return resolved_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model or weights file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "train")
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
    print(f"\nTo evaluate the held-out test split, run: python src/train/test.py --weights {best_weights}")


if __name__ == "__main__":
    main()