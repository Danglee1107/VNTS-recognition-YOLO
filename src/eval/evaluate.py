"""Evaluate the saved VNTS YOLO model on a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "vnts"
SOURCE_DATASET_YAML = DATASET_ROOT / "data.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "vnts-yolo26n-best.pt"


def load_dataset_config() -> Path:
    """Create an Ultralytics dataset YAML with this repository's split paths."""
    if not SOURCE_DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {SOURCE_DATASET_YAML}")

    with SOURCE_DATASET_YAML.open(encoding="utf-8") as file:
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
        raise ValueError(
            f"Expected a class-name list or mapping in {SOURCE_DATASET_YAML}"
        )

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
    resolved_yaml.write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    return resolved_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Trained YOLO checkpoint (default: {DEFAULT_WEIGHTS})",
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
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "eval")
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
            "Save the trained best.pt checkpoint to weights/vnts-yolo26n-best.pt "
            "or pass --weights with its path."
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