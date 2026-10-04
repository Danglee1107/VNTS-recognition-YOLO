"""Shared repository paths and VNTS dataset configuration helpers."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DATASET_ROOT = RAW_DIR / "vnts"
DATASET_YAML = DATASET_ROOT / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "runs"
TRAIN_RUNS_DIR = RUNS_DIR / "train"
TEST_RUNS_DIR = RUNS_DIR / "test"
EVAL_RUNS_DIR = RUNS_DIR / "eval"
RESOLVED_DATASET_YAML = RUNS_DIR / ".vnts-dataset.yaml"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
STARTING_MODEL = PROJECT_ROOT / "yolo26n.pt"
SAVED_MODEL_WEIGHTS = WEIGHTS_DIR / "vnts-yolo26n-best.pt"


def load_dataset_config() -> Path:
    """Return an Ultralytics YAML corrected for this repository's split paths."""
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

    split_folders = ("train", "valid", "test")
    missing = [
        DATASET_ROOT / folder / "images"
        for folder in split_folders
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
    RESOLVED_DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED_DATASET_YAML.write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    return RESOLVED_DATASET_YAML