"""Train, evaluate, and launch the VNTS Streamlit demo in one command."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.path import (  # noqa: E402
    PROJECT_ROOT,
    SAVED_MODEL_WEIGHTS,
    STARTING_MODEL,
    TRAIN_RUNS_DIR,
)

TRAIN_SCRIPT = SRC_DIR / "train" / "train.py"
EVALUATE_SCRIPT = SRC_DIR / "eval" / "evaluate.py"
STREAMLIT_APP = SRC_DIR / "demo" / "app.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the VNTS detector, test it, and launch its Streamlit demo."
    )
    parser.add_argument("--model", default=str(STARTING_MODEL), help="Starting YOLO model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument(
        "--weights", type=Path, default=None, help="Checkpoint to use when skipping training"
    )
    parser.add_argument("--skip-train", action="store_true", help="Use an existing checkpoint")
    parser.add_argument("--skip-eval", action="store_true", help="Skip test evaluation")
    parser.add_argument("--no-demo", action="store_true", help="Do not start Streamlit")
    parser.add_argument("--host", default="localhost", help="Streamlit bind address")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    return parser.parse_args()


def run_stage(
    title: str, command: list[str], *, env: dict[str, str] | None = None
) -> None:
    """Run one pipeline stage and stop if it fails."""
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def resolve_checkpoint(args: argparse.Namespace, run_id: str) -> Path:
    """Train when requested, save the best checkpoint, and return its path."""
    if args.skip_train:
        checkpoint = (args.weights or SAVED_MODEL_WEIGHTS).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Train first or pass --weights."
            )
        return checkpoint

    run_name = f"vnts-yolo26n-{run_id}"
    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--project",
        str(TRAIN_RUNS_DIR),
        "--name",
        run_name,
    ]
    if args.device is not None:
        train_command.extend(["--device", args.device])
    run_stage("TRAIN", train_command)

    trained_weights = TRAIN_RUNS_DIR / run_name / "weights" / "best.pt"
    if not trained_weights.is_file():
        raise FileNotFoundError(
            f"Training completed, but best weights are missing: {trained_weights}"
        )

    SAVED_MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_weights, SAVED_MODEL_WEIGHTS)
    print(f"Saved trained checkpoint to: {SAVED_MODEL_WEIGHTS}")
    return SAVED_MODEL_WEIGHTS


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = resolve_checkpoint(args, run_id)

    if not args.skip_eval:
        eval_command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--weights",
            str(checkpoint),
            "--split",
            "test",
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--name",
            f"vnts-yolo26n-{run_id}-test",
        ]
        if args.device is not None:
            eval_command.extend(["--device", args.device])
        run_stage("TEST / EVALUATE", eval_command)

    if args.no_demo:
        print("\nPipeline complete.")
        return

    demo_env = os.environ.copy()
    demo_env["VNT_MODEL_PATH"] = str(checkpoint)
    streamlit_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    print(f"\nStreamlit will be available at http://{args.host}:{args.port}", flush=True)
    run_stage("STREAMLIT DEMO", streamlit_command, env=demo_env)


if __name__ == "__main__":
    main()