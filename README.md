# Introduction
This project recognize Vietnam traffic using YOLO26

---
# Packages requirment
| Library | Details |
|---|---|
| `ultralytics` | **YOLO** — training, detection, validation, inference |
| `opencv-python` | Read/process images and videos, webcam, bounding boxes |
| `numpy` | Arrays and numerical operations |
| `pandas` | Dataset/result analysis |
| `matplotlib` | Plot images, training metrics, results |
| `scikit-learn` | ML evaluation: confusion matrix, precision, recall, F1, etc. |
| `pillow` | Image manipulation/loading |
| `pyyaml` | YOLO dataset configuration (`data.yaml`) |
| `streamlit` | Browser-based image upload and model demo |
| `tqdm` | Progress bars |

Download and setup manually or copy this line:
```bash 
git clone https://github.com/Danglee1107/VNTS-recognition-YOLO.git
cd VNTS-recognition-YOLO
uv sync 
```

## Run the full pipeline

From the project root, run one command to train, save the best checkpoint,
evaluate it on the held-out test split, and start the Streamlit demo:

```bash
uv run python run.py
```

Training defaults to 100 epochs. Use `--epochs`, `--batch`, `--device`, or
`--imgsz` to change training/evaluation settings. To use the existing checkpoint
without retraining, run `uv run python run.py --skip-train`. Add `--no-demo` to
finish after evaluation, or `--skip-eval` to go directly from training to the
demo. Streamlit remains running in the terminal until stopped with Ctrl+C.
