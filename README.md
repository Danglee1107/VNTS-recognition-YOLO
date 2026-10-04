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

## Run the image demo

Start the Streamlit app from the project root:

```bash
uv run streamlit run src/demo/app.py
```

Upload a JPG, PNG, WEBP, or BMP image. The app displays detected traffic-sign
labels, confidence scores, and annotated bounding boxes, or reports when no
signs are detected. The default checkpoint is
`weights/vnts-yolo26n-best.pt`; set `VNT_MODEL_PATH` to use a checkpoint stored
at a different path.
