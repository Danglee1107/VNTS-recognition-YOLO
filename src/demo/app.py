"""Streamlit application for detecting Vietnamese traffic signs in images."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "vnts-yolo26n-best.pt"
MODEL_PATH = Path(os.getenv("VNT_MODEL_PATH", str(DEFAULT_WEIGHTS))).expanduser()

st.set_page_config(
    page_title="VNTS · Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; }
    .hero {
        background: linear-gradient(120deg, #102a43 0%, #176b87 100%);
        padding: 2rem 2.2rem; border-radius: 22px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 14px 34px rgba(16, 42, 67, .15);
    }
    .hero h1 { color: white; margin: 0 0 .4rem 0; font-size: 2.25rem; }
    .hero p { color: #d6eaf2; margin: 0; font-size: 1.05rem; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e4eaf1; padding: 1rem;
        border-radius: 14px; box-shadow: 0 4px 14px rgba(16, 42, 67, .04);
    }
    </style>
    <section class="hero">
      <h1>🚦 Vietnam Traffic Sign Recognition</h1>
      <p>Upload a road image to detect traffic signs and identify their labels.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained YOLO model…")
def load_model(weights_path: str) -> YOLO:
    """Load the detector once and reuse it across Streamlit reruns."""
    return YOLO(weights_path)


def get_detections(result: object) -> list[dict[str, float | int | str]]:
    """Convert an Ultralytics result into rows for the detections table."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {})
    detections: list[dict[str, float | int | str]] = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detections.append(
            {
                "Traffic sign": str(label),
                "Confidence": float(box.conf[0].item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 800, 1024], value=640)
    st.caption("Higher confidence reduces weak detections. Image size can affect speed and accuracy.")

uploaded_image = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload a clear image containing zero or more traffic signs.",
)

if not MODEL_PATH.is_file():
    st.error(f"Model weights were not found at: {MODEL_PATH}")
    st.info("Place the trained checkpoint at weights/vnts-yolo26n-best.pt or set VNT_MODEL_PATH.")
    st.stop()

if uploaded_image is None:
    st.info("Upload an image to get started. The model will also tell you when no traffic sign is detected.")
else:
    try:
        image = Image.open(uploaded_image).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Detect traffic signs", type="primary", use_container_width=True):
        try:
            model = load_model(str(MODEL_PATH.resolve()))
            with st.spinner("Analyzing image…"):
                result = model.predict(
                    source=image,
                    conf=confidence,
                    iou=iou,
                    imgsz=image_size,
                    verbose=False,
                )[0]

            detections = get_detections(result)
            if not detections:
                st.warning("No traffic signs detected in this image.")
            else:
                annotated_bgr = result.plot()
                annotated_rgb = annotated_bgr[:, :, ::-1]
                left, right = st.columns([1.2, 1])
                with left:
                    st.subheader("Detection result")
                    st.image(annotated_rgb, caption="Detected signs and bounding boxes", use_container_width=True)
                with right:
                    st.subheader("Recognized signs")
                    st.metric("Signs detected", len(detections))
                    table = pd.DataFrame(detections)
                    table["Confidence"] = table["Confidence"].map(lambda value: f"{value:.1%}")
                    st.dataframe(table, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")