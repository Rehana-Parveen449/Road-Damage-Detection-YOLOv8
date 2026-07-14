import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="Road Defect Detection System",
    page_icon="🛣️",
    layout="wide"
)

# ── Header ────────────────────────────────────────────
st.title("🛣️ Road Defect Detection System")
st.markdown("Detect **Potholes**, **Cracks**, and **Road Markings** using AI.")
st.divider()

# ── Sidebar ───────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "🔍 Select Detection Model",
    [
        "🕳️ Pothole Detection",
        "🔱 Crack Detection",
        "⚠️ Road Markings Detection"
    ]
)

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app uses YOLOv8 segmentation "
    "models to detect road defects. "
    "Upload an image to get started."
)

# ── Load Model ────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    if "Pothole" in model_name:
        return YOLO('models/pothole.pt')
    elif "Crack" in model_name:
        return YOLO('models/crack.pt')
    else:
        return YOLO('models/marking.pt')

model = load_model(model_choice)

# ── Model Info Cards ──────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    if "Pothole" in model_choice:
        st.success("🕳️ Pothole Detection — Active")
    else:
        st.info("🕳️ Pothole Detection")

with col2:
    if "Crack" in model_choice:
        st.success("🔱 Crack Detection — Active")
    else:
        st.info("🔱 Crack Detection")

with col3:
    if "Marking" in model_choice:
        st.success("⚠️ Road Markings Detection — Active")
    else:
        st.info("⚠️ Road Markings Detection")

st.divider()

# ── Image Upload ──────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload a Road Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear road image for best results"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # show original and result side by side
    col_orig, col_result = st.columns(2)

    with col_orig:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    # run detection
    with st.spinner(f"Running {model_choice}..."):
        start = time.time()
        results = model.predict(
            source=img_array,
            conf=confidence,
            task='segment'
        )
        end = time.time()
        inference_time = round(end - start, 2)

    # result image
    result_img = results[0].plot()
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    with col_result:
        st.subheader("🔍 Detection Result")
        st.image(result_img_rgb, use_column_width=True)

    st.divider()

    # ── Metrics ───────────────────────────────────────
    st.subheader("📊 Detection Summary")

    num_detections = len(results[0].boxes)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        label = (
            "Potholes" if "Pothole" in model_choice
            else "Cracks" if "Crack" in model_choice
            else "Markings"
        )
        st.metric(f"Total {label} Found", num_detections)

    with m2:
        st.metric("Inference Time", f"{inference_time}s")

    with m3:
        if num_detections > 0:
            avg_conf = np.mean([
                float(b.conf[0])
                for b in results[0].boxes
            ])
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        else:
            st.metric("Avg Confidence", "N/A")

    with m4:
        if num_detections > 0:
            max_conf = max([
                float(b.conf[0])
                for b in results[0].boxes
            ])
            st.metric("Highest Confidence", f"{max_conf:.2%}")
        else:
            st.metric("Highest Confidence", "N/A")

    st.divider()

    # ── Per Detection Details ─────────────────────────
    if num_detections > 0:
        st.subheader(f"🔎 Individual {label} Details")

        for i, box in enumerate(results[0].boxes):
            conf_score = float(box.conf[0])
            with st.expander(f"{label[:-1]} {i+1} — Confidence: {conf_score:.2%}"):
                coords = box.xyxy[0].tolist()
                st.write(f"**Confidence:** {conf_score:.2%}")
                st.write(f"**Location (x1,y1,x2,y2):** {[round(c,1) for c in coords]}")
                st.progress(conf_score)

    else:
        st.warning(
            f"No {label.lower()} detected. "
            "Try lowering the confidence threshold in the sidebar."
        )

else:
    # placeholder when no image uploaded
    st.markdown(
        """
        ### 👆 Upload an image to get started

        **This app can detect:**
        - 🕳️ **Potholes** — detects and segments pothole regions
        - 🔱 **Cracks** — identifies crack patterns on road surface  
        - ⚠️ **Road Markings** — detects faded or damaged road markings

        **How to use:**
        1. Select a model from the sidebar
        2. Adjust confidence threshold if needed
        3. Upload a road image
        4. View results instantly!
        """
    )