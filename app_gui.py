import os
import time
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
)


st.set_page_config(page_title="Bird vs Drone - GUI", layout="wide")

MODEL_PATH = "bird_drone_classifier_model.h5"


def apply_light_theme():
    """Inject CSS to enforce a light / white theme for Streamlit UI."""
    css = """
    <style>
    /* Global */
    html, body, .stApp {
        background: #ffffff !important;
        color: #0f172a !important;
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }

    /* Main container card look */
    .css-1d391kg, .css-1v0mbdj, .css-18e3th9 { background-color: #ffffff !important; box-shadow: 0 6px 18px rgba(15,23,42,0.06) !important; border-radius: 10px !important; }

    /* Headings */
    h1, h2, h3 { color: #0f172a !important; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#06b6d4,#0ea5e9) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 8px 14px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover { opacity: 0.95 !important; transform: translateY(-1px); }

    /* Inputs */
    input, textarea, select { background: #ffffff !important; color: #0f172a !important; }
    .stTextInput>div>div>input { border: 1px solid #e6eef8 !important; box-shadow: none !important; }

    /* Progress bar */
    .stProgress .stProgress>div>div { background: linear-gradient(90deg,#60a5fa,#06b6d4) !important; }

    /* Metrics */
    .stMetricValue { color: #065f46 !important; font-weight: 700 !important; }
    .stMetricDelta { color: #065f46 !important; }

    /* Tables / JSON */
    .element-container pre { background: #fbfdff !important; border-radius: 8px !important; }

    /* Webcam area */
    .stImage > img { border-radius: 8px !important; box-shadow: 0 6px 18px rgba(15,23,42,0.06) !important; }

    /* Small helpers */
    .stMarkdown, .stText, .stNumberInput>div>label { color: #0f172a !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_keras_model(path=MODEL_PATH):
    if not os.path.isfile(path):
        st.error(f"Model file not found: {path}")
        return None
    model = load_model(path)
    return model


def get_input_size(model):
    try:
        shape = model.input_shape
        # shape is (None, H, W, C)
        return (shape[1] or 224, shape[2] or 224)
    except Exception:
        return (224, 224)


def preprocess_pil(img: Image.Image, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_image(model, img_pil, threshold=0.5):
    target_size = get_input_size(model)
    x = preprocess_pil(img_pil, target_size)
    probs = model.predict(x, verbose=0)[0]
    # handle shape (1,) or (1,1) or (2,) depending on model
    if probs.shape == ():
        prob = float(probs)
    else:
        prob = float(np.squeeze(probs))
    label = "Drone" if prob >= threshold else "Bird"
    return label, prob


def read_label_from_yolo(image_filename, labels_dir):
    base = os.path.splitext(os.path.basename(image_filename))[0]
    lf = os.path.join(labels_dir, base + ".txt")
    if os.path.isfile(lf):
        try:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(float(parts[0]))
                    return 1 if cls == 1 else 0
        except Exception:
            pass
    # fallback
    return 1 if "drone" in image_filename.lower() else 0


def evaluate_on_folder(model, images_dir, labels_dir=None, limit=0):
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = sorted(files)
    if limit and limit > 0:
        files = files[:limit]

    y_true = []
    y_scores = []
    paths = []
    target_size = get_input_size(model)

    for f in stqdm(files, desc="Evaluating", key="eval_loop"):
        p = os.path.join(images_dir, f)
        try:
            img = Image.open(p).convert("RGB")
            x = preprocess_pil(img, target_size)
            prob = float(np.squeeze(model.predict(x, verbose=0)))
            y_scores.append(prob)
            y_true.append(read_label_from_yolo(f, labels_dir) if labels_dir else (1 if "drone" in f.lower() else 0))
            paths.append(p)
        except Exception as e:
            print(f"Skipped {p}: {e}")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    # binary predictions at 0.5
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "cm": cm,
        "report": report,
        "y_true": y_true,
        "y_scores": y_scores,
        "y_pred": y_pred,
        "paths": paths,
    }


def plot_confusion_matrix(cm, labels=("Bird", "Drone")):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def plot_roc(fpr, tpr, roc_auc):
    pass



def stqdm(iterable, desc=None, key=None):
    # simple wrapper to show progress in Streamlit
    # returns iterator
    total = len(iterable)
    pbar = st.progress(0, text=desc if desc else "")
    for i, item in enumerate(iterable, start=1):
        yield item
        pbar.progress(i / total)
    pbar.empty()


def main():
    st.title("🕊️🚁 Bird vs Drone — Demo")

    model = load_keras_model()
    if model is None:
        st.stop()

    # apply a nicer light theme/css
    try:
        apply_light_theme()
    except Exception:
        pass

    # Layout: left = controls (upload + evaluation), right = webcam
    left, right = st.columns([2, 1])

    # Shared threshold
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)

    with left:
        st.header("Upload image for prediction")
        uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_column_width=True)
            if st.button("Predict"):
                label, prob = predict_image(model, img, threshold)
                st.metric("Prediction", f"{label}", delta=f"{prob*100:.2f}%")

        st.markdown("---")
        st.header("Evaluate model on a labeled folder")
        images_dir = st.text_input("Images folder", value=os.path.join("train", "images"))
        labels_dir = st.text_input("Labels folder (optional)", value=os.path.join("train", "labels"))
        limit = st.number_input("Limit images for evaluation (0 = all)", min_value=0, value=0)
        if st.button("Run evaluation"):
            if not os.path.isdir(images_dir):
                st.error("Images folder not found")
            else:
                with st.spinner("Running evaluation — this may take a while"):
                    stats = evaluate_on_folder(model, images_dir, labels_dir if os.path.isdir(labels_dir) else None, limit=limit)

                st.subheader("Summary metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{stats['acc']*100:.2f}%")
                col2.metric("Precision", f"{stats['precision']*100:.2f}%")
                col3.metric("Recall", f"{stats['recall']*100:.2f}%")

                st.subheader("Confusion matrix")
                fig_cm = plot_confusion_matrix(stats['cm'])
                st.pyplot(fig_cm)

                st.subheader("Classification report")
                st.json(stats['report'])

    with right:
        st.header("Webcam live detection")
        if "run_cam" not in st.session_state:
            st.session_state.run_cam = False

        if st.button("Start Webcam"):
            st.session_state.run_cam = True
        if st.button("Stop Webcam"):
            st.session_state.run_cam = False

        cam_display = st.empty()

        if st.session_state.run_cam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Make sure it's connected and accessible.")
            else:
                try:
                    while st.session_state.run_cam:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # convert to PIL for model
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(frame_rgb)
                        label, prob = predict_image(model, pil, threshold=threshold)
                        # annotate
                        text = f"{label} {prob*100:.1f}%"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cam_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # small sleep to reduce CPU
                        time.sleep(0.05)
                finally:
                    cap.release()
                    st.session_state.run_cam = False


if __name__ == "__main__":
    main()
