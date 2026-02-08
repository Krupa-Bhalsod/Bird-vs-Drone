# main.py
# Bird vs Drone Detection using Random Forest + Streamlit
# Author: Prafulla

import os
import cv2
import numpy as np
import streamlit as st
import joblib
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# CONFIGURATION
# -----------------------------
# Use os.path.join to be OS-independent
DATA_DIR = os.path.join("train", "images")   # update to your dataset path if needed
LABELS_DIR = os.path.join("train", "labels")
MODEL_PATH = "bird_vs_drone_rf_model.pkl"

# -----------------------------
# FEATURE EXTRACTOR (ResNet50)
# -----------------------------
@st.cache_resource
def load_feature_extractor():
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model

feature_extractor = load_feature_extractor()

def extract_features(img_path):
    """Extracts deep CNN features for Random Forest input"""
    # image.load_img accepts file-like objects (Streamlit uploads) or paths
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_extractor.predict(x, verbose=0)
    return features.flatten()

# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model():
    st.info("🔄 Extracting features and training Random Forest...")

    if not os.path.isdir(DATA_DIR):
        st.error(f"Data directory not found: {DATA_DIR}")
        return

    # helper to get label from YOLO txt if present
    def get_label_for_image(image_filename):
        base, _ = os.path.splitext(image_filename)
        label_file = os.path.join(LABELS_DIR, base + ".txt")
        # If label file exists, parse first token as class id (YOLO format)
        if os.path.isfile(label_file):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        # first value is class id
                        cls = int(float(parts[0]))
                        # assume class 1 == drone, else bird
                        return 1 if cls == 1 else 0
            except Exception as e:
                # fall back to filename heuristic
                print(f"Warning: failed to read label file {label_file}: {e}")

        # fallback: infer from filename
        return 1 if "drone" in image_filename.lower() else 0

    X, y = [], []
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(image_files) == 0:
        st.error(f"No image files found in {DATA_DIR}")
        return

    # Streamlit progress
    progress_bar = st.progress(0)
    total = len(image_files)
    for idx, file in enumerate(image_files, start=1):
        path = os.path.join(DATA_DIR, file)
        try:
            label = get_label_for_image(file)
            feat = extract_features(path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            # skip problematic file but inform user
            print(f"Skipped {path}: {e}")
        if idx % 1 == 0:
            progress_bar.progress(min(idx / total, 1.0))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

    joblib.dump(rf, MODEL_PATH)
    st.info(f"📦 Model saved as {MODEL_PATH}")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(img):
    rf = joblib.load(MODEL_PATH)
    feat = extract_features(img)
    pred = rf.predict([feat])[0]
    label = "🚁 Drone" if pred == 1 else "🕊️ Bird"
    return label

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Bird vs Drone Detection", page_icon="🚁", layout="centered")

st.title("🕊️ Bird vs Drone Detection using Random Forest")
st.markdown("Detect whether an image is a **Bird** or a **Drone** using Machine Learning (ResNet + Random Forest).")

tab1, tab2 = st.tabs(["📘 Train Model", "🔍 Predict Image"])

with tab1:
    st.header("📘 Train the Model")
    if st.button("Train Now"):
        train_model()

with tab2:
    st.header("🔍 Test an Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if os.path.exists(MODEL_PATH):
            result = predict_image("temp.jpg")
            st.success(f"Prediction: {result}")
        else:
            st.error("⚠️ Model not found! Please train it first.")

st.markdown("---")
st.caption("Made with ❤️ by **Prafulla**")
