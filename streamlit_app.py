import os
import streamlit as st
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Page config
st.set_page_config(
    page_title="Bird vs Drone Detector",
    page_icon="🕊️",
    layout="wide"
)

# Model configuration
MODEL_PATH = 'bird_drone_classifier_model.h5'
GDRIVE_MODEL_ID = '1BGKiZfFshSCIFMv8n3q0Lv6LUPVgCD5E'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_MODEL_ID}'

@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive if not present and load it"""
    if not os.path.isfile(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
    
    model = load_model(MODEL_PATH)
    return model

MODEL = download_and_load_model()

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32')
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict(image, threshold=0.5):
    """Predict if image is Bird or Drone"""
    if image is None:
        return None, None, None
    
    # Get model input size
    target = MODEL.input_shape
    if target and len(target) >= 3:
        size = (target[1] or 224, target[2] or 224)
    else:
        size = (224, 224)
    
    # Preprocess and predict
    x = preprocess_image(image, target_size=size)
    probs = MODEL.predict(x, verbose=0)
    prob = float(np.squeeze(probs))
    
    # Determine label
    if prob >= threshold:
        label = '🚁 Drone'
        confidence = prob
    else:
        label = '🕊️ Bird'
        confidence = 1 - prob
    
    return label, prob, confidence

# Header
st.title("🕊️ Bird vs Drone — Visual Classifier")
st.markdown("Upload an image or use your webcam to detect whether the subject is a **bird** or a **drone**.")

# Tabs for Upload and Webcam
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Image for Prediction")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01, key="upload_threshold")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Predict", key="predict_upload"):
                with st.spinner("Analyzing..."):
                    label, prob, confidence = predict(image, threshold)
                    
                    st.success("Prediction Complete!")
                    st.markdown(f"### {label}")
                    st.markdown(f"**Probability:** {prob:.4f}")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
    
    with col2:
        st.info("""
        **How to use:**
        1. Upload an image
        2. Adjust threshold if needed
        3. Click Predict
        
        **Threshold:**
        - Higher = More likely to detect Drone
        - Lower = More likely to detect Bird
        """)

with tab2:
    st.subheader("Live Webcam Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_image = st.camera_input("Take a picture", key="webcam")
        webcam_threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01, key="webcam_threshold")
        
        if camera_image:
            image = Image.open(camera_image)
            
            with st.spinner("Analyzing..."):
                label, prob, confidence = predict(image, webcam_threshold)
                
                if label:
                    st.markdown(f"### Last Prediction: {label}")
                    st.markdown(f"**Probability:** {prob:.4f}")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    st.progress(confidence)
    
    with col2:
        st.info("""
        **Webcam Mode:**
        - Click to capture image
        - Prediction happens automatically
        - Adjust threshold for sensitivity
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using TensorFlow and Streamlit | Model: ResNet50 + Custom Classifier")
