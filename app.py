import os
import io
import base64
import gdown
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'bird_drone_classifier_model.h5')
STATIC_TMP = os.path.join(BASE_DIR, 'static', 'tmp')
os.makedirs(STATIC_TMP, exist_ok=True)

# Google Drive model URL
GDRIVE_MODEL_ID = '1BGKiZfFshSCIFMv8n3q0Lv6LUPVgCD5E'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_MODEL_ID}'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_model():
    """Download model from Google Drive if not present"""
    if not os.path.isfile(MODEL_PATH):
        print(f"Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print(f"Model downloaded to {MODEL_PATH}")


def load_keras():
    download_model()
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model


MODEL = load_keras()


def preprocess_pil(img: Image.Image, target_size=(224, 224)):
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32')
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_pil(img: Image.Image):
    target = MODEL.input_shape
    if target and len(target) >= 3:
        size = (target[1] or 224, target[2] or 224)
    else:
        size = (224, 224)
    x = preprocess_pil(img, target_size=size)
    probs = MODEL.predict(x, verbose=0)
    prob = float(np.squeeze(probs))
    label = 'Drone' if prob >= 0.5 else 'Bird'
    return label, prob


def read_label_from_yolo(image_filename, labels_dir):
    base = os.path.splitext(os.path.basename(image_filename))[0]
    lf = os.path.join(labels_dir, base + '.txt')
    if os.path.isfile(lf):
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(float(parts[0]))
                    return 1 if cls == 1 else 0
        except Exception:
            pass
    return 1 if 'drone' in image_filename.lower() else 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # handle file upload
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    # threshold can be passed as form field (string) or default 0.5
    threshold = request.form.get('threshold')
    try:
        threshold = float(threshold) if threshold is not None else 0.5
    except Exception:
        threshold = 0.5

    if file and allowed_file(file.filename):
        img = Image.open(file.stream)
        # predict probability
        target = MODEL.input_shape
        if target and len(target) >= 3:
            size = (target[1] or 224, target[2] or 224)
        else:
            size = (224, 224)
        x = preprocess_pil(img, target_size=size)
        probs = MODEL.predict(x, verbose=0)
        prob = float(np.squeeze(probs))
        label = 'Drone' if prob >= threshold else 'Bird'
        return jsonify({'label': label, 'prob': prob})
    return jsonify({'error': 'invalid file type'}), 400


@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    data = request.json
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({'error': 'no image'}), 400
    header, encoded = img_b64.split(',', 1)
    binary = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(binary))
    # threshold may be provided in JSON
    threshold = data.get('threshold', 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    target = MODEL.input_shape
    if target and len(target) >= 3:
        size = (target[1] or 224, target[2] or 224)
    else:
        size = (224, 224)
    x = preprocess_pil(img, target_size=size)
    probs = MODEL.predict(x, verbose=0)
    prob = float(np.squeeze(probs))
    label = 'Drone' if prob >= threshold else 'Bird'
    return jsonify({'label': label, 'prob': prob})


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if request.method == 'GET':
        return render_template('evaluate.html', results=None)

    images_dir = request.form.get('images_dir') or os.path.join(BASE_DIR, 'train', 'images')
    labels_dir = request.form.get('labels_dir') or os.path.join(BASE_DIR, 'train', 'labels')
    limit = int(request.form.get('limit') or 0)

    if not os.path.isdir(images_dir):
        return render_template('evaluate.html', results={'error': 'images folder not found'})

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files = sorted(files)
    if limit > 0:
        files = files[:limit]

    y_true = []
    y_pred = []
    y_score = []

    for f in files:
        p = os.path.join(images_dir, f)
        try:
            img = Image.open(p).convert('RGB')
            label, prob = predict_pil(img)
            y_score.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
            y_true.append(read_label_from_yolo(f, labels_dir) if os.path.isdir(labels_dir) else (1 if 'drone' in f.lower() else 0))
        except Exception as e:
            print('skip', p, e)

    import numpy as _np
    y_true = _np.array(y_true)
    y_pred = _np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
    report = classification_report(y_true, y_pred, output_dict=True) if len(y_true) else {}
    cm = confusion_matrix(y_true, y_pred) if len(y_true) else _np.zeros((2, 2), dtype=int)

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bird', 'Drone'], yticklabels=['Bird', 'Drone'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_path = os.path.join(STATIC_TMP, f'cm_{timestamp}.png')
    fig.savefig(cm_path, bbox_inches='tight')
    plt.close(fig)

    results = {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'report': report,
        'confusion_matrix_img': os.path.relpath(cm_path, BASE_DIR).replace('\\', '/'),
        'n': len(y_true),
    }
    return render_template('evaluate.html', results=results)


@app.route('/static/tmp/<path:filename>')
def tmp_static(filename):
    return send_from_directory(STATIC_TMP, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
