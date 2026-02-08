#!/usr/bin/env python3
"""
train.py

Standalone training script for Bird vs Drone dataset.

Features:
- Uses ResNet50 (imagenet weights, include_top=False, pooling='avg') to extract deep features.
- Processes images in batches for speed and memory control.
- Reads YOLO-format label files from a labels directory if present (first token is class id).
- Falls back to filename heuristic (filename contains 'drone') when no label file exists.
- Trains a RandomForest and saves the model with joblib.

Usage examples:
    python train.py --data-dir train/images --labels-dir train/labels --model-path bird_vs_drone_rf_model.pkl

"""
import os
import argparse
from tqdm import tqdm
import joblib
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_feature_extractor():
    """Load ResNet50 backbone for feature extraction."""
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model


def load_image_paths(data_dir):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = sorted(files)
    return [os.path.join(data_dir, f) for f in files]


def get_label_for_image(image_path, labels_dir=None):
    """Return 1 for drone, 0 for bird.

    Tries reading a YOLO-style label file named <basename>.txt in labels_dir.
    If not present, falls back to filename heuristic.
    """
    filename = os.path.basename(image_path)
    base, _ = os.path.splitext(filename)
    if labels_dir:
        label_file = os.path.join(labels_dir, base + ".txt")
        if os.path.isfile(label_file):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls = int(float(parts[0]))
                        return 1 if cls == 1 else 0
            except Exception:
                # ignore and fall back
                pass

    # fallback: infer from filename
    return 1 if "drone" in filename.lower() else 0


def extract_features_batch(image_paths, model, batch_size=16, target_size=(224, 224)):
    """Extract features for a list of image paths using the provided model in batches.

    Returns an array of shape (len(image_paths), feature_dim).
    """
    features = []
    n = len(image_paths)
    for start in range(0, n, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                img = image.load_img(p, target_size=target_size)
                x = image.img_to_array(img)
                batch_images.append(x)
            except Exception as e:
                # Could not read image; append a zero image to keep indexing consistent
                print(f"Warning: could not read image {p}: {e}")
                batch_images.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float32))

        batch_x = np.array(batch_images, dtype=np.float32)
        batch_x = preprocess_input(batch_x)
        preds = model.predict(batch_x, verbose=0)
        for row in preds:
            features.append(row.flatten())

    return np.array(features)


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest on ResNet features for Bird vs Drone")
    parser.add_argument("--data-dir", default=os.path.join("train", "images"), help="Directory with training images")
    parser.add_argument("--labels-dir", default=os.path.join("train", "labels"), help="Directory with YOLO .txt labels (optional)")
    parser.add_argument("--model-path", default="bird_vs_drone_rf_model.pkl", help="Output path for the trained model")
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of trees for RandomForest")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--limit", type=int, default=0, help="Optional: limit number of images (0 = no limit)")
    args = parser.parse_args()

    data_dir = args.data_dir
    labels_dir = args.labels_dir if args.labels_dir and os.path.isdir(args.labels_dir) else None

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    image_paths = load_image_paths(data_dir)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if len(image_paths) == 0:
        print(f"No image files found in {data_dir}")
        return

    print(f"Found {len(image_paths)} images. Loading ResNet50 feature extractor...")
    feature_extractor = load_feature_extractor()

    print("Extracting features in batches...")
    features = []
    # We'll also build labels in the same order
    labels = []

    # We extract features by batches to keep memory usage reasonable
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start:start + args.batch_size]
        feats = extract_features_batch(batch_paths, feature_extractor, batch_size=args.batch_size)
        features.append(feats)
        for p in batch_paths:
            labels.append(get_label_for_image(p, labels_dir))

    X = np.vstack(features)
    y = np.array(labels)

    print(f"Feature matrix shape: {X.shape}; Labels shape: {y.shape}")

    # Train/test split and training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training RandomForest (n_estimators={args.n_estimators})...")
    rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"Saving model to {args.model_path} ...")
    joblib.dump(rf, args.model_path)
    print("Done.")


if __name__ == "__main__":
    main()
