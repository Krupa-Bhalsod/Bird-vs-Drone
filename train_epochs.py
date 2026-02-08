#!/usr/bin/env python3
"""
train_epochs.py

Train a Keras model (ResNet50 backbone + small head) on the dataset using epoch-based training.

Features:
- Reads image paths from a data directory and labels from YOLO .txt files in a labels directory (optional).
- Builds a tf.data pipeline that decodes, resizes, and preprocesses images for ResNet50.
- Trains a binary classifier with configurable epochs, batch size, learning rate, and fine-tuning.
- Saves the final Keras model to disk.

Usage example:
    python train_epochs.py --data-dir train/images --labels-dir train/labels --epochs 10 --batch-size 32

"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def get_label_for_image(image_path, labels_dir=None):
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
                pass
    return 1 if "drone" in filename.lower() else 0


def load_image(path, label, img_size=(224, 224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = preprocess_input(image)
    return image, label


def make_dataset(paths, labels, batch_size=32, shuffle=True, img_size=(224, 224)):
    paths = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p, l: load_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size=(224, 224), lr=1e-4, fine_tune_at=None):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=preds)

    # If fine_tune_at is provided, unfreeze from that layer index
    if fine_tune_at is not None:
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        # Freeze base by default
        for layer in base.layers:
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("train", "images"), help="Image folder")
    parser.add_argument("--labels-dir", default=os.path.join("train", "labels"), help="Optional YOLO labels folder")
    parser.add_argument("--model-path", default="bird_vs_drone_resnet.h5", help="Where to save the trained Keras model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fine-tune-at", type=int, default=None, help="If set, unfreeze base model layers from this index for fine-tuning")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images for quick tests (0 = all)")
    args = parser.parse_args()

    data_dir = args.data_dir
    labels_dir = args.labels_dir if os.path.isdir(args.labels_dir) else None

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    # collect image paths
    image_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if args.limit and args.limit > 0:
        image_files = image_files[: args.limit]

    image_paths = [os.path.join(data_dir, f) for f in image_files]
    labels = [get_label_for_image(p, labels_dir) for p in image_files]

    # split
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    train_ds = make_dataset(train_paths, train_labels, batch_size=args.batch_size, shuffle=True, img_size=(args.img_size, args.img_size))
    val_ds = make_dataset(val_paths, val_labels, batch_size=args.batch_size, shuffle=False, img_size=(args.img_size, args.img_size))

    model = build_model(img_size=(args.img_size, args.img_size), lr=args.lr, fine_tune_at=args.fine_tune_at)
    model.summary()

    callbacks = [
        ModelCheckpoint(args.model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # final save (in case ModelCheckpoint didn't save)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
