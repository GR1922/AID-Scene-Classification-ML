import argparse
import os
import random
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from data_classes.aid_dataset import AIDDataset
from model_classes.feature_extractor import EfficientNetB0FeatureExtractor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess(path, label, extractor: EfficientNetB0FeatureExtractor):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = extractor.preprocess(img)
    return img, label


def build_tf_dataset(X, y, extractor, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda p, l: load_and_preprocess(p, l, extractor), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def extract_features(ds, extractor):
    feats = []
    labels = []
    for batch_imgs, batch_labels in ds:
        f = extractor.extract(batch_imgs).numpy()
        feats.append(f)
        labels.append(batch_labels.numpy())
    return np.vstack(feats), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to AID dataset directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dataset
    aid = AIDDataset(args.dataset_dir, seed=args.seed)
    X_train, X_val, y_train, y_val = aid.train_val_split(test_size=0.2)

    # 2) Feature extractor
    extractor = EfficientNetB0FeatureExtractor(img_size=224)

    # 3) tf.data pipeline (NO SHUFFLE!)
    train_ds = build_tf_dataset(X_train, y_train, extractor, batch_size=args.batch_size)
    val_ds = build_tf_dataset(X_val, y_val, extractor, batch_size=args.batch_size)

    # 4) Feature extraction
    print("Extracting train features...")
    X_train_feat, y_train_aligned = extract_features(train_ds, extractor)

    print("Extracting val features...")
    X_val_feat, y_val_aligned = extract_features(val_ds, extractor)

    # 5) Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    # 6) Train classifier (Logistic Regression)
    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=4000, multi_class="ovr", n_jobs=-1)
    logreg.fit(X_train_scaled, y_train_aligned)

    # 7) Evaluate quick
    y_pred = logreg.predict(X_val_scaled)
    acc = accuracy_score(y_val_aligned, y_pred)

    print("\n==== TRAINING DONE ====")
    print(f"Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_val_aligned, y_pred, target_names=aid.class_names))

    # 8) Save artifacts
    joblib.dump(logreg, artifacts_dir / "logreg.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    joblib.dump(aid.class_names, artifacts_dir / "class_names.joblib")

    # Salviamo anche split e features (utile per riproducibilità/debug)
    np.savez_compressed(
        artifacts_dir / "features_split.npz",
        X_train_feat=X_train_feat,
        y_train=y_train_aligned,
        X_val_feat=X_val_feat,
        y_val=y_val_aligned
    )

    print(f"\nSaved artifacts in: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()

