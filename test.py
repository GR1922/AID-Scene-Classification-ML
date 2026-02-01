import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from data_classes.aid_dataset import AIDDataset
from model_classes.feature_extractor import EfficientNetB0FeatureExtractor


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    logreg = joblib.load(artifacts_dir / "logreg.joblib")
    scaler: StandardScaler = joblib.load(artifacts_dir / "scaler.joblib")
    class_names = joblib.load(artifacts_dir / "class_names.joblib")

    # Dataset split
    aid = AIDDataset(args.dataset_dir, seed=42)
    X_train, X_val, y_train, y_val = aid.train_val_split(test_size=0.2)

    # Feature extraction
    extractor = EfficientNetB0FeatureExtractor(img_size=224)
    val_ds = build_tf_dataset(X_val, y_val, extractor, batch_size=args.batch_size)

    print("Extracting validation features...")
    X_val_feat, y_val_aligned = extract_features(val_ds, extractor)

    X_val_scaled = scaler.transform(X_val_feat)

    # Predict
    y_pred = logreg.predict(X_val_scaled)
    acc = accuracy_score(y_val_aligned, y_pred)

    print("\n==== TEST RESULTS ====")
    print(f"Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_val_aligned, y_pred, target_names=class_names))

    # Confusion matrix normalized
    plt.figure(figsize=(13, 13))
    ConfusionMatrixDisplay.from_predictions(
        y_val_aligned,
        y_pred,
        display_labels=class_names,
        xticks_rotation=90,
        normalize="true"
    )
    plt.title("Confusion Matrix Normalized - Logistic Regression")
    plt.tight_layout()

    cm_path = out_dir / "confusion_matrix_logreg.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix: {cm_path.resolve()}")


if __name__ == "__main__":
    main()
