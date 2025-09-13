#!/usr/bin/env python3
"""
Cats vs Dogs Classification with KNN

1. Loads 100 cat and 100 dog images from "./data/cats" and "./data/dogs"
2. Extracts simple image features (color statistics + basic texture)
3. Splits into train (80%) and test (20%) sets, plus 10 unseen images for final evaluation
4. Trains a K-Nearest Neighbors classifier
5. Evaluates on the 10 unseen images, prints accuracy, and visualizes failed cases

Usage:
    python cats_dogs_knn.py
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ---- Feature extraction ----

def extract_features(img_array):
    """Extract color and texture features from an RGB image array."""
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    feats = []
    for ch in (r, g, b):
        feats.extend([
            np.mean(ch), np.std(ch),
            np.min(ch), np.max(ch),
            np.median(ch)
        ])
    # grayscale texture
    gray = 0.299*r + 0.587*g + 0.114*b
    grad_x = np.abs(np.gradient(gray, axis=1)).mean()
    grad_y = np.abs(np.gradient(gray, axis=0)).mean()
    feats.extend([gray.mean(), gray.std(), grad_x, grad_y])
    return np.array(feats, dtype=np.float32)

def load_dataset(cat_dir, dog_dir, n_per_class=100, img_size=(224, 224)):
    """Load and extract features for cats and dogs.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    paths : list[str]
    """
    X, y, paths = [], [], []
    for label, d in enumerate((cat_dir, dog_dir)):  # 0=cats, 1=dogs
        directory = Path(d)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory.resolve()}")
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
        files = []
        for ext in exts:
            files.extend(directory.glob(ext))
        files = sorted(files)[:n_per_class]

        if not files:
            print(f"Warning: no images found in {directory} matching {exts}")

        before = len(X)
        for f in files:
            try:
                img = Image.open(f).convert("RGB").resize(img_size)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(extract_features(arr))
                y.append(label)
                paths.append(str(f))
            except Exception:
                # skip unreadable/corrupt files
                continue
        loaded = len(X) - before
        if loaded == 0:
            print(f"Warning: no valid images loaded from {directory}")

    if len(X) == 0:
        raise ValueError(
            "No images were loaded for either class. Check CAT_DIR/DOG_DIR paths and image files."
        )

    return np.vstack(X), np.array(y), paths


# ---- Main script ----

def main():
    random.seed(42)
    np.random.seed(42)

    # Resolve dataset paths relative to the repository root (this file's directory)
    REPO_ROOT = Path(__file__).resolve().parent
    DATA_ROOT = REPO_ROOT / "data" / "dog-and-cat-classification-dataset" / "versions" / "1" / "PetImages"
    CAT_DIR = str(DATA_ROOT / "Cat")
    DOG_DIR = str(DATA_ROOT / "Dog")
    N = 100
    
    # Training image sizes of Resnet
    IMG_SIZE = (224, 224)
    TEST_UNSEEN = 10
    K = 5

    # Load features
    X, y, paths = load_dataset(CAT_DIR, DOG_DIR, N, IMG_SIZE)

    # Split into train+test and reserve unseen
    idx = list(range(len(y)))
    random.shuffle(idx)
    unseen_idx = idx[:TEST_UNSEEN]
    rest_idx = idx[TEST_UNSEEN:]
    X_unseen, y_unseen, paths_unseen = X[unseen_idx], y[unseen_idx], [paths[i] for i in unseen_idx]
    X_rest, y_rest = X[rest_idx], y[rest_idx]

    # Further split rest into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_rest, y_rest, test_size=0.2, random_state=42, stratify=y_rest
    )

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_unseen_s = scaler.transform(X_unseen)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_s, y_train)

    # Evaluate on test set
    y_pred = knn.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["cat","dog"]))

    # Evaluate on unseen images
    y_unseen_pred = knn.predict(X_unseen_s)
    unseen_acc = accuracy_score(y_unseen, y_unseen_pred)
    print(f"\nUnseen images accuracy: {unseen_acc*100:.2f}%")

    # Visualize failures on unseen
    failures = [(p, true, pred) for p, true, pred in zip(paths_unseen, y_unseen, y_unseen_pred) if true!=pred]
    if failures:
        n = len(failures)
        fig, axes = plt.subplots(n, 2, figsize=(6,3*n), squeeze=False)
        fig.suptitle("Failed Unseen Cases", fontsize=16)
        for i, (p, t, pr) in enumerate(failures):
            img = Image.open(p)
            axes[i,0].imshow(img); axes[i,0].axis("off")
            axes[i,0].set_title(f"True: {['cat','dog'][t]}")
            axes[i,1].imshow(img); axes[i,1].axis("off")
            axes[i,1].set_title(f"Pred: {['cat','dog'][pr]}")
        plt.tight_layout()
        plt.show()
    else:
        print("No failures on unseen images.")

if __name__ == "__main__":
    main()
