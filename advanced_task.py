#!/usr/bin/env python3
"""
Cats vs Dogs Classification with KNN features

- Loads up to N images from Cat100 and Dog100
- Extracts simple numeric features per image
- Splits into train/test and trains a KNN classifier
- Evaluates on test split and on 10 unseen images under ./unseen (cat_unseen/ and dog_unseen/)

Run:
  python advanced_task.py
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from download import download_dataset

# ----------------- Helpers -----------------

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return [p for p in sorted(dir_path.iterdir()) if p.suffix.lower() in EXTS]


def load_dataset(cat_dir: Path, dog_dir: Path, n_per_class: int, img_size=(224, 224)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the cat and dog dataset, extract features, and return as numpy arrays.

    Parameters
    ----------
    cat_dir : Path
        Path to the directory containing cat images.
    dog_dir : Path
        Path to the directory containing dog images.
    n_per_class : int
        Number of images to load per class (cat/dog).
    img_size : tuple of int
        Target size for resizing images.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Tuple containing:
        - Features array X of shape (n_samples, n_features)
        - Labels array y of shape (n_samples,)
        - List of image file paths.
    """
    X, y, paths = [], [], []
    for label, d in enumerate((cat_dir, dog_dir)):
        files = list_images(d)[: n_per_class]
        if not files:
            print(f"Warning: no images found in {d}")
        for f in files:
            try:
                img = Image.open(f).convert("RGB").resize(img_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(extract_features(arr))
                y.append(label)
                paths.append(str(f))
            except Exception as e:
                print(f"Warning: failed to load {f}: {e}")
                continue
    if len(X) == 0:
        raise ValueError("No images loaded from Cat/Dog folders. Check paths and files.")
    return np.vstack(X), np.asarray(y, dtype=np.int64), paths


def extract_features(img_array: np.ndarray) -> np.ndarray:
    """Compute simple color+texture features from an HxWx3 RGB array in [0,1].

    Parameters
    ----------
    img_array : np.ndarray
        Input image as a 3D array (height, width, channels).

    Returns
    -------
    np.ndarray
        Feature vector of shape (n_features,) for the input image.
    """
    # Split into R, G, B channels
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    feats: List[float] = []
    # Compute per-channel statistics
    for ch in (r, g, b):
        feats.extend([
            float(np.mean(ch)), float(np.std(ch)),
            float(np.min(ch)), float(np.max(ch)),
            float(np.median(ch)),
        ])
    # Compute gradients and additional features
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    grad_x = float(np.abs(np.gradient(gray, axis=1)).mean())
    grad_y = float(np.abs(np.gradient(gray, axis=0)).mean())
    feats.extend([float(gray.mean()), float(gray.std()), grad_x, grad_y])
    return np.array(feats, dtype=np.float32)


def infer_label_from_path(path: str) -> int:
    path = Path(path)
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "cat" in name or "cat" in parent:
        return 0
    if "dog" in name or "dog" in parent:
        return 1
    return -1


def load_unseen(unseen_dir: Path, k: int = 10, img_size=(224, 224)) -> Tuple[np.ndarray, List[int], List[str]]:
    """Load unseen images from ./unseen.

    Prefers two subfolders: cat_unseen/ and dog_unseen/ (lowercase). Also supports
    legacy names Cat_unseen/ and Dog_unseen/ if present. Falls back to a flat folder.

    Returns features, labels (0=cat,1=dog,-1 unknown), and paths.
    """
    cat_sub = unseen_dir / "cat_unseen"
    dog_sub = unseen_dir / "dog_unseen"

    X, y, paths = [], [], []

    if cat_sub.exists() and dog_sub.exists():
        k_per_class = max(1, k // 2)
        cat_files = list_images(cat_sub)[:k_per_class]
        dog_files = list_images(dog_sub)[:k_per_class]
        files_with_labels: List[Tuple[Path, int]] = [(p, 0) for p in cat_files] + [(p, 1) for p in dog_files]
        if not files_with_labels:
            raise ValueError(f"No images found under {cat_sub} and {dog_sub}")
        for f, lbl in files_with_labels:
            try:
                img = Image.open(f).convert("RGB").resize(img_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(extract_features(arr))
                y.append(lbl)
                paths.append(str(f))
            except Exception as e:
                print(f"Warning: failed to load unseen {f}: {e}")
                continue
        if len(X) == 0:
            raise ValueError("Failed to load any unseen images from subfolders.")
        return np.vstack(X), y, paths

    # Fallback: flat folder
    files = list_images(unseen_dir)[:k]
    if len(files) == 0:
        raise ValueError(f"No images found in unseen dir: {unseen_dir}")
    for f in files:
        try:
            img = Image.open(f).convert("RGB").resize(img_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(extract_features(arr))
            y.append(infer_label_from_path(str(f)))
            paths.append(str(f))
        except Exception as e:
            print(f"Warning: failed to load unseen {f}: {e}")
            continue
    if len(X) == 0:
        raise ValueError("Failed to load any unseen images.")
    return np.vstack(X), y, paths


def main():

    # TODO: It will take some time to download the dataset the first time from kaggle
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parent

    # Ensure dataset available locally and derive defaults
    version_dir = None
    try:
        version_dir = download_dataset(repo_root)
    except Exception as e:
        print(f"Dataset prep warning: {e}")

    if version_dir is not None:
        data_root = version_dir / "PetImages"
    else:
        data_root = repo_root / "data" / "dog-and-cat-classification-dataset" / "versions" / "1" / "PetImages"

    default_cat = data_root / "Cat100"
    default_dog = data_root / "Dog100"

    parser.add_argument("--cat-dir", type=str, default=str(default_cat))
    parser.add_argument("--dog-dir", type=str, default=str(default_dog))
    parser.add_argument("--unseen-dir", type=str, default=str(repo_root / "unseen"))
    parser.add_argument("--limit-per-class", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    cat_dir = Path(args.cat_dir)
    dog_dir = Path(args.dog_dir)
    unseen_dir = Path(args.unseen_dir)

    print(f"Loading dataset from:\n- {cat_dir}\n- {dog_dir}")
    X, y, paths = load_dataset(cat_dir, dog_dir, n_per_class=args.limit_per_class, img_size=(args.img_size, args.img_size))
    print(f"Loaded {len(y)} images. Feature dim = {X.shape[1]}")

    # Split into train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=args.neighbors, algorithm="kd_tree", n_jobs=1)
    knn.fit(X_train_s, y_train)

    # Evaluate on test set
    y_pred = knn.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {test_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["cat", "dog"]))

    # Unseen
    try:
        Xu, yu, pu = load_unseen(unseen_dir, k=10, img_size=(args.img_size, args.img_size))
    except Exception as e:
        print(f"Unseen load warning: {e}")
        return

    Xu_s = scaler.transform(Xu)
    yu_pred = knn.predict(Xu_s)

    # Compute accuracy only on images where label can be inferred
    labeled_mask = np.array([lbl in (0, 1) for lbl in yu])
    labeled_total = int(labeled_mask.sum())
    if labeled_total > 0:
        unseen_acc = accuracy_score(np.array(yu)[labeled_mask], yu_pred[labeled_mask])
        print(f"Unseen labeled accuracy: {unseen_acc*100:.2f}% ({labeled_total}/{len(yu)} images labeled)")
    else:
        print("No labeled unseen images (filenames without 'cat'/'dog'); accuracy not computed.")

    # Print predictions for all unseen images
    label_map = {0: "cat", 1: "dog", -1: "unknown"}
    print("Predictions on unseen:")
    for p, t, pr in zip(pu, yu, yu_pred):
        print(f"- {p} -> pred={label_map.get(int(pr), str(pr))} true={label_map.get(int(t), 'unknown')}")

    # Visualize ALL unseen cases (correct=green, incorrect=red, unknown=gray)
    entries = list(zip(pu, yu, yu_pred))
    n = len(entries)
    if n > 0:
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()
        fig.suptitle("Unseen Results (True vs Pred)", fontsize=16)
        for i, (p, t, pr) in enumerate(entries):
            ax = axes_flat[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                ax.axis("off"); ax.set_title(f"Failed to load: {Path(p).name}")
                continue
            ax.imshow(img); ax.axis("off")
            t_known = t in (0, 1)
            is_correct = (int(pr) == int(t)) if t_known else False
            color = "green" if is_correct else ("red" if t_known else "gray")
            t_str = label_map[int(t)] if t_known else "unknown"
            pr_str = label_map[int(pr)] if int(pr) in (0,1) else str(pr)
            ax.set_title(f"True: {t_str} | Pred: {pr_str}", color=color)
        # Hide any unused subplots
        for j in range(n, rows*cols):
            axes_flat[j].axis("off")
        plt.tight_layout(rect=(0,0,1,0.96))
        out_file_all = "unseen_results.png"
        plt.savefig(out_file_all)
        print(f"Saved {out_file_all}")
        plt.show()

    # Per-animal pie charts (cats and dogs) for labeled images
    for cls, name in [(0, "cat"), (1, "dog")]:
        cls_mask = np.array([t == cls for t in yu])
        total_cls = int(cls_mask.sum())
        if total_cls == 0:
            print(f"No {name} images in unseen set; skipping {name} pie chart.")
            continue
        pr_cls = yu_pred[cls_mask]
        t_cls = np.array(yu)[cls_mask]
        correct_cls = int(np.sum(pr_cls == t_cls))
        incorrect_cls = total_cls - correct_cls
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie([correct_cls, incorrect_cls], labels=["Correct", "Incorrect"],
               colors=["#2ca02c", "#d62728"], autopct=lambda pct: f"{pct:.1f}%")
        ax.set_title(f"Unseen {name.capitalize()} Accuracy ({correct_cls}/{total_cls})")
        out_file_pie = f"unseen_pie_{name}.png"
        plt.savefig(out_file_pie)
        print(f"Saved {out_file_pie}")
        plt.show()

    # Combined pie chart across both groups (labeled images only)
    if labeled_total > 0:
        correct_overall = int(np.sum(yu_pred[labeled_mask] == np.array(yu)[labeled_mask]))
        incorrect_overall = labeled_total - correct_overall
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie([correct_overall, incorrect_overall], labels=["Correct", "Incorrect"],
               colors=["#2ca02c", "#d62728"], autopct=lambda pct: f"{pct:.1f}%")
        ax.set_title(f"Unseen Overall Accuracy ({correct_overall}/{labeled_total})")
        out_file_overall = "unseen_pie_overall.png"
        plt.savefig(out_file_overall)
        print(f"Saved {out_file_overall}")
        plt.show()


if __name__ == "__main__":
    main()
