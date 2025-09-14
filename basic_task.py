#!/usr/bin/env python3
"""
Cat Image Similarity Analysis using PyTorch and ResNet
Processes exactly the first 100 images from Cat100.

1. Loads the first 100 cat images from Cat100
2. Uses a pre-trained ResNet-50 model to extract features
3. Computes cosine similarity between all images
4. Identifies and visualizes the top 5 most similar and dissimilar pairs
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import kagglehub

# ---------------- Dataset prep helpers ----------------

def download_dataset_to_repo_data(repo_root: Path) -> Path:
    """Download the Kaggle dataset and copy it into repo ./data structure.

    Returns the path to the version folder under ./data/dog-and-cat-classification-dataset/versions/<N>.
    """
    src_path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
    src = Path(src_path).resolve()
    dest = repo_root / "data" / "dog-and-cat-classification-dataset" / "versions" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Copy entire version directory
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"Dataset available at: {dest}")
    return dest


def create_catdog100_subset(version_dir: Path) -> None:
    """Create Cat100 and Dog100 subsets with files 0.jpg..99.jpg if present."""
    pet = version_dir / "PetImages"
    cat_src = pet / "Cat"
    dog_src = pet / "Dog"
    cat_dst = pet / "Cat100"
    dog_dst = pet / "Dog100"
    cat_dst.mkdir(parents=True, exist_ok=True)
    dog_dst.mkdir(parents=True, exist_ok=True)

    def copy_range(src: Path, dst: Path):
        copied = 0
        for i in range(100):
            f = src / f"{i}.jpg"
            if f.exists():
                try:
                    shutil.copy2(f, dst / f.name)
                    copied += 1
                except Exception as e:
                    print(f"Warning: failed to copy {f}: {e}")
        print(f"Prepared {copied} images in {dst}")

    copy_range(cat_src, cat_dst)
    copy_range(dog_src, dog_dst)

# ---------------- Existing analysis pipeline ----------------

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def load_feature_extractor(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval().to(device)
    print("Loaded ResNet-50 feature extractor")
    return extractor


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_images(data_dir, max_images=100):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = sorted([f for f in data_path.iterdir() if f.suffix.lower() in exts])[:max_images]
    if not files:
        raise ValueError(f"No images found in '{data_dir}'")
    print(f"Processing first {len(files)} images from {data_dir}")
    transform = get_transform()
    tensors, paths = [], []
    for i, f in enumerate(files):
        try:
            img = Image.open(f).convert("RGB")
            tensors.append(transform(img))
            paths.append(str(f))
            if (i+1) % 20 == 0:
                print(f"Loaded {i+1}/{len(files)} images")
        except Exception as e:
            print(f"Warning: failed to load {f.name}: {e}")
    if len(tensors) < 2:
        raise ValueError("Fewer than 2 valid images loaded; cannot compute pairwise similarity.")
    batch = torch.stack(tensors)
    return batch, paths


def extract_features(model, batch, device, bs=16):
    n = batch.size(0)
    print(f"Extracting features for {n} images...")
    feats = []
    with torch.no_grad():
        for i in range(0, n, bs):
            b = batch[i:i+bs].to(device)
            out = model(b).view(b.size(0), -1).cpu().numpy()
            feats.append(out)
            print(f"Processed {min(i+bs, n)}/{n}")
    feats = np.vstack(feats)
    print(f"Features shape: {feats.shape}")
    return feats


def compute_sim_matrix(feats):
    print("Computing cosine similarity matrix...")
    sim = cosine_similarity(feats)
    print(f"Matrix shape: {sim.shape}")
    return sim


def find_top_pairs(sim, paths, k=5):
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    vals = sim[iu]
    idxs = np.argsort(vals)
    sim_pairs = []
    for idx in idxs[-k:][::-1]:
        i, j = iu[0][idx], iu[1][idx]
        sim_pairs.append((i, j, vals[idx], paths[i], paths[j]))
    dis_pairs = []
    for idx in idxs[:k]:
        i, j = iu[0][idx], iu[1][idx]
        dis_pairs.append((i, j, vals[idx], paths[i], paths[j]))
    return sim_pairs, dis_pairs


def visualize(pairs, title, out_file):
    n = len(pairs)
    fig, ax = plt.subplots(n, 2, figsize=(12, 4*n))
    fig.suptitle(title, fontsize=16)
    for i, (a, b, score, pa, pb) in enumerate(pairs):
        img1 = Image.open(pa)
        img2 = Image.open(pb)
        ax[i,0].imshow(img1); ax[i,0].axis("off")
        ax[i,0].set_title(f"Idx {a+1}: {Path(pa).name}")
        ax[i,1].imshow(img2); ax[i,1].axis("off")
        ax[i,1].set_title(f"Idx {b+1}: {Path(pb).name}")
        fig.text(0.5, 0.95 - i*0.9/n, f"Score: {score:.4f}", ha="center")
    plt.tight_layout(rect=(0,0,1,0.96))
    plt.savefig(out_file)
    print(f"Saved {out_file}")
    plt.show()


def print_summary(sim_pairs, dis_pairs, total, paths):
    print("\n" + "="*50)
    print(f"Results for first {total} images")
    print("="*50)
    print("\nTop similar pairs:")
    for i,(a,b,s,pa,pb) in enumerate(sim_pairs,1):
        print(f"{i}. Image#{a+1} ({Path(pa).name}) vs Image#{b+1} ({Path(pb).name}), Score={s:.4f}")
    print("\nTop dissimilar pairs:")
    for i,(a,b,s,pa,pb) in enumerate(dis_pairs,1):
        print(f"{i}. Image#{a+1} ({Path(pa).name}) vs Image#{b+1} ({Path(pb).name}), Score={s:.4f}")
    if total >= 37:
        print(f"\nImage#37 path: {paths[36]}")


def main():
    repo_root = Path(__file__).resolve().parent
    # Ensure dataset present under ./data and subsets ready
    version_dir = None
    try:
        version_dir = download_dataset_to_repo_data(repo_root)
        create_catdog100_subset(version_dir)
    except Exception as e:
        print(f"Dataset prep warning: {e}")

    # Prefer the dynamically detected version_dir if available
    if version_dir is not None:
        data_dir = version_dir / "PetImages" / "Cat100"
        DATA_DIR = str(data_dir)
    else:
        DATA_DIR = "./data/dog-and-cat-classification-dataset/versions/1/PetImages/Cat100"

    MAX = 100; TOP_K = 5
    dev = setup_device()
    model = load_feature_extractor(dev)
    batch, paths = load_images(DATA_DIR, MAX)
    feats = extract_features(model, batch, dev)
    sim = compute_sim_matrix(feats)
    sim_p, dis_p = find_top_pairs(sim, paths, TOP_K)
    print_summary(sim_p, dis_p, len(paths), paths)
    visualize(sim_p, "Top Similar (Cat100)", "most_similar_100.png")
    visualize(dis_p, "Top Dissimilar (Cat100)", "most_dissimilar_100.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
