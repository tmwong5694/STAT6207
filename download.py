#!/usr/bin/env python3
"""
Shared dataset download utility for STAT6207.

Provides:
- download_dataset_to_repo_data(repo_root: Path) -> Path

This downloads the Kaggle dataset using kagglehub and mirrors the version
folder into <repo_root>/data/dog-and-cat-classification-dataset/versions/<N>
so the project can use a stable local path regardless of the cache location.
"""
from pathlib import Path
import shutil
import kagglehub

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _ensure_cat_dog_100(version_root: Path) -> None:
    """Create PetImages/Cat100 and Dog100 with images 0.jpg..99.jpg.

    If any of 0..99 is missing in the source, fall back to the first 100 images
    (numeric-sort by stem when possible), and log a warning.
    """
    pet = version_root / "PetImages"
    cat_src = pet / "Cat"
    dog_src = pet / "Dog"
    cat_100 = pet / "Cat100"
    dog_100 = pet / "Dog100"

    def select_first_100(src: Path) -> list[Path]:
        files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        # Prefer numeric sort by filename stem if possible
        def key(p: Path):
            try:
                return (0, int(p.stem))
            except Exception:
                return (1, p.name)
        files.sort(key=key)
        return files[:100]

    def collect_exact_range(src: Path) -> list[Path]:
        picks = []
        missing = []
        # try .jpg exact first, then any known extension
        for i in range(100):
            candidate = src / f"{i}.jpg"
            if candidate.exists():
                picks.append(candidate)
            else:
                # search for any ext match (e.g., 5.jpeg)
                alt = None
                for ext in IMG_EXTS:
                    c = src / f"{i}{ext}"
                    if c.exists():
                        alt = c
                        break
                if alt is not None:
                    picks.append(alt)
                else:
                    missing.append(i)
        if missing:
            print(f"Warning: Missing files {missing} in {src.name}; using first-100 fallback.")
            return select_first_100(src)
        return picks

    def repopulate(src: Path, dst: Path):
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        picks = collect_exact_range(src)
        for p in picks:
            shutil.copy2(p, dst / p.name)
        print(f"Prepared {dst} with {len(picks)} images from {src}")

    if not cat_src.exists() or not dog_src.exists():
        print(f"Warning: Expected Cat/Dog folders under {pet}, skipping Cat100/Dog100 prep.")
        return

    repopulate(cat_src, cat_100)
    repopulate(dog_src, dog_100)


def download_dataset(repo_root: Path) -> Path:
    """Download the Kaggle dataset and copy it into repo ./data structure.

    Returns the path to the version folder under ./data/dog-and-cat-classification-dataset/versions/<N>.
    Also ensures PetImages/Cat100 and PetImages/Dog100 contain 0.jpg..99.jpg (with fallback).
    """
    src_path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
    src = Path(src_path).resolve()
    dest = repo_root / "data" / "dog-and-cat-classification-dataset" / "versions" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    # Prepare Cat100 and Dog100 subsets
    try:
        _ensure_cat_dog_100(dest)
    except Exception as e:
        print(f"Warning preparing Cat100/Dog100: {e}")
    print(f"Dataset available at: {dest}")
    return dest
