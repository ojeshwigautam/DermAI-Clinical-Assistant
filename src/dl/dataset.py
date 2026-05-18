"""
dataset.py — HAM10000 Dataset loader with train/val/test splitting and augmentation.

Dataset: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
  - 10,015 dermoscopic images
  - 7 diagnostic classes
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.config import (
    HAM_CSV, HAM_IMAGE_DIR, HAM_IMAGE_DIR2,
    IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    TRAIN_SPLIT, VAL_SPLIT, CLASS_LABELS, AUGMENTATION,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(mode: str = "train") -> transforms.Compose:
    """
    Returns torchvision transform pipeline.

    Args:
        mode: 'train' applies augmentation; 'val'/'test' applies only normalization.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet statistics
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=AUGMENTATION["horizontal_flip"]),
            transforms.RandomVerticalFlip(p=AUGMENTATION["vertical_flip"]),
            transforms.RandomRotation(degrees=AUGMENTATION["rotation"]),
            transforms.ColorJitter(**AUGMENTATION["color_jitter"]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


# ─── Dataset Class ────────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 dermoscopic image classification.

    Args:
        df:        DataFrame subset (train/val/test).
        image_dirs: List of directories to search for images.
        transform: torchvision transforms to apply.
    """

    def __init__(self, df: pd.DataFrame, image_dirs: list, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform
        self.label2idx = {label: idx for idx, label in enumerate(CLASS_LABELS)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label_str = row["dx"]

        # Search both image directories
        img_path = None
        for d in self.image_dirs:
            candidate = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(
                f"Image {image_id}.jpg not found in {self.image_dirs}"
            )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label2idx[label_str]
        return image, torch.tensor(label, dtype=torch.long)


# ─── Data Loading Utilities ───────────────────────────────────────────────────

def load_metadata(csv_path: str = HAM_CSV) -> pd.DataFrame:
    """Load and validate HAM10000 metadata CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}.\n"
            "Please download HAM10000 from Kaggle:\n"
            "  kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection"
        )
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded metadata: {len(df)} records, {df['dx'].nunique()} classes.")
    logger.info(f"Class distribution:\n{df['dx'].value_counts().to_string()}")
    return df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle class imbalance.

    Returns:
        FloatTensor of shape (NUM_CLASSES,)
    """
    label2idx = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    counts = df["dx"].value_counts()
    total = len(df)
    weights = []
    for label in CLASS_LABELS:
        count = counts.get(label, 1)
        weights.append(total / (len(CLASS_LABELS) * count))
    weights_tensor = torch.FloatTensor(weights)
    logger.info(f"Class weights: {dict(zip(CLASS_LABELS, [f'{w:.3f}' for w in weights]))}")
    return weights_tensor


def get_dataloaders(
    csv_path: str = HAM_CSV,
    image_dirs: list = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> dict:
    """
    Build train / val / test DataLoaders.

    Returns:
        dict with keys 'train', 'val', 'test', 'class_weights'
    """
    if image_dirs is None:
        image_dirs = [HAM_IMAGE_DIR, HAM_IMAGE_DIR2]

    df = load_metadata(csv_path)

    # Deduplicate by lesion_id (keep one image per lesion)
    df_unique = df.drop_duplicates(subset="lesion_id").copy()
    logger.info(f"Unique lesions after deduplication: {len(df_unique)}")

    # Stratified train/val/test split
    train_df, temp_df = train_test_split(
        df_unique, test_size=(1 - TRAIN_SPLIT),
        stratify=df_unique["dx"], random_state=RANDOM_SEED
    )
    val_ratio = VAL_SPLIT / (VAL_SPLIT + (1 - TRAIN_SPLIT - VAL_SPLIT))
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio),
        stratify=temp_df["dx"], random_state=RANDOM_SEED
    )

    logger.info(f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    datasets = {
        "train": HAM10000Dataset(train_df, image_dirs, transform=get_transforms("train")),
        "val":   HAM10000Dataset(val_df,   image_dirs, transform=get_transforms("val")),
        "test":  HAM10000Dataset(test_df,  image_dirs, transform=get_transforms("test")),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }

    loaders["class_weights"] = compute_class_weights(train_df)
    return loaders


if __name__ == "__main__":
    loaders = get_dataloaders()
    images, labels = next(iter(loaders["train"]))
    print(f"Batch images: {images.shape}, labels: {labels.shape}")
