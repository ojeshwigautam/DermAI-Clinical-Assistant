"""
test_dataset.py — Unit tests for the HAM10000 dataset module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from src.dl.dataset import (
    get_transforms, HAM10000Dataset, compute_class_weights
)
from src.utils.config import CLASS_LABELS, IMAGE_SIZE


class TestTransforms:
    def test_train_transform_output_shape(self):
        transform = get_transforms("train")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_transform_output_shape(self):
        transform = get_transforms("val")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_test_transform_output_shape(self):
        transform = get_transforms("test")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_tensor_values_normalized(self):
        transform = get_transforms("val")
        img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        tensor = transform(img)
        # After normalization, values should be close to 0
        assert tensor.abs().max() < 5.0


class TestClassWeights:
    def test_weights_shape(self):
        df = pd.DataFrame({"dx": CLASS_LABELS * 10})
        weights = compute_class_weights(df)
        assert weights.shape == (len(CLASS_LABELS),)

    def test_weights_positive(self):
        df = pd.DataFrame({"dx": CLASS_LABELS * 10})
        weights = compute_class_weights(df)
        assert (weights > 0).all()

    def test_imbalanced_weights(self):
        # Minority class should have higher weight
        dx = ["nv"] * 100 + ["mel"] * 10
        df = pd.DataFrame({"dx": dx})
        weights = compute_class_weights(df)
        nv_idx  = CLASS_LABELS.index("nv")
        mel_idx = CLASS_LABELS.index("mel")
        assert weights[mel_idx] > weights[nv_idx]


class TestHAM10000Dataset:
    def _make_dummy_df(self, n=5):
        return pd.DataFrame({
            "image_id": [f"img_{i:04d}" for i in range(n)],
            "dx":       [CLASS_LABELS[i % len(CLASS_LABELS)] for i in range(n)],
        })

    def test_len(self, tmp_path):
        # Create dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(tmp_path / f"img_{i:04d}.jpg")

        df = self._make_dummy_df(5)
        transform = get_transforms("val")
        dataset = HAM10000Dataset(df, [str(tmp_path)], transform)
        assert len(dataset) == 5

    def test_getitem_shapes(self, tmp_path):
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
            img.save(tmp_path / f"img_{i:04d}.jpg")

        df = self._make_dummy_df(3)
        transform = get_transforms("val")
        dataset = HAM10000Dataset(df, [str(tmp_path)], transform)

        image, label = dataset[0]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(label, torch.Tensor)
        assert 0 <= label.item() < len(CLASS_LABELS)

    def test_missing_image_raises(self):
        df = self._make_dummy_df(1)
        transform = get_transforms("val")
        dataset = HAM10000Dataset(df, ["/nonexistent/path"], transform)
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
