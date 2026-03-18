"""Data processing, preprocessing, and dataset classes.
dataset.py - Handles all data-related operations, including loading, augmentation, and batching.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from xclinvision.config import get_class_map, get_class_names
from xclinvision.processing import read_image_grayscale

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Albumentations Transforms (Optimized & Medical-Safe)
# ---------------------------------------------------------------------------
def get_train_transforms(
    image_size: int = 384,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip: bool = True,
) -> A.Compose:
    """Robust medical-safe augmentation pipeline.

    Args:
        horizontal_flip: Enable horizontal flipping. Safe for normal / pneumonia /
            cardiomegaly classification. Disable (via
            ``system.yaml augmentation.horizontal_flip: false``) for any
            laterality-sensitive disease set such as pneumothorax or pleural effusion.
    """
    transforms_list = []
    
    if horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))

    transforms_list.extend([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
            interpolation=cv2.INTER_AREA, 
            p=1
        ),

        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-6, 6),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.RandomGamma(
            gamma_limit=(90, 110),
            p=0.2
        ),

        A.GaussNoise(
            std_range=(0.012, 0.025), 
            p=0.2
        ),

        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill=0,
            p=0.2
        ),

        A.Normalize(mean=mean, std=std),

        ToTensorV2()
    ])

    return A.Compose(transforms_list)


def get_val_transforms(
    image_size: int = 384,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# ---------------------------------------------------------------------------
# 2. PyTorch Dataset
# ---------------------------------------------------------------------------
class ChestXrayDataset(Dataset):
    """Chest X-ray dataset for multi-class classification."""
    
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None, cache_size: int = 0, fallback_size: int = 384):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        self._cache_size = cache_size
        self.fallback_size = fallback_size
        class_map = get_class_map()
        mapped = self.df['class'].str.lower().map(class_map)
        invalid_mask = mapped.isna()
        if invalid_mask.any():
            bad_vals = self.df.loc[invalid_mask, 'class'].unique().tolist()
            raise ValueError(
                f"Unknown class labels found in manifest: {bad_vals}. "
                f"Expected one of: {list(class_map.keys())}"
            )
        self.labels = mapped.tolist()
        self._image_cache: dict = {}
            
    def _load_image_impl(self, image_path: str) -> np.ndarray:
        try:
            image = read_image_grayscale(image_path)
            if image is None: raise ValueError("Image None")
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            image = np.zeros((self.fallback_size, self.fallback_size), dtype=np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with FIFO dict cache that avoids reference cycles."""
        if self._cache_size <= 0:
            return self._load_image_impl(image_path)
        if image_path not in self._image_cache:
            if len(self._image_cache) >= self._cache_size:
                self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[image_path] = self._load_image_impl(image_path)
        return self._image_cache[image_path].copy()
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.df.loc[idx, 'filepath_processed']
        label = self.labels[idx]
        
        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label

# ---------------------------------------------------------------------------
# 3. DataModule (Clean Orchestration)
# ---------------------------------------------------------------------------
class ChestXrayDataModule(pl.LightningDataModule):
    """LightningDataModule for orchestrating datasets and dataloaders."""
    def __init__(
        self, 
        manifest_path: str, 
        batch_size: int = 32, 
        num_workers: int = 4,
        image_size: int = 512, 
        cache_size: int = 1000, 
        use_weighted_sampler: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip: bool = True,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.cache_size = cache_size
        self._cache_per_worker = max(0, cache_size // num_workers) if num_workers > 0 else cache_size
        self.use_weighted_sampler = use_weighted_sampler
        self.mean = mean
        self.std = std
        self.horizontal_flip = horizontal_flip
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            
        df = pd.read_csv(self.manifest_path)
        if {'split', 'class', 'filepath_processed'} - set(df.columns):
            raise ValueError("Manifest missing required columns")

        if stage in ("fit", None) and self.train_dataset is None:
            self.train_dataset = ChestXrayDataset(df[df['split'] == 'train'], get_train_transforms(self.image_size, self.mean, self.std, self.horizontal_flip), self._cache_per_worker, fallback_size=self.image_size)
            self.val_dataset = ChestXrayDataset(df[df['split'] == 'val'], get_val_transforms(self.image_size, self.mean, self.std), cache_size=0, fallback_size=self.image_size)
        if stage in ("test", "predict", None) and self.test_dataset is None:
            self.test_dataset = ChestXrayDataset(df[df['split'] == 'test'], get_val_transforms(self.image_size, self.mean, self.std), cache_size=0, fallback_size=self.image_size)
            self.predict_dataset = self.test_dataset

    def get_class_weights(self) -> torch.Tensor:
        class_map = get_class_map()
        counts = self.train_dataset.df['class'].str.lower().value_counts().to_dict()
        
        num_classes = max(class_map.values()) + 1
        weights = torch.zeros(num_classes, dtype=torch.float32)
        
        total = sum(counts.values()) or 1
        
        for class_name, label_idx in class_map.items():
            count = max(counts.get(class_name.lower(), 0), 1)
            weights[label_idx] = total / (num_classes * count)
            
        return weights / weights.sum() * len(weights)

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        if not self.use_weighted_sampler: return None
        weights = self.get_class_weights()
        sample_weights = torch.DoubleTensor([weights[l].item() for l in self.train_dataset.labels])
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_dataloader(self):
        sampler = self.get_sampler()
        pw = self.num_workers > 0
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=(sampler is None),
            sampler=sampler, num_workers=self.num_workers, pin_memory=True,
            drop_last=True, persistent_workers=pw,
        )
         
    def val_dataloader(self):
        pw = self.num_workers > 0
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=pw)
        
    def test_dataloader(self):
        pw = self.num_workers > 0
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=pw)

    def predict_dataloader(self):
        pw = self.num_workers > 0
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=pw)

    def get_statistics(self) -> dict:
        """Compute and return dataset statistics for reporting."""
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.setup()
            
        train_counts = self.train_dataset.df['class'].str.lower().value_counts().to_dict()
        val_counts = self.val_dataset.df['class'].str.lower().value_counts().to_dict()
        test_counts = self.test_dataset.df['class'].str.lower().value_counts().to_dict()
        
        total_train = sum(train_counts.values())
        
        class_names = get_class_names()
        stats = {
            'train_samples': total_train,
            'val_samples': sum(val_counts.values()),
            'test_samples': sum(test_counts.values()),
            'class_distribution': {
                name.lower(): {
                    'train': train_counts.get(name.lower(), 0),
                    'val': val_counts.get(name.lower(), 0),
                    'test': test_counts.get(name.lower(), 0),
                }
                for name in class_names
            },
            'class_weights': self.get_class_weights().tolist()
        }
        
        return stats

