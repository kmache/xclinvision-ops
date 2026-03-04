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

logger = logging.getLogger(__name__)

CLASS_MAP = {"normal": 0, "pneumonia": 1, "tuberculosis": 2}

# ---------------------------------------------------------------------------
# 1. Albumentations Transforms (Optimized & Medical-Safe)
# ---------------------------------------------------------------------------
def get_train_transforms(image_size: int = 384) -> A.Compose:
    """Robust medical-safe augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size), 
            scale=(0.85, 1.0),
            ratio=(1.0, 1.0),
            p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        #A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-7, 7), p=0.3),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        #A.GaussNoise(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=40, max_width=40, fill_value=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
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
        self.labels = [CLASS_MAP[row['class'].lower()] for _, row in self.df.iterrows()]
        # H-2 fix: use a plain dict instead of lru_cache on a bound method.
        # lru_cache wrapping self._load_image_impl creates a strong-reference cycle
        # (cache -> bound-method -> self -> cache) that delays GC and bloats memory.
        self._image_cache: dict = {}
            
    def _load_image_impl(self, image_path: str) -> np.ndarray:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise ValueError("Image None")
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            image = np.zeros((self.fallback_size, self.fallback_size), dtype=np.uint8)
        return np.stack([image, image, image], axis=-1)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with FIFO dict cache that avoids reference cycles."""
        if self._cache_size <= 0:
            return self._load_image_impl(image_path)
        if image_path not in self._image_cache:
            if len(self._image_cache) >= self._cache_size:
                # Evict the oldest inserted entry (Python 3.7+ dict preserves insertion order)
                self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[image_path] = self._load_image_impl(image_path)
        return self._image_cache[image_path]
        
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
        image_size: int = 384, 
        cache_size: int = 1000, 
        use_weighted_sampler: bool = True,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.cache_size = cache_size
        self.use_weighted_sampler = use_weighted_sampler
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            
        df = pd.read_csv(self.manifest_path)
        if {'split', 'class', 'filepath_processed'} - set(df.columns):
            raise ValueError("Manifest missing required columns")

        # Respect PL's stage convention to avoid building unused datasets
        if stage in ("fit", None):
            self.train_dataset = ChestXrayDataset(df[df['split'] == 'train'], get_train_transforms(self.image_size), self.cache_size, fallback_size=self.image_size)
            self.val_dataset = ChestXrayDataset(df[df['split'] == 'val'], get_val_transforms(self.image_size), self.cache_size, fallback_size=self.image_size)
        if stage in ("test", None):
            self.test_dataset = ChestXrayDataset(df[df['split'] == 'test'], get_val_transforms(self.image_size), self.cache_size, fallback_size=self.image_size)

    def get_class_weights(self) -> torch.Tensor:
        counts = self.train_dataset.df['class'].value_counts().to_dict()
        total = sum(counts.values()) or 1
        class_counts = [max(counts.get(k, 0), 1) for k in ["normal", "pneumonia", "tuberculosis"]]
        weights = torch.FloatTensor([total / (3 * max(class_counts[i], 1)) for i in range(3)])
        return weights / weights.sum() * len(weights)

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        if not self.use_weighted_sampler: return None
        weights = self.get_class_weights()
        sample_weights = torch.DoubleTensor([weights[l].item() for l in self.train_dataset.labels])
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)

    def train_dataloader(self):
        sampler = self.get_sampler()
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=(sampler is None),
            sampler=sampler, num_workers=self.num_workers, pin_memory=True, drop_last=bool(sampler)
        )
         
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def get_statistics(self) -> dict:
        """Compute and return dataset statistics for reporting."""
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.setup()
            
        train_counts = self.train_dataset.df['class'].value_counts().to_dict()
        val_counts = self.val_dataset.df['class'].value_counts().to_dict()
        test_counts = self.test_dataset.df['class'].value_counts().to_dict()
        
        total_train = sum(train_counts.values())
        
        stats = {
            'train_samples': total_train,
            'val_samples': sum(val_counts.values()),
            'test_samples': sum(test_counts.values()),
            'class_distribution': {
                'normal': {
                    'train': train_counts.get('normal', 0),
                    'val': val_counts.get('normal', 0),
                    'test': test_counts.get('normal', 0)
                },
                'pneumonia': {
                    'train': train_counts.get('pneumonia', 0),
                    'val': val_counts.get('pneumonia', 0),
                    'test': test_counts.get('pneumonia', 0)
                },
                'tuberculosis': {
                    'train': train_counts.get('tuberculosis', 0),
                    'val': val_counts.get('tuberculosis', 0),
                    'test': test_counts.get('tuberculosis', 0)
                }
            },
            'class_weights': self.get_class_weights().tolist()
        }
        
        return stats

