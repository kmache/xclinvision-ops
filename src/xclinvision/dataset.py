"""Data processing, preprocessing, and dataset classes."""

from pathlib import Path
from typing import Optional, Tuple, Callable
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Standard Class Mapping
CLASS_MAP = {"normal": 0, "pneumonia": 1, "tuberculosis": 2}

# ---------------------------------------------------------------------------
# 1. Albumentations Transforms (Optimized & Medical-Safe)
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Get training augmentation pipeline."""
    return A.Compose([
        # Random zoom, but STRICTLY locked to a 1:1 aspect ratio to prevent stretching
        A.RandomResizedCrop(
            height=image_size, width=image_size, 
            scale=(0.85, 1.0), ratio=(1.0, 1.0), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        # Rotate slightly, filling empty corners with black (0)
        A.Rotate(limit=7, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        # Safely vary exposure
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Get validation/test transformation pipeline."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

# ---------------------------------------------------------------------------
# 2. PyTorch Dataset
# ---------------------------------------------------------------------------

class ChestXrayDataset(Dataset):
    """Chest X-ray dataset for multi-class classification."""
    
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None, cache_size: int = 0):
        """
        Args:
            df: DataFrame with 'filepath_processed' and 'class' columns
            transform: Albumentations transform pipeline
            cache_size: Number of images to cache in memory (0 = no caching)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_names = ["Normal", "Pneumonia", "Tuberculosis"]
        self._cache_size = cache_size
        if cache_size > 0:
            self._load_image = lru_cache(maxsize=cache_size)(self._load_image_impl)
        else:
            self._load_image = self._load_image_impl
        
    def _load_image_impl(self, image_path: str) -> np.ndarray:
        """Load and cache image from disk."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return np.stack([image, image, image], axis=-1)
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get path and label from dataframe
        image_path = self.df.loc[idx, 'filepath_processed']
        label_str = self.df.loc[idx, 'class']
        label = CLASS_MAP[label_str.lower()]
        
        # Load cached or fresh image
        image = self._load_image(image_path)
        
        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image, label

# ---------------------------------------------------------------------------
# 3. DataModule (Clean Orchestration)
# ---------------------------------------------------------------------------

class ChestXrayDataModule:
    """Data module for managing train/val/test dataloaders from the manifest."""
    
    def __init__(
        self,
        manifest_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        cache_size: int = 1000,
    ):
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.cache_size = cache_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Reads manifest and splits into train/val/test datasets."""
        df = pd.read_csv(self.manifest_path)
        
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        self.train_dataset = ChestXrayDataset(
            train_df, transform=get_train_transforms(self.image_size), cache_size=self.cache_size
        )
        self.val_dataset = ChestXrayDataset(
            val_df, transform=get_val_transforms(self.image_size), cache_size=self.cache_size
        )
        self.test_dataset = ChestXrayDataset(
            test_df, transform=get_val_transforms(self.image_size), cache_size=self.cache_size
        )

    def get_class_weights(self) -> torch.Tensor:
        """Calculates inverse frequency class weights to fix dataset imbalance."""
        if self.train_dataset is None:
            self.setup()
            
        counts = self.train_dataset.df['class'].value_counts().to_dict()
        total = sum(counts.values())
        
        # Target order: 0: normal, 1: pneumonia, 2: tuberculosis
        class_counts = [
            counts.get("normal", 0), 
            counts.get("pneumonia", 0), 
            counts.get("tuberculosis", 0)
        ]
        
        # Weight formula penalizes mistakes on the minority class more heavily
        weights = [total / (3 * count) for count in class_counts]
        return torch.FloatTensor(weights)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )