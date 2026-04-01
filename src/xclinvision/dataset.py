"""Data processing, preprocessing, and dataset classes.
dataset.py - Handles all data-related operations, including loading, augmentation, and batching.
"""

import logging
from collections import OrderedDict
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

from xclinvision.config import PipelineConfig
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
    aug_strength: float = 1.0,
) -> A.Compose:
    """Robust medical-safe augmentation pipeline with configurable strength."""
    transforms_list = []
    
    if horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=min(1.0, 0.5 * aug_strength)))

    transforms_list.extend([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
            interpolation=cv2.INTER_AREA,
            p=1.0,
        ),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-6, 6),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=min(1.0, 0.5 * aug_strength)
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=min(1.0, 0.5 * aug_strength)
        ),
        A.RandomGamma(
            gamma_limit=(90, 110), 
            p=min(1.0, 0.2 * aug_strength)
        ),
        A.GaussNoise(
            std_range=(0.012, 0.025), 
            p=min(1.0, 0.2 * aug_strength)
        ),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill=0,
            p=min(1.0, 0.2 * aug_strength)
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
    """Chest X-ray dataset for multi-class or multi-label classification."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        transform: Optional[Callable] = None,
        cache_size: int = 0,
        fallback_size: int = 384,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._cache_size = cache_size
        self.fallback_size = fallback_size
        self._config = config
        
        self.multilabel = config.multilabel
        self._class_names = config.class_names
        _class_map = config.class_map

        if self.multilabel:
            filtered = [n for n in self._class_names if n.lower() != 'no finding']
            if len(filtered) < len(self._class_names):
                logger.warning(
                    "'No finding' excluded from multilabel columns — "
                    "healthy state is represented by an all-zero label vector."
                )
                self._class_names = filtered

            missing = set(self._class_names) - set(self.df.columns)
            if missing:
                raise ValueError(
                    f"Multilabel mode requires one binary column per class in the "
                    f"manifest CSV. Missing columns: {sorted(missing)}"
                )
            self.labels = self.df[self._class_names].values.astype(np.float32)
        else:
            mapped = self.df['class'].str.lower().map(_class_map)
            if mapped.isna().any():
                bad_vals = self.df.loc[mapped.isna(), 'class'].unique().tolist()
                raise ValueError(
                    f"Unknown class labels found in manifest: {bad_vals}. "
                    f"Expected one of: {list(_class_map.keys())}"
                )
            self.labels = mapped.tolist()
            
        self._image_cache: OrderedDict = OrderedDict()
        self._preloaded: dict | None = None
            
    def _load_image_impl(self, image_path: str) -> np.ndarray:
        image = read_image_grayscale(image_path)
        if image is None:  
            raise IOError(f"Failed to load processed image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with True LRU dict cache."""
        if self._preloaded is not None:
            img = self._preloaded.get(image_path)
            if img is not None:
                return img.copy()
                
        if self._cache_size <= 0:
            return self._load_image_impl(image_path)
            
        # LRU Logic
        if image_path in self._image_cache:
            self._image_cache.move_to_end(image_path) # Mark as recently used
            return self._image_cache[image_path].copy()
            
        if len(self._image_cache) >= self._cache_size:
            self._image_cache.popitem(last=False) # Evict oldest
            
        img = self._load_image_impl(image_path)
        self._image_cache[image_path] = img
        return img.copy()
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.df.loc[idx, 'filepath_processed']
        image = self._load_image(image_path)
        
        if self.transform:
            image = self.transform(image=image)["image"]

        # Strictly return Tensors regardless of mode
        if self.multilabel:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            assert label.shape == (len(self._class_names),)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label

# ---------------------------------------------------------------------------
# 3. DataModule 
# ---------------------------------------------------------------------------
class ChestXrayDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        manifest_path: str,
        config: PipelineConfig,
        batch_size: int = 32, 
        num_workers: int = 4,
        image_size: int = 512, 
        cache_size: int = 1000, 
        use_weighted_sampler: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip: bool = True,
        preload: bool = False,
        aug_strength: float = 1.0,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self._cache_per_worker = max(0, cache_size // num_workers) if num_workers > 0 else cache_size
        self.use_weighted_sampler = use_weighted_sampler
        self.mean, self.std = mean, std
        self.horizontal_flip = horizontal_flip
        self.preload = preload
        self.aug_strength = aug_strength
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            
        df = pd.read_csv(self.manifest_path)
        required = {'split', 'filepath_processed'}
        required |= set(self.config.class_names) if self.config.multilabel else {'class'}
        
        if required - set(df.columns):
            raise ValueError(f"Manifest missing required columns: {required - set(df.columns)}")

        if stage in ("fit", None) and self.train_dataset is None:
            self.train_dataset = ChestXrayDataset(
                df[df['split'] == 'train'], self.config, 
                get_train_transforms(self.image_size, self.mean, self.std, self.horizontal_flip, self.aug_strength), 
                self._cache_per_worker, self.image_size
            )
            # Shuffle val split once so positive samples are spread across
            # batches — avoids all-zero batches when prevalence is low.
            val_df = df[df['split'] == 'val'].sample(frac=1, random_state=42)
            self.val_dataset = ChestXrayDataset(
                val_df, self.config,
                get_val_transforms(self.image_size, self.mean, self.std), 
                0, self.image_size
            )
            
        if stage in ("test", "predict", None):
            if self.test_dataset is None:
                test_df = df[df['split'] == 'test'].sample(frac=1, random_state=42)
                self.test_dataset = ChestXrayDataset(
                    test_df, self.config,
                    get_val_transforms(self.image_size, self.mean, self.std), 
                    0, self.image_size
                )
            self.predict_dataset = self.test_dataset

        if self.preload:
            self._preload_images()
            
        # Log dataset state on initial setup
        if stage in ("fit", None):
            stats = self.get_statistics()
            logger.info("Dataset stats: train=%d, val=%d, test=%d", 
                        stats['train_samples'], stats['val_samples'], stats['test_samples'])

    def _preload_images(self) -> None:
        datasets = [
            ds for ds in (self.train_dataset, self.val_dataset, self.test_dataset)
            if ds is not None and hasattr(ds, '_load_image_impl')
        ]
        
        if not datasets:
            logger.warning("No datasets available for preloading")
            return
            
        all_paths = {p for ds in datasets for p in ds.df["filepath_processed"]}
        logger.info("Preloading %d images into RAM …", len(all_paths))
        
        shared = {path: datasets[0]._load_image_impl(path) for path in all_paths}
        for ds in datasets:
            ds._preloaded = shared
            
        total_bytes = sum(img.nbytes for img in shared.values())
        logger.info("Preload complete: %.2f GB (%d images)", total_bytes / 1e9, len(shared))

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes weights for class imbalance correction.
        
        Returns:
            torch.Tensor:
                - Multilabel: returns `pos_weight` for BCEWithLogitsLoss (num_neg / num_pos).
                - Multiclass: returns sample probability weights for WeightedRandomSampler.
        """
        if self.config.multilabel:
            label_matrix = self.train_dataset.labels
            pos_counts = label_matrix.sum(axis=0)
            neg_counts = len(label_matrix) - pos_counts
            return torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)

        counts = self.train_dataset.df['class'].str.lower().value_counts().to_dict()
        num_classes = len(self.config.class_names)
        weights = torch.zeros(num_classes, dtype=torch.float32)
        total = sum(counts.values()) or 1
        
        for class_name, label_idx in self.config.class_map.items():
            count = max(counts.get(class_name.lower(), 0), 1)
            weights[label_idx] = total / (num_classes * count)
            
        return weights / weights.sum() * len(weights)

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        if not self.use_weighted_sampler: return None
        
        if self.config.multilabel:
            label_matrix = self.train_dataset.labels  
            pos_count = label_matrix.sum(axis=0)
            
            # Effective number of samples (Class-Balanced weighting strategy)
            beta = 0.9999
            eff_num = (1.0 - (beta ** np.maximum(pos_count, 1.0))) / (1.0 - beta)
            class_weights = 1.0 / eff_num
            
            sample_weights_np = (label_matrix * class_weights).sum(axis=1)
            
            # Fix: Assign weight to "No finding" samples (all-zero label vectors)
            no_finding_mask = label_matrix.sum(axis=1) == 0
            no_finding_count = no_finding_mask.sum()
            if no_finding_count > 0:
                eff_num_nf = (1.0 - (beta ** no_finding_count)) / (1.0 - beta)
                weight_nf = 1.0 / eff_num_nf
                sample_weights_np[no_finding_mask] = weight_nf

            sample_weights = torch.DoubleTensor(sample_weights_np)
            # Normalization to prevent NaN/Inf in the sampler
            sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
            return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            
        weights = self.get_class_weights()
        sample_weights = torch.DoubleTensor([weights[l].item() for l in self.train_dataset.labels])
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def train_dataloader(self):
        sampler = self.get_sampler()
        pw = self.num_workers > 0 and len(self.train_dataset) > self.batch_size
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=(sampler is None),
            sampler=sampler, num_workers=self.num_workers, pin_memory=True,
            drop_last=True, persistent_workers=pw,
        )
         
    def val_dataloader(self):
        pw = self.num_workers > 0 and len(self.val_dataset) > self.batch_size
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=pw)
        
    def test_dataloader(self):
        pw = self.num_workers > 0 and len(self.test_dataset) > self.batch_size
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=pw)

    def predict_dataloader(self):
        pw = self.num_workers > 0 and len(self.predict_dataset) > self.batch_size
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=pw)

    def get_statistics(self) -> dict:
        """Compute and return dataset statistics for reporting."""
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.setup()

        if self.config.multilabel:
            # 1. Dist for explicit diseases (using the filtered _class_names to avoid IndexError)
            dist = {
                name: {
                    'train': int(self.train_dataset.labels[:, i].sum()),
                    'val': int(self.val_dataset.labels[:, i].sum()),
                    'test': int(self.test_dataset.labels[:, i].sum()),
                }
                for i, name in enumerate(self.train_dataset._class_names)
            }
            
            # 2. Add 'No finding' back into the stats by counting the all-zero rows
            dist['no finding'] = {
                'train': int((self.train_dataset.labels.sum(axis=1) == 0).sum()),
                'val': int((self.val_dataset.labels.sum(axis=1) == 0).sum()),
                'test': int((self.test_dataset.labels.sum(axis=1) == 0).sum()),
            }

            return {
                'train_samples': len(self.train_dataset),
                'val_samples': len(self.val_dataset),
                'test_samples': len(self.test_dataset),
                'class_distribution': dist,
                'class_weights': self.get_class_weights().tolist(),
            }

        # --- Multiclass Logic ---
        train_counts = self.train_dataset.df['class'].str.lower().value_counts().to_dict()
        val_counts = self.val_dataset.df['class'].str.lower().value_counts().to_dict()
        test_counts = self.test_dataset.df['class'].str.lower().value_counts().to_dict()
        
        total_train = sum(train_counts.values())
        
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
                for name in self.config.class_names
            },
            'class_weights': self.get_class_weights().tolist()
        }
        
        return stats

