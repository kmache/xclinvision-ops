"""Training script for XClinVision models."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import yaml
import argparse

from xclinvision.models import build_model
from xclinvision.dataset import ChestXrayDataModule
from xclinvision.trainer import XClinVisionModel, MetricsCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train XClinVision model")
    parser.add_argument("--config", type=str, default="configs/train/efficientnet_b2_baseline.yaml")
    parser.add_argument("--model", type=str, default="efficientnet_b2")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--cache-size", type=int, default=1000, help="LRU cache size for images")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create DataModule
    data_module = ChestXrayDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=224,
        cache_size=args.cache_size,
    )
    data_module.setup()
    
    # Compute class weights from training data
    class_weights = data_module.get_class_weights().tolist()
    print(f"Computed class weights: {class_weights}")
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Training {args.model} model...")
    
    # Create model
    model = build_model(
        model_name=args.model,
        num_classes=3,
        pretrained=True,
    )
    
    # Create Lightning module
    pl_module = XClinVisionModel(
        model=model,
        num_classes=3,
        learning_rate=args.lr,
        loss_type="focal",
        class_weights=class_weights,
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            min_delta=0.001,
        ),
        ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            filename=f"{args.model}-{{epoch:02d}}-{{val_auc:.4f}}",
        ),
        MetricsCallback(),
    ]
    
    # Logger
    logger = MLFlowLogger(
        experiment_name="xclinvision",
        run_name=f"{args.model}_training",
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
    )
    
    # Train
    print(f"Starting training with {len(data_module.train_dataset)} train samples...")
    trainer.fit(pl_module, 
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())


if __name__ == "__main__":
    main()
