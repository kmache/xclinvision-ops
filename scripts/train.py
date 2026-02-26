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

from xclinvision.architecture import create_model
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Training {args.model} model...")
    
    # Create model
    model = create_model(
        model_name=args.model,
        num_classes=3,
        dropout_rate=0.3,
        pretrained=True,
    )
    
    # Create Lightning module
    pl_module = XClinVisionModel(
        model=model,
        num_classes=3,
        learning_rate=args.lr,
        loss_type="focal",
        class_weights=[1.0, 1.5, 1.5],
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
    
    print("Note: Data loading not yet implemented. Add DataModule integration.")
    print("Training pipeline ready for integration with processed datasets.")
    
    # Placeholder for actual training
    # trainer.fit(pl_module, datamodule=data_module)


if __name__ == "__main__":
    main()
