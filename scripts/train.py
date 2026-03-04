#!/usr/bin/env python
"""Main training script for XClinVision chest X-ray classification."""

import argparse
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path

import yaml

# Add src to python path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress noisy UserWarnings from torch/PL internals only
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch|lightning")
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from xclinvision.dataset import ChestXrayDataModule
from xclinvision.modeling import build_model
from xclinvision.trainer import (
    XClinVisionModel,
    MetricsCallback,
    XAIValidationCallback,
)


SYSTEM_CONFIG = Path(__file__).parent.parent / "configs" / "system.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(model_config_path: str) -> dict:
    """Load system.yaml as base, then deep-merge the model config on top.

    Returns the merged dict.  Missing files emit a warning and are skipped.
    """
    base: dict = {}
    if SYSTEM_CONFIG.exists():
        with open(SYSTEM_CONFIG, "r") as f:
            base = yaml.safe_load(f) or {}
    else:
        print(f"[warn] System config not found at '{SYSTEM_CONFIG}'.")

    model_cfg: dict = {}
    path = Path(model_config_path)
    if path.exists():
        with open(path, "r") as f:
            model_cfg = yaml.safe_load(f) or {}
    else:
        print(f"[warn] Model config not found at '{model_config_path}', using system defaults + CLI args.")

    return _deep_merge(base, model_cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XClinVision model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/efficientnet_b2.yaml", help="Path to config")
    parser.add_argument("--model", type=str, default="efficientnet_b2", help="Model architecture")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, choices=["focal", "ce"], default="focal")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override config weight_decay")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Override config label_smoothing")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cache-size", type=int, default=1000, help="LRU cache size for images")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--no-progressive-unfreeze", action="store_true", help="Disable progressive unfreezing")
    parser.add_argument("--deterministic", action="store_true", help="Enable CUDA deterministic mode")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config (CLI args take precedence over config file values)
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})

    # Resolve values: CLI flag > config > hardcoded default
    image_size = config.get("input", {}).get("size", [224, 224])
    image_size = image_size[0] if isinstance(image_size, list) else image_size
    weight_decay = args.weight_decay if args.weight_decay is not None else opt_cfg.get("weight_decay", 1e-4)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else train_cfg.get("label_smoothing", 0.1)

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # 1. Init DataModule
    data_module = ChestXrayDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        cache_size=args.cache_size,
        use_weighted_sampler=False,  # class_weights passed to loss instead
    )
    data_module.setup(stage="fit")
    
    # Compute class weights from training data
    class_weights = data_module.get_class_weights().tolist()
    print(f"Computed class weights: {class_weights}")

    # 2. Build Base Model
    base_model = build_model(
        model_name=args.model,
        num_classes=3,
        pretrained=not args.no_pretrained,
        img_size=image_size,
    )

    # 3. Setup Lightning Module
    pl_module = XClinVisionModel(
        model=base_model,
        num_classes=3,
        learning_rate=args.lr,
        weight_decay=weight_decay,
        loss_type=args.loss,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        progressive_unfreezing=not args.no_progressive_unfreeze,
    )

    # 4. Callbacks & Loggers
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"{args.model}_{run_ts}"
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            min_delta=0.001,
        ),
        ModelCheckpoint(
            dirpath=str(output_path),
            filename=f"{args.model}-{{epoch:02d}}-{{val_auc:.4f}}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
        ),
    ]

    # Add custom diagnostic callbacks
    callbacks.extend([
        MetricsCallback(
            save_predictions=True, 
            output_dir=str(output_path / "preds")
        ),
        XAIValidationCallback(
            every_n_epochs=5,
            output_dir=str(output_path / "xai_val"),
            architecture=args.model,
            image_size=image_size,
        ),
    ])
    
    # Using both TensorBoard and MLFlow loggers
    loggers = [
        TensorBoardLogger(save_dir=f"{args.output_dir}/logs", name=args.model),
        MLFlowLogger(experiment_name="xclinvision", run_name=f"{args.model}_{run_ts}"),
    ]

    # 5. Execute Training
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,
        deterministic=args.deterministic,
    )

    # Log full hyperparameters to experiment trackers
    hparams = {
        "model": args.model,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "loss": args.loss,
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
        "progressive_unfreezing": not args.no_progressive_unfreeze,
        "class_weights": class_weights,
        "num_workers": args.num_workers,
        "cache_size": args.cache_size,
        "deterministic": args.deterministic,
    }
    for lgr in trainer.loggers:
        lgr.log_hyperparams(hparams)

    print(f"\n--- Starting XClinVision Train | Model: {args.model} | Epochs: {args.epochs} | Image size: {image_size} ---")
    trainer.fit(pl_module, datamodule=data_module)

    # 6. Final Evaluation on test set
    # PL calls data_module.setup("test") internally before running test_dataloader()
    print("\n--- Running Final Evaluation ---")
    trainer.test(pl_module, datamodule=data_module)


if __name__ == "__main__":
    main()
