#!/usr/bin/env python
"""Main training script for XClinVision chest X-ray classification."""

import argparse
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path

import torch
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

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def log_dataset_summary(manifest_path: str) -> None:
    """Print a verbose dataset summary (directories + per-split/class counts)."""
    import pandas as pd

    manifest = Path(manifest_path)
    if not manifest.exists():
        print(f"[warn] Manifest not found: {manifest} – skipping dataset summary.")
        return

    df = pd.read_csv(manifest)
    required = {"split", "class", "filepath_processed"}
    if required - set(df.columns):
        print(f"[warn] Manifest missing columns {required - set(df.columns)} – skipping summary.")
        return

    splits = [s for s in ("train", "val", "test") if s in df["split"].unique()]
    classes = sorted(df["class"].unique().tolist())

    SEP  = "=" * 64
    DASH = "-" * 64
    print(SEP)
    print("  XClinVision  –  Dataset Summary")
    print(SEP)
    print(f"  Manifest : {manifest.resolve()}")
    print(DASH)

    grand_total = 0
    for split in splits:
        split_df = df[df["split"] == split]
        # Derive the split directory from the first filepath in this split
        sample_path = Path(split_df["filepath_processed"].iloc[0])
        # filepath_processed is typically  data/processed/<split>/<class>/img.jpg
        # so the split dir is two levels up from the file
        split_dir = sample_path.parent.parent.resolve()
        print(f"  Split : {split.upper():<5}  |  {split_dir}")
        split_total = 0
        for cls in classes:
            n = int((split_df["class"] == cls).sum())
            split_total += n
            print(f"    {cls:<18} : {n:>6} images")
        grand_total += split_total
        print(f"    {'SPLIT TOTAL':<18} : {split_total:>6} images")
        print(DASH)

    print(f"  {'GRAND TOTAL':<22} : {grand_total:>6} images")
    print(SEP)
    print()


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
    num_classes = config.get("model", {}).get("num_classes", 3)
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
        use_weighted_sampler=False,  
    )
    data_module.setup(stage="fit")

    # ---- verbose dataset summary ----------------------------------------
    log_dataset_summary(args.manifest)
    # ---------------------------------------------------------------------

    class_weights = data_module.get_class_weights().tolist()
    print(f"Computed class weights: {class_weights}")

    # 2. Build Base Model
    base_model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        img_size=image_size,
    )

    # 3. Setup Lightning Module
    pl_module = XClinVisionModel(
        model=base_model,
        num_classes=num_classes,
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
            monitor="val_loss",
            mode="min",
            patience=7,
            min_delta=0.001,
        ),
        ModelCheckpoint(
            dirpath=str(output_path),
            filename=f"{args.model}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

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
    try:
        trainer.fit(pl_module, datamodule=data_module)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            f"\n[OOM] Training '{args.model}' ran out of GPU memory.\n"
            f"  Current batch_size : {args.batch_size}\n"
            f"  Current img_size   : {image_size}\n"
            f"  Suggestions:\n"
            f"    --batch-size {max(1, args.batch_size // 2)}   (halve batch size)\n"
            f"    export PYTORCH_ALLOC_CONF=expandable_segments:True\n"
        )
        sys.exit(1)

    # 6. Final Evaluation on test set
    print("\n--- Running Final Evaluation ---")
    trainer.test(pl_module, datamodule=data_module)

if __name__ == "__main__":
    main()
