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

torch.set_float32_matmul_precision("high")

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
from pytorch_lightning.loggers import TensorBoardLogger
try:
    from pytorch_lightning.loggers import MLFlowLogger
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    MLFlowLogger = None  # type: ignore[assignment,misc]

from xclinvision.dataset import ChestXrayDataModule
from xclinvision.modeling import build_model, get_model_normalization
from xclinvision.processing import run_processing_pipeline, PROCESSING_VERSION, get_processed_dir_for_size
from xclinvision.trainer import (
    XClinVisionModel,
    MetricsCallback,
    XAIValidationCallback,
    BestModelExportCallback,
)
from xclinvision.config import get_class_names

SYSTEM_CONFIG = Path(__file__).parent.parent / "configs" / "system.yaml"

# ---------------------------------------------------------------------------
# Dynamic processed data caching
# ---------------------------------------------------------------------------
# get_processed_dir_for_size is imported from xclinvision.processing
# (single source of truth shared with scripts/evaluate.py).


def ensure_processed_data_exists(
    image_size: int,
    raw_dir: str,
    processed_base_dir: str,
    quarantine_dir: str,
    force_reprocess: bool = False,
    class_names: list | None = None,
) -> Path:
    """
    Check if processed data exists for the given image_size and class set.
    If not, automatically run the processing pipeline.

    The directory name encodes both the image size and a hash of the class names,
    so changing system.yaml class_names automatically triggers a re-run without
    needing --force-reprocess.

    Returns: Path to the processed directory for this image_size
    """
    processed_dir = get_processed_dir_for_size(processed_base_dir, image_size, class_names)
    manifest_path = processed_dir / "manifest.csv"
    
    if manifest_path.exists() and not force_reprocess:
        print(f"[Info] Found existing processed data at {processed_dir}. Skipping processing.")
        return processed_dir
    
    # Clean stale outputs so class changes don't leave orphaned files
    if force_reprocess:
        import shutil
        for stale_dir in [processed_dir, Path(quarantine_dir)]:
            if stale_dir.exists():
                print(f"[Info] Cleaning {stale_dir} before reprocessing...")
                shutil.rmtree(stale_dir)
        duplicate_dir = Path(processed_base_dir).parent / "duplicate"
        if duplicate_dir.exists():
            print(f"[Info] Cleaning {duplicate_dir} before reprocessing...")
            shutil.rmtree(duplicate_dir)

    print(f"[Info] Processed data not found for image_size={image_size}.")
    print(f"[Info] Running processing pipeline to generate {processed_dir} (this happens once)...")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir_full = Path(quarantine_dir)
    
    report = run_processing_pipeline(
        raw_dir=raw_dir,
        processed_dir=str(processed_dir),
        quarantine_dir=str(quarantine_dir_full),
        target_size=image_size,
        detect_duplicates=True,
    )
    
    if report.processed == 0:
        raise RuntimeError(
            f"Processing pipeline produced 0 valid images. "
            f"Check {quarantine_dir_full} for quarantine reasons."
        )
    
    print(f"[Info] ✓ Processing complete. {report.processed} images saved to {processed_dir}")
    return processed_dir

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=None, help="Override config image size (e.g. 512)")
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
    parser.add_argument("--force-reprocess", action="store_true", help="Force re-processing even if data exists")
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=2,
        help="Gradient accumulation steps. Effective batch size = batch_size × this value. LR is scaled proportionally. Default: 2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config (CLI args take precedence over config file values)
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})
    paths_cfg = config.get("paths", {})

    # Resolve values: CLI flag > config > hardcoded default
    if args.image_size:
        image_size = args.image_size
    else:
        raw_size = config.get("input", {}).get("size", [224, 224])
        image_size = raw_size[0] if isinstance(raw_size, list) else raw_size
    # Single source of truth: system.yaml → model.class_names via get_class_names().
    # Do NOT fall back to a hardcoded list here; if the config is missing the
    # pipeline should fail loudly rather than silently train on the wrong classes.
    class_names = get_class_names()
    num_classes = len(class_names)
    weight_decay = args.weight_decay if args.weight_decay is not None else opt_cfg.get("weight_decay", 1e-4)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else train_cfg.get("label_smoothing", 0.1)

    # Set up dynamically processed data
    raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
    processed_base_dir = paths_cfg.get("processed_data_dir", "data/processed")
    quarantine_dir = paths_cfg.get("quarantine_dir", "data/quarantine")
    
    # Ensure processed data exists at this specific resolution and class set
    processed_dir = ensure_processed_data_exists(
        image_size=image_size,
        raw_dir=raw_dir,
        processed_base_dir=processed_base_dir,
        quarantine_dir=quarantine_dir,
        force_reprocess=args.force_reprocess,
        class_names=class_names,
    )
    
    manifest_path = str(processed_dir / "manifest.csv")

    # Set seed
    pl.seed_everything(args.seed, workers=True)
    if args.deterministic:
        # Enable op-level determinism; warn_only=True avoids hard errors for
        # the handful of ops (e.g. upsample_bilinear2d) that have no
        # deterministic kernel, while still flagging non-deterministic paths.
        torch.use_deterministic_algorithms(True, warn_only=True)

    # 2. Build Base Model
    base_model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        img_size=image_size,
    )

    # Use model-specific normalization stats
    norm_stats = get_model_normalization(base_model, args.model)

    # 1. Init DataModule
    use_weighted_sampler = True  # set False to disable and use loss-level weighting instead
    aug_cfg = config.get("augmentation", {})
    horizontal_flip = aug_cfg.get("horizontal_flip", True)
    data_module = ChestXrayDataModule(
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        cache_size=args.cache_size,
        use_weighted_sampler=use_weighted_sampler,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        horizontal_flip=horizontal_flip,
    )
    data_module.setup(stage="fit")

    # 3. Setup Lightning Module
    # ---- verbose dataset summary ----------------------------------------
    log_dataset_summary(manifest_path)
    # ---------------------------------------------------------------------

    class_weights = data_module.get_class_weights().tolist()
    print(f"Computed class weights: {class_weights}")
    # When WeightedRandomSampler is active it already re-balances per-batch class
    # representation, so passing class_weights to the loss as well would
    # double-correct for imbalance and destabilise training.
    # When the sampler is disabled, fall back to loss-level class weighting.
    class_weights_for_loss = None if use_weighted_sampler else class_weights

    # Scale LR before constructing the module so save_hyperparameters captures
    # the effective learning rate (args.lr × accumulate_grad_batches), not the
    # raw CLI value.
    accumulate_grad_batches = args.accumulate_grad_batches
    effective_lr = args.lr * accumulate_grad_batches

    pl_module = XClinVisionModel(
        model=base_model,
        num_classes=num_classes,
        learning_rate=effective_lr,
        weight_decay=weight_decay,
        loss_type=args.loss,
        label_smoothing=label_smoothing,
        class_weights=class_weights_for_loss,
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
            monitor="val_f1_macro",
            mode="max",
            patience=7,
            min_delta=0.001,
        ),
        ModelCheckpoint(
            dirpath=str(output_path),
            filename=f"{args.model}-{{epoch:02d}}-{{val_f1_macro:.4f}}",
            monitor="val_f1_macro",
            mode="max",
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
        BestModelExportCallback(
            model_name=args.model,
            # Anchor to project root so the path is stable regardless of CWD.
            export_dir=str(Path(__file__).parent.parent / args.output_dir / "best_models"),
            num_classes=num_classes,
            class_names=class_names,
        ),
    ])
    
    loggers = [
        TensorBoardLogger(
            save_dir=str(output_path), 
            name="tensorboard",
            version="",
        ),
    ]
    if _MLFLOW_AVAILABLE:
        loggers.append(
            MLFlowLogger(experiment_name="xclinvision", run_name=f"{args.model}_{run_ts}")
        )
    else:
        print("[Info] MLflow not installed — skipping MLflow logging. "
              "Install with: pip install mlflow")

    # 5. Execute Training
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=args.deterministic,
    )
    
    if not args.no_progressive_unfreeze:
        print(
            "\n[Info] Skipping LR finder — progressive unfreezing is enabled.\n"
            "       The backbone is currently frozen; the finder would return a "
            "head-only LR that is too high for full fine-tuning.\n"
            f"       Using config LR: {pl_module.learning_rate:.2e}"
        )
    else:
        print("\n--- Running Learning Rate Finder ---")
        tuner = pl.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(pl_module, datamodule=data_module, min_lr=1e-6, max_lr=1e-2)

        if lr_finder is not None:
            fig = lr_finder.plot(suggest=True)
            print(f"Suggested LR: {lr_finder.suggestion()}")
            pl_module.learning_rate = lr_finder.suggestion()
            print(f"Applied suggested LR: {pl_module.learning_rate}")
        else:
            print("Learning rate finder failed to suggest a learning rate.")

    torch.cuda.empty_cache()
    # ---------------------------

    # Log full hyperparameters to experiment trackers
    hparams = {
        "model": args.model,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": pl_module.learning_rate,
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
        "accumulate_grad_batches": accumulate_grad_batches,
    }
    for lgr in trainer.loggers:
        lgr.log_hyperparams(hparams)

    print(f"\n--- Starting XClinVision Train | Model: {args.model} | Epochs: {args.epochs} | Image size: {image_size} ---")
    try:
        trainer.fit(pl_module, datamodule=data_module)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Training interrupted by user. Proceeding to evaluation...\n")
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
    trainer.test(pl_module, datamodule=data_module, ckpt_path="best")

if __name__ == "__main__":
    main()
