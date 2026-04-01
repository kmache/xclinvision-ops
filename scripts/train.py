#!/usr/bin/env python
"""Main training script for XClinVision chest X-ray classification."""

import argparse
import sys
import logging
import warnings
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

torch.set_float32_matmul_precision("high")

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

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from xclinvision.dataset import ChestXrayDataModule
from xclinvision.modeling import build_model, get_model_normalization
from xclinvision.processing import run_processing_pipeline, PROCESSING_VERSION, get_processed_dir_for_size
from xclinvision.trainer import (
    XClinVisionModel,
    MetricsCallback,
    XAIValidationCallback,
    BestModelExportCallback,
)
from xclinvision.config import get_class_names, is_multilabel, PipelineConfig

SYSTEM_CONFIG = Path(__file__).parent.parent / "configs" / "system.yaml"


# ---------------------------------------------------------------------------
# Dynamic processed data caching
# ---------------------------------------------------------------------------

def ensure_processed_data_exists(
    image_size: int,
    raw_dir: str,
    processed_base_dir: str,
    quarantine_dir: str,
    force_reprocess: bool = False,
    class_names: list | None = None,
) -> Path:
    """Check if processed data exists; if not, run the processing pipeline."""
    processed_dir = get_processed_dir_for_size(processed_base_dir, image_size, class_names)
    manifest_path = processed_dir / "manifest.csv"
    
    if manifest_path.exists() and not force_reprocess:
        _current_names = class_names or get_class_names()
        try:
            _cols = set(pd.read_csv(manifest_path, nrows=0).columns)
            if is_multilabel():
                missing = set(_current_names) - _cols
                if missing:
                    print(
                        f"[Info] Processed data is stale: manifest is missing class columns "
                        f"{sorted(missing)}. Re-processing..."
                    )
                    force_reprocess = True
        except Exception:
            pass

        if not force_reprocess:
            print(f"[Info] Found existing processed data at {processed_dir}. Skipping processing.")
            return processed_dir
    
    if force_reprocess:
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
    
    pipeline_config = PipelineConfig.from_yaml()
    report = run_processing_pipeline(
        raw_dir=raw_dir,
        processed_dir=str(processed_dir),
        quarantine_dir=str(quarantine_dir_full),
        target_size=image_size,
        detect_duplicates=True,
        config=pipeline_config,
    )
    
    if report.processed == 0:
        raise RuntimeError(
            f"Processing pipeline produced 0 valid images. "
            f"Check {quarantine_dir_full} for quarantine reasons."
        )
    
    print(f"[Info] ✓ Processing complete. {report.processed} images saved to {processed_dir}")
    return processed_dir


# ---------------------------------------------------------------------------
# Dataset summary & Config Logic
# ---------------------------------------------------------------------------

def log_dataset_summary(manifest_path: str) -> None:
    """Print a verbose dataset summary."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        print(f"[warn] Manifest not found: {manifest} – skipping dataset summary.")
        return

    df = pd.read_csv(manifest)
    required = {"split", "filepath_processed"}
    if required - set(df.columns):
        print(f"[warn] Manifest missing columns {required - set(df.columns)} – skipping summary.")
        return

    _multilabel = is_multilabel()
    class_names = get_class_names()
    splits = [s for s in ("train", "val", "test") if s in df["split"].unique()]

    SEP  = "=" * 64
    DASH = "-" * 64
    print(SEP)
    print("  XClinVision  –  Dataset Summary")
    print(SEP)
    print(f"  Manifest : {manifest.resolve()}")
    print(f"  Mode     : {'multilabel' if _multilabel else 'multiclass'}")
    print(DASH)

    grand_total = 0
    for split in splits:
        split_df = df[df["split"] == split]
        print(f"  Split : {split.upper():<5}  ({len(split_df)} samples)")
        if _multilabel:
            for cls in class_names:
                if cls in split_df.columns:
                    n = int(split_df[cls].sum())
                    print(f"    {cls:<28} : {n:>6} positive")
        else:
            if "class" in split_df.columns:
                classes = sorted(split_df["class"].unique().tolist())
                for cls in classes:
                    n = int((split_df["class"] == cls).sum())
                    print(f"    {cls:<28} : {n:>6} images")
        grand_total += len(split_df)
        print(DASH)

    print(f"  {'GRAND TOTAL':<22} : {grand_total:>6} samples")
    print(SEP)
    print()

def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged

def load_config(model_config_path: str) -> dict:
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
        print(f"[warn] Model config not found at '{model_config_path}', using defaults.")

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
    parser.add_argument("--image-size", type=int, default=None, help="Override config image size")
    parser.add_argument("--process-size", type=int, default=1024, help="Resolution for storing processed images on disk")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to pre-processed manifest CSV (skips auto-processing)")
    parser.add_argument("--loss", type=str, choices=["focal", "ce", "asl"], default="ce")
    parser.add_argument("--pooling", type=str, choices=["avg", "gem"], default="gem", help="Global pooling type")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-progressive-unfreeze", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true", help="Disable weighted random sampler")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true")
    
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--monitor", type=str, choices=["val_f1_macro", "val_auc"], default="val_f1_macro",
                        help="Metric to monitor for checkpointing and early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--devices", type=str, default="auto", help="Number of GPUs or 'auto'")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training")
    
    # NEW: Dedicated LR Finder Argument
    parser.add_argument("--lr-finder", action="store_true", help="Run Learning Rate Finder before training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})
    paths_cfg = config.get("paths", {})
    transfer_cfg = config.get("transfer_learning", {})
    diff_lr_cfg = transfer_cfg.get("differential_lr", {})

    if args.image_size:
        image_size = args.image_size
    else:
        raw_size = config.get("input", {}).get("size", [384, 384])
        image_size = raw_size[0] if isinstance(raw_size, list) else int(raw_size)

    process_size = args.process_size
        
    class_names = get_class_names()
    num_classes = len(class_names)
    weight_decay = args.weight_decay if args.weight_decay is not None else opt_cfg.get("weight_decay", 1e-4)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else train_cfg.get("label_smoothing", 0.1)

    # Read loss type from model config if not overridden via CLI
    loss_type = args.loss if args.loss != "ce" else train_cfg.get("loss", "ce")

    # Read differential LR factor from config
    head_lr_cfg = diff_lr_cfg.get("head_lr")
    backbone_lr_cfg = diff_lr_cfg.get("backbone_lr")
    if head_lr_cfg and backbone_lr_cfg and float(head_lr_cfg) > 0:
        backbone_lr_factor = float(backbone_lr_cfg) / float(head_lr_cfg)
    else:
        backbone_lr_factor = 0.1
    
    devices_parsed = "auto" if args.devices.lower() == "auto" else int(args.devices)
    use_weighted_sampler = not args.no_weighted_sampler

    if args.manifest:
        manifest_path = args.manifest
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        print(f"[Info] Using provided manifest: {manifest_path}")
    else:
        raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
        processed_base_dir = paths_cfg.get("processed_data_dir", "data/processed")
        quarantine_dir = paths_cfg.get("quarantine_dir", "data/quarantine")
        
        processed_dir = ensure_processed_data_exists(
            image_size=process_size,
            raw_dir=raw_dir,
            processed_base_dir=processed_base_dir,
            quarantine_dir=quarantine_dir,
            force_reprocess=args.force_reprocess,
            class_names=class_names,
        )
        manifest_path = str(processed_dir / "manifest.csv")

    pl.seed_everything(args.seed, workers=True)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    arch_cfg = config.get("architecture", {})
    dropout_rate = arch_cfg.get("dropout_rate", 0.3)

    base_model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        img_size=image_size,
        dropout=dropout_rate,
        pooling=args.pooling,
    )
    norm_stats = get_model_normalization(base_model, args.model)

    full_config = PipelineConfig.from_yaml()
    disease_config = PipelineConfig(
        class_names=class_names,
        classification_mode=full_config.classification_mode,
        clinical_rules={
            k: v for k, v in full_config.clinical_rules.items()
            if k != 'no finding'
        },
    )

    aug_cfg = config.get("augmentation", {})
    data_module = ChestXrayDataModule(
        manifest_path=manifest_path,
        config=disease_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        cache_size=args.cache_size,
        use_weighted_sampler=use_weighted_sampler,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        horizontal_flip=aug_cfg.get("horizontal_flip", True),
    )
    data_module.setup(stage="fit")

    log_dataset_summary(manifest_path)

    class_weights = data_module.get_class_weights().tolist()
    print(f"Computed class weights: {class_weights}")
    class_weights_for_loss = class_weights
    
    effective_lr = args.lr

    pl_module = XClinVisionModel(
        model=base_model,
        num_classes=num_classes,
        learning_rate=effective_lr,
        weight_decay=weight_decay,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        class_weights=class_weights_for_loss,
        progressive_unfreezing=not args.no_progressive_unfreeze,
        config=disease_config,
        backbone_lr_factor=backbone_lr_factor,
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"{args.model}_{run_ts}"
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        EarlyStopping(
            monitor=args.monitor,
            mode="max",
            patience=args.patience,
            min_delta=0.0001,
        ),
        ModelCheckpoint(
            dirpath=str(output_path),
            filename=f"{args.model}-{{epoch:02d}}-{{{args.monitor}:.4f}}",
            monitor=args.monitor,
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
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
            export_dir=str(Path(__file__).parent.parent / args.output_dir / "best_models"),
            num_classes=num_classes,
            class_names=class_names,
        ),
    ]
    
    loggers = [
        TensorBoardLogger(save_dir=str(output_path), name="tensorboard", version=""),
    ]
    if _MLFLOW_AVAILABLE:
        loggers.append(MLFlowLogger(experiment_name="xclinvision", run_name=f"{args.model}_{run_ts}"))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=devices_parsed,
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=args.deterministic,
    )
    
    # ---------------------------------------------------------
    # LR Finder Logic
    # ---------------------------------------------------------
    if args.lr_finder:
        if not args.no_progressive_unfreeze:
            print(
                "\n[Warning] Running LR finder with progressive unfreezing enabled.\n"
                "          The backbone is currently frozen, so the suggested LR will be for the head only.\n"
                "          This might be too high when the backbone unfreezes later."
            )
            
        print("\n--- Running Learning Rate Finder ---")
        tuner = pl.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(pl_module, datamodule=data_module, min_lr=1e-6, max_lr=1e-2)

        if lr_finder is not None:
            print(f"Suggested LR: {lr_finder.suggestion()}")
            pl_module.learning_rate = lr_finder.suggestion()
            print(f"Applied suggested LR: {pl_module.learning_rate}")
            
            if _MATPLOTLIB_AVAILABLE:
                fig = lr_finder.plot(suggest=True)
                plot_path = output_path / "lr_finder.png"
                fig.savefig(plot_path)
                plt.close(fig)
                print(f"Saved LR finder plot to {plot_path}")
        else:
            print("Learning rate finder failed to suggest a learning rate.")
    else:
        print(f"\n[Info] Skipping LR finder. Using config LR: {pl_module.learning_rate:.2e}")
    # ---------------------------------------------------------

    torch.cuda.empty_cache()

    hparams = {
        "model": args.model,
        "num_classes_model": num_classes,
        "num_classes_manifest": len(class_names),
        "pathology_classes": class_names,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": pl_module.learning_rate,
        "loss": args.loss,
        "pooling": args.pooling,
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
        "computed_class_weights": str(class_weights),
        "num_workers": args.num_workers,
        "cache_size": args.cache_size,
        "deterministic": args.deterministic,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "weighted_sampler_enabled": use_weighted_sampler,
    }
    for lgr in trainer.loggers:
        lgr.log_hyperparams(hparams)

    print(f"\n--- Starting XClinVision Train | Model: {args.model} | Epochs: {args.epochs} | Image size: {image_size} ---")
    try:
        trainer.fit(pl_module, datamodule=data_module, ckpt_path=args.resume_from)
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
            f"    --accumulate-grad-batches {args.accumulate_grad_batches * 2}\n"
            f"    export PYTORCH_ALLOC_CONF=expandable_segments:True\n"
        )
        sys.exit(1)

    print("\n--- Running Final Evaluation ---")
    trainer.test(pl_module, datamodule=data_module, ckpt_path="best")

if __name__ == "__main__":
    main()