#!/usr/bin/env python
"""Evaluation script for XClinVision models.

Performs detailed metrics, calibration, failure analysis, and saves a full
evaluation report (JSON + text classification report + per-sample CSV).

Usage
-----
python scripts/evaluate.py \
    --checkpoint-path models/efficientnet_b2_best.ckpt \
    --model-name efficientnet_b2 \
    --manifest data/processed/manifest.csv \
    --output-dir outputs/evaluation \
    --calibrate
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Silence Lightning deprecation warnings
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xclinvision.dataset import ChestXrayDataModule
from xclinvision.evaluator import (
    CalibrationAnalyzer,
    MetricsComputer,
    TemperatureScaler,
)
from xclinvision.modeling import build_model, get_model_normalization
from xclinvision.processing import get_processed_dir_for_size
from xclinvision.reliability import FailureAnalyzer
from xclinvision.trainer import XClinVisionModel
from xclinvision.config import get_class_names, get_num_classes, is_multilabel, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xclinvision.evaluate")


# ---------------------------------------------------------------------------
# Dynamic processed data resolution
# ---------------------------------------------------------------------------
# get_processed_dir_for_size is imported from xclinvision.processing
# (single source of truth shared with scripts/train.py).


def resolve_manifest_path(
    manifest_arg: str | None, 
    image_size: int, 
    processed_dir: str
) -> str:
    """Resolve manifest path: explicit arg takes precedence, else derive from image_size."""
    if manifest_arg:
        return manifest_arg
    return str(get_processed_dir_for_size(processed_dir, image_size, get_class_names()) / "manifest.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an XClinVision model checkpoint on the test split."
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to the Lightning .ckpt checkpoint file."
    )
    parser.add_argument(
        "--model-name", type=str, default="efficientnet_b2",
        help="Architecture name (must match the checkpoint). Default: efficientnet_b2."
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to the processed dataset manifest CSV. Default: auto-resolved from --processed-dir and --image-size."
    )
    parser.add_argument(
        "--processed-dir", type=str, default="data/processed",
        help="Base processed data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evaluation",
        help="Directory to write evaluation artefacts. Default: outputs/evaluation."
    )
    parser.add_argument(
        "--image-size", type=int, default=384,
        help="Image resolution (must match training). Default: 384."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference. Default: 32."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes. Default: 4."
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run temperature scaling calibration using the validation set."
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"],
        help="Dataset split to evaluate on. Default: test."
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable test-time augmentation (horizontal flip + multi-scale averaging)."
    )
    parser.add_argument(
        "--pooling", type=str, choices=["avg", "gem"], default="avg",
        help="Pooling type used during training. Default: avg."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Inference",
    tta: bool = False,
    norm_mean: tuple = (0.485, 0.456, 0.406),
    norm_std: tuple = (0.229, 0.224, 0.225),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model over all batches and collect arrays.

    When *tta* is True, predictions are averaged over the original image
    and a horizontally-flipped copy for improved robustness.

    Returns:
        y_true   (N,)   – ground-truth integer labels
        logits   (N, C) – raw unnormalised logits
        y_probs  (N, C) – softmax probabilities
    """
    all_targets, all_logits = [], []

    model.eval()
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device)
        logits = model(x)
        if tta:
            # Fix #4: multi-augmentation TTA — horizontal flip + slight brightness
            # boost averaged with the original logits for better robustness.
            # Brightness is applied in the *pre-normalized* pixel space to avoid
            # corrupting the normalized tensor distribution:
            #   x_bright = Normalize((x * std + mean) * 1.05)
            # Algebraically: x_bright = x * 1.05 + mean * 0.05 / std
            mean = torch.tensor(norm_mean, device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor(norm_std, device=x.device).view(1, 3, 1, 1)
            x_bright = x * 1.05 + mean * 0.05 / std

            logits_flip   = model(torch.flip(x, dims=[-1]))   # horizontal flip
            logits_bright = model(x_bright)
            logits = (logits + logits_flip + logits_bright) / 3.0
        all_targets.append(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    y_true  = np.concatenate(all_targets, axis=0)
    logits  = np.concatenate(all_logits,  axis=0)
    if is_multilabel():
        y_probs = _sigmoid(logits)
    else:
        y_probs = _softmax(logits)
    return y_true, logits, y_probs


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)

    # Resolve manifest path dynamically
    manifest_path = resolve_manifest_path(
        args.manifest, 
        args.image_size, 
        args.processed_dir
    )
    
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_file.resolve()}. "
            f"Ensure data was processed for image_size={args.image_size}."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Split:  {args.split}")
    logger.info(f"Manifest: {manifest_file.resolve()}")

    # ------------------------------------------------------------------
    # 1. Model
    # ------------------------------------------------------------------
    logger.info(f"Loading checkpoint: {args.checkpoint_path}")
    base_model = build_model(
        model_name=args.model_name,
        num_classes=get_num_classes(),
        pretrained=False,
        img_size=args.image_size,
        pooling=args.pooling,
    )

    norm_stats = get_model_normalization(base_model, args.model_name)

    pl_module = XClinVisionModel.load_from_checkpoint(
        args.checkpoint_path,
        model=base_model,
        map_location="cpu",
    )
    model = pl_module.model
    model.to(device)
    model.eval()
    logger.info("Checkpoint loaded.")

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    logger.info("Initialising DataModule …")
    eval_config = PipelineConfig.from_yaml()
    data_module = ChestXrayDataModule(
        manifest_path=manifest_path,
        config=eval_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
    )

    if args.split == "test":
        data_module.setup(stage="test")
        eval_loader = data_module.test_dataloader()
    else:
        data_module.setup(stage="fit")
        eval_loader = data_module.val_dataloader()

    n_eval = len(eval_loader.dataset)
    logger.info(f"Evaluation dataset: {n_eval} samples")

    # ------------------------------------------------------------------
    # 3. Test-set predictions
    # ------------------------------------------------------------------
    logger.info("Running inference …")
    y_true, logits_eval, y_probs = collect_predictions(
        model, eval_loader, device, desc=f"Eval ({args.split})", tta=args.tta,
        norm_mean=norm_stats["mean"], norm_std=norm_stats["std"],
    )
    _multilabel = is_multilabel()
    if _multilabel:
        y_pred = (y_probs > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_probs, axis=1)

    # ------------------------------------------------------------------
    # 4. Optional: Temperature Scaling (calibration)
    # ------------------------------------------------------------------
    temperature_scaler: TemperatureScaler | None = None
    learned_temperature: float = 1.0

    if args.calibrate:
        if args.split == "val":
            logger.warning(
                "⚠️ DATA LEAKAGE WARNING: You are fitting Temperature Scaling on the "
                "Validation set, and evaluating on the Validation set. Your ECE and "
                "calibration curve will be artificially over-optimistic!"
            )
        logger.info("Collecting validation logits for temperature scaling …")
        if args.split == "test":
            data_module.setup(stage="fit")
        val_loader = data_module.val_dataloader()

        y_val, logits_val, _ = collect_predictions(
            model, val_loader, device, desc="Calibration (val)",
            norm_mean=norm_stats["mean"], norm_std=norm_stats["std"],
        )

        temperature_scaler = TemperatureScaler()
        if _multilabel:
            logger.warning(
                "Temperature scaling is designed for multiclass (softmax). "
                "Skipping calibration for multilabel mode."
            )
            temperature_scaler = None
        else:
            learned_temperature = temperature_scaler.fit(logits_val, y_val)
            logger.info(f"Optimal temperature: T = {learned_temperature:.4f}")

        # Re-calibrate eval predictions
        if temperature_scaler is not None:
            y_probs = temperature_scaler.predict_proba(logits_eval)
            if _multilabel:
                y_pred = (y_probs > 0.5).astype(int)
            else:
                y_pred  = np.argmax(y_probs, axis=1)

            # Persist scaler
            temp_path = output_dir / f"{args.model_name}_temperature.json"
            temperature_scaler.save(str(temp_path))

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics …")
    class_names = get_class_names()
    metrics_computer = MetricsComputer(class_names=class_names)

    metrics = metrics_computer.compute_all_metrics(y_true, y_pred, y_probs)
    cm      = metrics_computer.compute_confusion_matrix(y_true, y_pred)

    metrics["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "labels": class_names,
    }

    # Print summary table
    metrics_computer.print_summary(metrics, y_true=y_true, y_pred=y_pred)

    # ------------------------------------------------------------------
    # 6. Calibration analysis
    # ------------------------------------------------------------------
    logger.info("Computing calibration metrics …")
    calibrator = CalibrationAnalyzer(num_bins=15)
    if _multilabel:
        # CalibrationAnalyzer uses argmax internally — not meaningful for multilabel.
        # Skip ECE / calibration curve for now.
        ece = 0.0
        bin_centers = np.linspace(0, 1, 15)
        bin_accs = np.zeros(15)
        bin_counts = np.zeros(15, dtype=int)
        logger.info("Skipping ECE / calibration curve (not defined for multilabel).")
    else:
        ece = calibrator.compute_ece(y_true, y_probs)
        bin_centers, bin_accs, bin_counts = calibrator.compute_calibration_curve(
            y_true, y_probs
        )

    metrics["calibration"] = {
        "expected_calibration_error": float(ece),
        "temperature": float(learned_temperature),
        "curve_data": {
            "bin_centers":    bin_centers.tolist(),
            "bin_accuracies": bin_accs.tolist(),
            "bin_counts":     bin_counts.tolist(),
        },
    }
    logger.info(f"ECE: {ece:.4f}")

    # ------------------------------------------------------------------
    # 7. Failure analysis
    # ------------------------------------------------------------------
    logger.info("Analysing failures …")
    failure_analyzer = FailureAnalyzer(class_names=class_names, multilabel=is_multilabel())
    failures = failure_analyzer.analyze_failures(y_true, y_pred, y_probs)
    metrics["failure_analysis"] = failures

    for cls, fp in failures["false_positives"].items():
        logger.info(f"  False positives  [{cls}]: {fp['count']}")
    for cls, fn in failures["false_negatives"].items():
        logger.info(f"  False negatives  [{cls}]: {fn['count']}")
    hce = failures["high_confidence_errors"]
    logger.info(
        f"  High-confidence errors: {hce['count']} "
        f"(avg conf = {hce['avg_confidence']:.3f})"
    )

    # ------------------------------------------------------------------
    # 8. Per-sample CSV
    # ------------------------------------------------------------------
    logger.info("Writing per-sample predictions …")
    
    # Extract file paths from the dataset for proper traceability
    dataset_ref = data_module.test_dataset if args.split == "test" else data_module.val_dataset
    filepaths = dataset_ref.df["filepath_processed"].tolist()
    
    prob_cols = {
        f"prob_{name}": y_probs[:, i]
        for i, name in enumerate(class_names)
    }

    if _multilabel:
        # y_true / y_pred are (N, C) binary matrices
        true_label_cols = {
            f"true_{name}": y_true[:, i] for i, name in enumerate(class_names)
        }
        pred_label_cols = {
            f"pred_{name}": y_pred[:, i] for i, name in enumerate(class_names)
        }
        df_preds = pd.DataFrame(
            {
                "filepath": filepaths,
                **true_label_cols,
                **pred_label_cols,
                "confidence": np.max(y_probs, axis=1),
                **prob_cols,
            }
        )
    else:
        df_preds = pd.DataFrame(
            {
                "filepath":         filepaths,
                "true_label":       y_true,
                "true_class":       [class_names[lbl] for lbl in y_true],
                "predicted_label":  y_pred,
                "predicted_class":  [class_names[p] for p in y_pred],
                "confidence":       np.max(y_probs, axis=1),
                "correct":          (y_true == y_pred),
                **prob_cols,
            }
        )
    csv_path = output_dir / f"{args.model_name}_{args.split}_predictions.csv"
    df_preds.to_csv(csv_path, index=False)
    logger.info(f"Per-sample CSV saved to {csv_path}")

    # ------------------------------------------------------------------
    # 9. Text classification report
    # ------------------------------------------------------------------
    report_str = metrics_computer.generate_classification_report(y_true, y_pred)
    report_txt_path = output_dir / f"{args.model_name}_{args.split}_classification_report.txt"
    report_txt_path.write_text(report_str)
    logger.info("\nClassification Report:\n" + report_str)

    # ------------------------------------------------------------------
    # 10. JSON report
    # ------------------------------------------------------------------
    report_path = output_dir / f"{args.model_name}_{args.split}_evaluation_report.json"
    metrics_computer.save_results(metrics, str(report_path))
    logger.info(f"Full report saved to {report_path}")

    # Final summary line
    acc_key = 'subset_accuracy' if _multilabel else 'accuracy'
    logger.info(
        f"Done — Accuracy={metrics[acc_key]:.4f}  "
        f"MacroF1={metrics['macro_f1']:.4f}  "
        f"MacroAUC={metrics['macro_auc']:.4f}  "
        f"ECE={ece:.4f}"
    )


if __name__ == "__main__":
    main()
