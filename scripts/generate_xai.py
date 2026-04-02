#!/usr/bin/env python
"""Generate XAI heatmaps for XClinVision models.

Selects random test-set samples (stratified by class) and produces
side-by-side visualisations.

- **CNN / Swin models**: Original | Grad-CAM++ | Score-CAM
- **ViT models**: Original | Attention Rollout | Grad-CAM++

Usage
-----
python scripts/generate_xai.py \
    --checkpoint-path models/convnext_small_.../best.ckpt \
    --model-name convnext_small \
    --image-size 384 \
    --pooling gem \
    --output-dir outputs/xai_384 \
    --num-samples 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"torch|lightning")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xclinvision.config import get_class_names, get_num_classes, PipelineConfig
from xclinvision.dataset import ChestXrayDataModule
from xclinvision.modeling import build_model, get_model_normalization, get_target_layer
from xclinvision.trainer import XClinVisionModel
from xclinvision.xai import ExplainabilityEngine
from xclinvision.processing import get_processed_dir_for_size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xclinvision.generate_xai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XAI heatmaps (Attention Rollout for ViTs, Grad-CAM++/Score-CAM for CNNs)")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="convnext_small")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--pooling", type=str, choices=["avg", "gem"], default="gem")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs/xai")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_raw_image(filepath: str, img_size: int) -> np.ndarray:
    """Load image from disk and resize to img_size, returning float32 RGB in [0,1]."""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img.astype(np.float32) / 255.0


def save_comparison(
    original: np.ndarray,
    gradcam_overlay: np.ndarray,
    scorecam_overlay: np.ndarray,
    title: str,
    output_path: Path,
    primary_label: str = "Grad-CAM++",
    secondary_label: str = "Score-CAM",
) -> None:
    """Save a side-by-side comparison: original | primary | secondary."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(original)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(gradcam_overlay)
        axes[1].set_title(primary_label, fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(scorecam_overlay)
        axes[2].set_title(secondary_label, fontsize=12)
        axes[2].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        # Fallback: save individual images via OpenCV
        h = original.shape[0]
        w = original.shape[1]
        canvas = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
        for i, img in enumerate([original, gradcam_overlay, scorecam_overlay]):
            out = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img
            canvas[:, i * (w + 10): i * (w + 10) + w] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), canvas)

    logger.info(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class_names = get_class_names()
    num_classes = get_num_classes()

    # Resolve manifest
    if args.manifest:
        manifest_path = args.manifest
    else:
        manifest_path = str(
            get_processed_dir_for_size(args.processed_dir, args.image_size, class_names)
            / "manifest.csv"
        )
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {args.model_name} from {args.checkpoint_path}")
    base_model = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
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
    logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # 2. Resolve Grad-CAM target layer
    # ------------------------------------------------------------------
    # Detect ViT models — use Attention Rollout instead of Score-CAM
    is_vit = args.model_name.startswith("vit")

    target_layer = get_target_layer(model)
    if target_layer is None and not is_vit:
        raise RuntimeError("Could not resolve Grad-CAM target layer for this model.")
    if is_vit:
        logger.info("ViT detected — using Attention Rollout (primary) + Grad-CAM++ (secondary)")
    else:
        logger.info(f"Grad-CAM target layer: {target_layer.__class__.__name__}")

    # ------------------------------------------------------------------
    # 3. Build ExplainabilityEngine
    # ------------------------------------------------------------------
    engine = ExplainabilityEngine(
        model=model,
        class_names=class_names,
        architecture=args.model_name,
        target_layer=target_layer,
        device=str(device),
        img_size=args.image_size,
        dataset_mean=norm_stats["mean"],
        dataset_std=norm_stats["std"],
    )

    # ------------------------------------------------------------------
    # 4. Select stratified test samples
    # ------------------------------------------------------------------
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy()
    logger.info(f"Test set: {len(test_df)} samples")

    selected_indices = []
    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in test_df.columns:
            continue
        positive = test_df[test_df[cls_name] == 1]
        n = min(args.num_samples, len(positive))
        if n > 0:
            sampled = positive.sample(n=n, random_state=args.seed)
            selected_indices.extend(sampled.index.tolist())
            logger.info(f"  {cls_name}: selected {n} positive samples")

    # Deduplicate (multilabel images may appear in multiple classes)
    selected_indices = list(dict.fromkeys(selected_indices))
    selected_df = test_df.loc[selected_indices]
    logger.info(f"Total unique samples for XAI: {len(selected_df)}")

    # ------------------------------------------------------------------
    # 5. Generate heatmaps
    # ------------------------------------------------------------------
    for row_idx, (_, row) in enumerate(selected_df.iterrows()):
        filepath = row["filepath_processed"]
        img_path = Path(filepath)
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}, skipping")
            continue

        raw_image = load_raw_image(str(img_path), args.image_size)

        # Determine which classes are positive for this image
        positive_classes = [
            (i, name) for i, name in enumerate(class_names)
            if name in row and row[name] == 1
        ]
        if not positive_classes:
            positive_classes = [(0, class_names[0])]  # Fallback

        # Get model prediction probabilities
        input_tensor = engine._preprocess(raw_image)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()

        for cls_idx, cls_name in positive_classes:
            prob = float(probs[cls_idx])
            logger.info(
                f"[{row_idx+1}/{len(selected_df)}] {cls_name} "
                f"(GT=positive, pred={prob:.3f}) — {img_path.name}"
            )

            if is_vit:
                # ViT: Attention Rollout (primary) + Grad-CAM++ (secondary)
                primary_result = engine.generate_heatmap(
                    raw_image, target_class=cls_idx, method="attention_rollout",
                )
                secondary_result = engine.generate_heatmap(
                    raw_image, target_class=cls_idx, method="gradcam++",
                )
                primary_label = "Attention Rollout"
                secondary_label = "Grad-CAM++"
                qc_result = primary_result
            else:
                # CNN / Swin: Grad-CAM++ (primary) + Score-CAM (secondary)
                primary_result = engine.generate_heatmap(
                    raw_image, target_class=cls_idx, method="gradcam++",
                )
                try:
                    secondary_result = engine.generate_heatmap(
                        raw_image, target_class=cls_idx, method="scorecam",
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    logger.warning(f"  Score-CAM skipped (OOM at {args.image_size}px): {e}")
                    torch.cuda.empty_cache()
                    secondary_result = primary_result  # fallback
                primary_label = "Grad-CAM++"
                secondary_label = "Score-CAM"
                qc_result = primary_result

            title = (
                f"{cls_name} | GT=positive | Prob={prob:.3f} | "
                f"QC={'PASS' if qc_result['quality'].passes_qc else 'FAIL'}"
            )

            fname = f"{img_path.stem}_{cls_name.replace(' ', '_')}.png"
            save_comparison(
                original=raw_image,
                gradcam_overlay=primary_result["heatmap"],
                scorecam_overlay=secondary_result["heatmap"],
                title=title,
                output_path=output_dir / fname,
                primary_label=primary_label,
                secondary_label=secondary_label,
            )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    n_saved = len(list(output_dir.glob("*.png")))
    logger.info(f"Done — {n_saved} heatmap comparisons saved to {output_dir}/")


if __name__ == "__main__":
    main()
