"""Explainability (XAI) module for generating clinical-grade heatmaps.

Provides Grad-CAM++ visualizations, anatomical region scoring, and a
``ValidationXAI`` harness consumed by training callbacks.

No external ``pytorch-grad-cam`` dependency — the Grad-CAM++ algorithm is
implemented from scratch so the package stays lightweight in production.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xclinvision.config import get_class_names, get_clinical_rules

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Radiological Mappings
# ---------------------------------------------------------------------------

DEFAULT_CLASS_NAMES: List[str] = get_class_names()

LUNG_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    # ---------------------------------------------------------------
    # Mutually-exclusive lateral lung zones (relative x,y in [0,1]).
    # Patient RIGHT lung is on LEFT of image (x: 0.00–0.45)
    # Patient LEFT lung is on RIGHT of image (x: 0.55–1.00)
    # A deliberate 0.10-wide central corridor (x: 0.45–0.55) is left
    # unassigned to avoid double-counting mediastinal structures.
    # ---------------------------------------------------------------
    "right_upper":  (0.00, 0.00, 0.45, 0.33),
    "right_middle": (0.00, 0.33, 0.45, 0.66),
    "right_lower":  (0.00, 0.66, 0.45, 1.00),
    "left_upper":   (0.55, 0.00, 1.00, 0.33),
    "left_middle":  (0.55, 0.33, 1.00, 0.66),
    "left_lower":   (0.55, 0.66, 1.00, 1.00),
    # ---------------------------------------------------------------
    # Anatomical marker regions – intentionally overlap lateral zones
    # because these structures are physically located in those areas.
    # Treat scores here as supplementary, not exclusive.
    # ---------------------------------------------------------------
    # Hilar: para-mediastinal, strictly central-middle strip
    "hilar":   (0.35, 0.33, 0.65, 0.60),
    # Cardiac: lower-central, slightly above the diaphragm
    "cardiac": (0.38, 0.45, 0.62, 0.72),
    # Apical: uppermost dome — narrowed to reduce overlap with upper zones
    "apical":  (0.20, 0.00, 0.80, 0.20),
}

#: ImageNet statistics — used for normalization.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================================
# Grad-CAM++ (standalone implementation)
# =========================================================================

class GradCAMPlusPlus:
    """Grad-CAM++: Generalized Gradient-based Visual Explanations."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._bwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _fwd(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def _bwd(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            self.gradients = grad_output[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(_fwd)
        self._bwd_handle = self.target_layer.register_full_backward_hook(_bwd)

    def remove_hooks(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        with torch.set_grad_enabled(True):
            self.model.eval()
            self.model.zero_grad()

            # Input MUST require grad to build computation graph
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            
            # retain_graph=False prevents severe memory leaks
            output.backward(gradient=one_hot, retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM++ hooks did not fire. Verify that target_layer is part "
                "of the computation graph for the given input."
            )

        grads = self.gradients[0]
        acts = self.activations[0]

        # Transformers (Swin, ViT) produce (num_patches, C) — reshape to (C, H, W)
        if acts.ndim == 2:
            num_patches, C = acts.shape

            # Check if there is a CLS token (perfect square + 1)
            if int((num_patches - 1) ** 0.5) ** 2 == (num_patches - 1):
                acts = acts[1:, :] # Strip CLS token
                grads = grads[1:, :]
                num_patches -= 1

            # L-3 fix: handle non-square patch grids (e.g. Swin with non-power-of-2
            # image sizes) instead of blindly assuming a perfect square.
            H_sq = int(num_patches ** 0.5)
            if H_sq * H_sq == num_patches:
                H = W = H_sq
            else:
                # Find the largest factor <= sqrt(num_patches) for the most
                # square-like grid, then set W = num_patches // H.
                H, W = 1, num_patches
                for f in range(H_sq, 0, -1):
                    if num_patches % f == 0:
                        H, W = f, num_patches // f
                        break
            if H * W != num_patches:
                raise RuntimeError(
                    f"Cannot reshape transformer activations {acts.shape} into any "
                    f"rectangular spatial grid. num_patches={num_patches}."
                )
            acts = acts.permute(1, 0).reshape(C, H, W)
            grads = grads.permute(1, 0).reshape(C, H, W)
        elif acts.ndim != 3:
            raise RuntimeError(
                f"Expected 2D spatial feature map (C, H, W), got {acts.shape}. "
                "Ensure target_layer points to a Conv2d or reshaped Swin layer."
            )

        grad_2 = grads ** 2
        grad_3 = grads ** 3

        alpha_denom = 2.0 * grad_2 + (acts * grad_3).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0.0,
            alpha_denom,
            torch.ones_like(alpha_denom),
        )
        alphas = grad_2 / (alpha_denom + 1e-8)
        weights = (alphas * F.relu(grads)).sum(dim=(1, 2))

        cam = F.relu((weights.view(-1, 1, 1) * acts).sum(dim=0))

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().numpy(), target_class


# =========================================================================
# Overlay & Region Scoring
# =========================================================================

def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            base = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            base = np.clip(image, 0, 255).astype(np.uint8)
    else:
        base = image.copy()

    if base.ndim == 3 and base.shape[2] == 3:
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    if heatmap.shape[:2] != base.shape[:2]:
        heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))

    coloured = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), colormap)
    blended = cv2.addWeighted(base, 1.0 - alpha, coloured, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def score_lung_regions(
    cam: np.ndarray,
    regions: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Dict[str, float]:
    regions = regions or LUNG_REGIONS
    h, w = cam.shape[:2]
    scores: Dict[str, float] = {}

    for name, (x1, y1, x2, y2) in regions.items():
        r = cam[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        scores[name] = float(r.mean()) if r.size > 0 else 0.0

    return scores


# =========================================================================
# Clinical findings & Plausibility
# =========================================================================

def extract_findings(
    region_scores: Dict[str, float],
    class_name: str,
    confidence: float,
    *,
    high_threshold: float = 0.30,
    center_threshold: float = 0.35,
) -> List[str]:
    """Extract human-readable findings from region activation scores.

    Fully disease-agnostic: all logic is based on region activation patterns
    and laterality, not on specific disease names.
    """
    findings: List[str] = []
    if not region_scores:
        return findings

    top_region = max(region_scores, key=region_scores.get)
    top_score = region_scores[top_region]
    if top_score > high_threshold:
        findings.append(
            f"Highest activation in {top_region.replace('_', ' ')} "
            f"(score {top_score:.2f})"
        )

    for key in ("hilar", "cardiac"):
        if region_scores.get(key, 0.0) > center_threshold:
            findings.append(f"Notable {key} activation ({region_scores[key]:.2f})")

    left_sum = sum(region_scores.get(f"left_{z}", 0.0) for z in ("upper", "middle", "lower"))
    right_sum = sum(region_scores.get(f"right_{z}", 0.0) for z in ("upper", "middle", "lower"))
    total = left_sum + right_sum

    if total > 0.3:
        ratio = left_sum / (right_sum + 1e-8)
        if 0.6 < ratio < 1.7:
            findings.append("Bilateral distribution pattern")
        elif ratio >= 1.7:
            findings.append("Left-sided predominance")
        else:
            findings.append("Right-sided predominance")

    apical = region_scores.get("apical", 0.0)
    basal = (region_scores.get("left_lower", 0.0) + region_scores.get("right_lower", 0.0)) / 2.0

    if apical > basal * 1.5 and apical > high_threshold:
        findings.append("Apical predominance")
    elif basal > apical * 1.5 and basal > high_threshold:
        findings.append("Basal predominance")

    if confidence < 0.5:
        findings.append(f"Low confidence ({confidence:.0%}) for {class_name} — review recommended")

    return findings


def clinical_plausibility_score(
    region_scores: Dict[str, float],
    class_name: str,
) -> float:
    """Score how anatomically plausible the Grad-CAM activation is.

    Disease-specific logic is driven by ``clinical_rules`` in
    ``configs/system.yaml``.  If no rule exists for a class, a generic
    activation-spread heuristic is used — so the function works for
    **any** disease set without code changes.
    """
    if not region_scores:
        return 0.0

    lung_scores = [v for k, v in region_scores.items() if k != "cardiac"]
    mean_lung = float(np.mean(lung_scores)) if lung_scores else 0.0
    activation_range = max(region_scores.values()) - min(region_scores.values())

    score = 0.5
    rules = get_clinical_rules()
    rule = rules.get(class_name.lower(), {})

    if rule.get("normal_class"):
        # Normal / healthy class: low activation = plausible
        if mean_lung < 0.15:
            score += 0.3
        elif mean_lung < 0.25:
            score += 0.1
    elif rule.get("expected_regions"):
        # Disease localised to specific lung regions
        region_keys = rule["expected_regions"]
        region_max = max(
            (region_scores.get(k, 0.0) for k in region_keys), default=0.0
        )
        if region_max > 0.2:
            score += 0.3
    elif rule.get("expected_region_key"):
        # Disease localised to a single anatomical region (e.g. cardiac)
        key_val = region_scores.get(rule["expected_region_key"], 0.0)
        if key_val > 0.3:
            score += 0.3
        elif key_val > 0.15:
            score += 0.15
    else:
        # Generic fallback: reward any focused activation
        if mean_lung > 0.15:
            score += 0.15

    if activation_range > 0.1:
        score += 0.15

    return float(np.clip(score, 0.0, 1.0))


# =========================================================================
# CAM Quality Control
# =========================================================================

@dataclass
class CAMQualityReport:
    total_activation: float = 0.0
    max_activation: float = 0.0
    coverage: float = 0.0  
    is_degenerate: bool = False  
    lung_focus_ratio: float = 0.0  

    @property
    def passes_qc(self) -> bool:
        return (not self.is_degenerate) and (self.lung_focus_ratio > 0.20)

def assess_cam_quality(
    cam: np.ndarray,
    threshold: float = 0.15,
    degenerate_low: float = 0.005,
    degenerate_high: float = 0.995,
) -> CAMQualityReport:
    total = float(cam.mean())
    mx = float(cam.max())
    pixels_above = float((cam > threshold).mean())
    is_deg = total < degenerate_low or total > degenerate_high

    h, w = cam.shape[:2]
    lung_mask = np.zeros((h, w), dtype=np.float32)
    for name, (x1, y1, x2, y2) in LUNG_REGIONS.items():
        if name == "cardiac":
            continue 
        lung_mask[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 1.0
    lung_mask = np.clip(lung_mask, 0, 1)

    lung_act = float((cam * lung_mask).sum())
    total_act = float(cam.sum()) + 1e-8
    lung_ratio = lung_act / total_act

    return CAMQualityReport(
        total_activation=total,
        max_activation=mx,
        coverage=pixels_above,
        is_degenerate=is_deg,
        lung_focus_ratio=lung_ratio,
    )


# =========================================================================
# ExplainabilityEngine
# =========================================================================

class ExplainabilityEngine:
    """Generate clinical explanations for model predictions."""

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str] = DEFAULT_CLASS_NAMES,
        architecture: str = "unknown",
        target_layer: Optional[nn.Module] = None,
        device: str = "cpu",
        img_size: int = 384,
        dataset_mean: Optional[np.ndarray] = None,
        dataset_std: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.architecture = architecture
        self.device = device
        self.img_size = img_size
        self.dataset_mean = dataset_mean if dataset_mean is not None else IMAGENET_MEAN
        self.dataset_std = dataset_std if dataset_std is not None else IMAGENET_STD
        self._target_layer = target_layer or self._resolve_target_layer()

        if self._target_layer is None:
            logger.warning("Could not auto-detect Grad-CAM target layer.")

    def _resolve_target_layer(self) -> Optional[nn.Module]:
        if hasattr(self.model, "get_gradcam_target"):
            return self.model.get_gradcam_target()

        last_conv: Optional[nn.Module] = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = image.copy()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Ensure 3-channel RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=-1)

        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        normalised = (img - self.dataset_mean) / self.dataset_std
        tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def generate_heatmap(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        original_image: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate heatmap. Accepts preprocessed Tensor or raw Numpy array."""
        if self._target_layer is None:
            logger.error("No target layer found. Returning empty heatmap.")
            h, w = self.img_size, self.img_size
            if original_image is not None:
                h, w = original_image.shape[:2]
            elif isinstance(input_data, np.ndarray):
                h, w = input_data.shape[:2]

            empty_cam = np.zeros((h, w), dtype=np.float32)
            
            vis_image = original_image.copy() if original_image is not None else (input_data.copy() if isinstance(input_data, np.ndarray) else np.zeros((h, w, 3), dtype=np.uint8))
            if vis_image.ndim == 2:
                vis_image = np.stack([vis_image, vis_image, vis_image], axis=-1)
            elif vis_image.ndim == 3 and vis_image.shape[2] == 1:
                vis_image = np.concatenate([vis_image, vis_image, vis_image], axis=-1)

            if vis_image.dtype == np.uint8:
                vis_image = vis_image.astype(np.float32) / 255.0

            overlay = create_overlay(vis_image, empty_cam, alpha=alpha)
            return {
                "heatmap": overlay,
                "grayscale_cam": empty_cam,
                "region_scores": score_lung_regions(empty_cam),
                "quality": assess_cam_quality(empty_cam),
                "target_class": target_class if target_class is not None else 0,
                "method": "gradcam++ (fallback)",
            }

        # Handle double forward-pass prevention
        if isinstance(input_data, torch.Tensor):
            if original_image is None:
                raise ValueError("original_image required if passing a Tensor.")
            input_tensor = input_data
            vis_image = original_image.copy()
        else:
            input_tensor = self._preprocess(input_data)
            vis_image = input_data.copy()
            # Ensure vis_image is 3-channel so create_overlay never receives a 2D array
            if vis_image.ndim == 2:
                vis_image = np.stack([vis_image, vis_image, vis_image], axis=-1)
            elif vis_image.ndim == 3 and vis_image.shape[2] == 1:
                vis_image = np.concatenate([vis_image, vis_image, vis_image], axis=-1)

        cam_gen = GradCAMPlusPlus(self.model, self._target_layer)
        try:
            grayscale_cam, used_class = cam_gen.generate(input_tensor, target_class=target_class)
        finally:
            cam_gen.remove_hooks()

        if vis_image.dtype == np.uint8:
            vis_image = vis_image.astype(np.float32) / 255.0
        if vis_image.shape[:2] != grayscale_cam.shape[:2]:
            vis_image = cv2.resize(vis_image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

        overlay = create_overlay(vis_image, grayscale_cam, alpha=alpha)
        region_scores = score_lung_regions(grayscale_cam)
        quality = assess_cam_quality(grayscale_cam)

        return {
            "heatmap": overlay,
            "grayscale_cam": grayscale_cam,
            "region_scores": region_scores,
            "quality": quality,
            "target_class": used_class,
            "method": "gradcam++",
        }

    def explain_prediction(
        self,
        image: np.ndarray,
        prediction: int,
        confidence: float,
    ) -> Dict[str, Any]:
        heatmap_result = self.generate_heatmap(image, target_class=prediction)
        
        class_name = (
            self.class_names[prediction] 
            if prediction < len(self.class_names) 
            else f"class_{prediction}"
        )

        findings = extract_findings(
            heatmap_result["region_scores"],
            class_name=class_name,
            confidence=confidence,
        )

        plausibility = clinical_plausibility_score(
            heatmap_result["region_scores"],
            class_name=class_name,
        )

        return {
            "prediction": prediction,
            "class_name": class_name,
            "confidence": confidence,
            "visualization": heatmap_result,
            "key_findings": findings,
            "clinical_plausibility": plausibility,
            "quality": heatmap_result["quality"],
        }


# =========================================================================
# ValidationXAI — trainer integration
# =========================================================================

class ValidationXAI:
    """Batch XAI validation for training-loop integration."""

    def __init__(
        self,
        model: nn.Module,
        architecture: str,
        output_dir: Union[str, Path],
        class_names: List[str] = DEFAULT_CLASS_NAMES,
        img_size: int = 384,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.img_size = img_size

        if device is None:
            device = str(next(model.parameters()).device)
        self.device = device

        self.engine = ExplainabilityEngine(
            model,
            class_names=class_names,
            architecture=architecture,
            device=self.device,
            img_size=img_size,
        )

    def process_dataset(
        self,
        dataloader: Any,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        self.model.eval()
        processed = 0
        plausibility_scores: List[float] = []
        qc_failures = 0
        per_class: Dict[int, List[float]] = defaultdict(list)
        region_accum: Dict[str, List[float]] = defaultdict(list)

        for batch in dataloader:
            if processed >= max_samples:
                break

            images, labels = batch[0], batch[1]

            for i in range(images.size(0)):
                if processed >= max_samples:
                    break

                try:
                    img_np = self._tensor_to_numpy(images[i])
                    label = int(labels[i].item())
                    tensor_batch = images[i : i + 1].to(self.device)

                    result = self.engine.generate_heatmap(
                        input_data=tensor_batch,
                        original_image=img_np,
                        target_class=label,
                    )
                    
                    region_scores = result["region_scores"]
                    quality: CAMQualityReport = result["quality"]
                    class_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"

                    plaus = clinical_plausibility_score(region_scores, class_name)
                    plausibility_scores.append(plaus)
                    per_class[label].append(plaus)

                    if not quality.passes_qc:
                        qc_failures += 1

                    for rname, rscore in region_scores.items():
                        region_accum[rname].append(rscore)

                except Exception as exc:
                    logger.warning("XAI failed for sample %d: %s", processed, exc)
                    qc_failures += 1

                processed += 1

        mean_plaus = float(np.mean(plausibility_scores)) if plausibility_scores else 0.0
        qc_rate = qc_failures / max(processed, 1)
        region_summary = {k: float(np.mean(v)) for k, v in region_accum.items()}

        per_class_summary = {}
        for cls_idx, scores in per_class.items():
            cname = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"
            per_class_summary[cname] = {
                "mean_plausibility": float(np.mean(scores)),
                "n_samples": len(scores),
            }

        metrics = {
            "n_samples": processed,
            "clinical_plausibility": {
                "score": mean_plaus,
                "is_acceptable": mean_plaus >= 0.65,
            },
            "qc_failure_rate": qc_rate,
            "region_activation_summary": region_summary,
            "per_class_stats": per_class_summary,
        }

        report_path = self.output_dir / "xai_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not save XAI report: %s", exc)

        return metrics

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img


# =========================================================================
# Public convenience function 
# =========================================================================

def generate_explanation(
    model: nn.Module,
    image: np.ndarray,
    prediction: int,
    confidence: float,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
    architecture: str = "unknown",
    device: str = "cpu",
    img_size: int = 384,
) -> Dict[str, Any]:
    engine = ExplainabilityEngine(
        model,
        class_names=class_names,
        architecture=architecture,
        device=device,
        img_size=img_size,
    )
    return engine.explain_prediction(image, prediction, confidence)
