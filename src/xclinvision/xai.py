"""Explainability (XAI) module for generating clinical-grade heatmaps.

Provides three visualisation methods:

- **Grad-CAM++** and **Score-CAM** for CNN / Swin architectures.
- **Attention Rollout** (Abnar & Zuidema, 2020) for Vision Transformers.

Also includes anatomical region scoring, clinical plausibility checks,
and a ``ValidationXAI`` harness consumed by training callbacks.

No external ``pytorch-grad-cam`` dependency — all methods are implemented
from scratch so the package stays lightweight in production.
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

#: Fallback standard ImageNet statistics.
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_hooks()
        return False

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        with torch.set_grad_enabled(True):
            self.model.eval()
            self.model.zero_grad()
            
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

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
# Score-CAM (gradient-free implementation)
# =========================================================================

class ScoreCAM:
    """Score-CAM: gradient-free class activation mapping.

    Uses forward-pass-only perturbation scoring instead of backpropagation,
    making it architecture-agnostic and free of gradient noise.  A top-k
    channel selection and batched forward pass keep it production-viable.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        top_k: int = 32,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.top_k = top_k
        self.activations: Optional[torch.Tensor] = None
        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _fwd(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        self._fwd_handle = self.target_layer.register_forward_hook(_fwd)

    def remove_hooks(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_hooks()
        return False

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        self.model.eval()
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # 1. Forward pass to capture activations and baseline output
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        if self.activations is None:
            raise RuntimeError(
                "Score-CAM hook did not fire. Verify that target_layer is part "
                "of the computation graph for the given input."
            )

        acts = self.activations[0]  # (C, h, w) or (num_patches, C)

        # Handle transformer-style (num_patches, C) activations
        if acts.ndim == 2:
            num_patches, C = acts.shape
            if int((num_patches - 1) ** 0.5) ** 2 == (num_patches - 1):
                acts = acts[1:, :]
                num_patches -= 1
            H_sq = int(num_patches ** 0.5)
            if H_sq * H_sq == num_patches:
                H = W = H_sq
            else:
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
        elif acts.ndim != 3:
            raise RuntimeError(
                f"Expected (C, H, W) feature map, got {acts.shape}."
            )

        C, h_act, w_act = acts.shape
        input_h, input_w = input_tensor.shape[2:]

        # 2. Select top-k channels by mean activation
        k = min(self.top_k, C)
        mean_per_channel = acts.mean(dim=(1, 2))  # (C,)
        topk_indices = mean_per_channel.topk(k).indices  # (k,)
        selected_acts = acts[topk_indices]  # (k, h_act, w_act)

        # 3. Upsample each activation map to input size and normalize to [0, 1]
        upsampled = F.interpolate(
            selected_acts.unsqueeze(1),  # (k, 1, h_act, w_act)
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (k, input_h, input_w)

        # Per-map min-max normalization
        mins = upsampled.flatten(1).min(dim=1).values.view(k, 1, 1)
        maxs = upsampled.flatten(1).max(dim=1).values.view(k, 1, 1)
        upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

        # 4. Create masked inputs and run batched forward pass
        # input_tensor is (1, C_in, H, W) — broadcast multiply
        masked_inputs = input_tensor * upsampled.unsqueeze(1)  # (k, C_in, H, W)

        scores = self.model(masked_inputs)  # (k, num_classes)

        # 5. Extract target class scores and apply relu -> normalize
        target_scores = scores[:, target_class]  # (k,)
        target_scores = F.relu(target_scores)
        weights = target_scores / (target_scores.sum() + 1e-8)  # L1 normalize

        # 6. Weighted sum of the selected activation maps
        cam = (weights.view(k, 1, 1) * selected_acts).sum(dim=0)  # (h_act, w_act)

        # 7. ReLU and normalize
        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # 8. Resize to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().numpy(), target_class


# =========================================================================
# Attention Rollout (Abnar & Zuidema, 2020) — for Vision Transformers
# =========================================================================

class AttentionRollout:
    """Attention Rollout for Vision Transformers.

    Aggregates multi-head self-attention across all transformer layers to
    produce a single spatial map showing where the [CLS] token attends.
    This is the standard XAI method for ViTs — Grad-CAM++ was designed for
    convolutional architectures and produces noisy, less meaningful results
    when applied to linear (QKV) layers in transformers.

    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", 2020.
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> None:
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._blocks = self._find_blocks()

    def _find_blocks(self) -> nn.Module:
        """Locate the ``blocks`` sequential container in a ViT model."""
        if hasattr(self.model, "blocks"):
            return self.model.blocks
        for name, module in self.model.named_modules():
            if name == "blocks" and hasattr(module, "__len__"):
                return module
        raise RuntimeError(
            "Cannot find transformer blocks. "
            "AttentionRollout requires a ViT-style model with a 'blocks' attribute."
        )

    def remove_hooks(self) -> None:
        """No persistent hooks — attention weights are captured per generate() call."""

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate an attention rollout map.

        Returns ``(cam, target_class)`` matching the Grad-CAM++ / Score-CAM API
        so the caller can use any method interchangeably.
        """
        self.model.eval()
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # --- Collect attention weights via one-shot hooks ---
        attentions: List[torch.Tensor] = []
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        for block in self._blocks:
            attn_mod = block.attn

            def _make_hook(am: nn.Module):
                def _hook(_module: nn.Module, inp: Any, _out: Any) -> None:
                    x = inp[0]                          # (B, N, C)
                    B, N, C = x.shape
                    head_dim = getattr(am, "head_dim", C // am.num_heads)
                    qkv = am.qkv(x).reshape(B, N, 3, am.num_heads, head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
                    q, k = qkv[0], qkv[1]
                    scale = head_dim ** -0.5
                    attn_weights = (q @ k.transpose(-2, -1)) * scale
                    attn_weights = attn_weights.softmax(dim=-1)
                    attentions.append(attn_weights)
                return _hook

            hooks.append(attn_mod.register_forward_hook(_make_hook(attn_mod)))

        output = self.model(input_tensor)

        for h in hooks:
            h.remove()

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        if not attentions:
            raise RuntimeError("No attention weights captured. Is this a ViT model?")

        # --- Rollout: multiply fused attention matrices across layers ---
        result = None
        for attn in attentions:
            # attn: (B, num_heads, N, N)
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {self.head_fusion}")

            # Discard low-attention values for cleaner maps
            if self.discard_ratio > 0:
                flat = attn_fused.view(attn_fused.size(0), -1)
                thresh = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True)
                attn_fused = attn_fused * (attn_fused > thresh.unsqueeze(-1)).float()
                attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)

            # Add residual connection (identity) and re-normalise
            I = torch.eye(attn_fused.size(-1), device=attn_fused.device).unsqueeze(0)
            attn_fused = 0.5 * attn_fused + 0.5 * I
            attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)

            result = attn_fused if result is None else (result @ attn_fused)

        # --- Extract [CLS] → patch attention and reshape to spatial grid ---
        mask = result[0, 0, 1:]           # skip CLS self-attention
        num_patches = mask.shape[0]
        H = W = int(num_patches ** 0.5)
        if H * W != num_patches:
            H_sq = int(num_patches ** 0.5)
            for f in range(H_sq, 0, -1):
                if num_patches % f == 0:
                    H, W = f, num_patches // f
                    break
        if H * W != num_patches:
            raise RuntimeError(
                f"Cannot reshape {num_patches} patches into a rectangular grid."
            )

        mask = mask.reshape(H, W)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        cam = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
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
    roi_focus_ratio: float = 0.0  

    @property
    def passes_qc(self) -> bool:
        return (not self.is_degenerate) and (self.roi_focus_ratio > 0.20)

def assess_cam_quality(
    cam: np.ndarray,
    threshold: float = 0.15,
    degenerate_low: float = 0.005,
    degenerate_high: float = 0.995,
    expected_regions: Optional[List[str]] = None,
) -> CAMQualityReport:
    """Assess heatmap quality.
    
    Dynamically constructs an expected Region of Interest (ROI) mask. If the 
    underlying clinical rule defines expected regions (e.g., cardiac), QC evaluates
    focus within that targeted anatomy rather than general lung space.
    """
    total = float(cam.mean())
    mx = float(cam.max())
    pixels_above = float((cam > threshold).mean())
    is_deg = total < degenerate_low or total > degenerate_high

    h, w = cam.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.float32)
    
    # Default to lung fields if no specific rule specifies otherwise
    target_zones = expected_regions if expected_regions else [k for k in LUNG_REGIONS.keys() if k != "cardiac"]
    
    for name in target_zones:
        if name in LUNG_REGIONS:
            x1, y1, x2, y2 = LUNG_REGIONS[name]
            roi_mask[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 1.0
            
    roi_mask = np.clip(roi_mask, 0, 1)

    roi_act = float((cam * roi_mask).sum())
    total_act = float(cam.sum()) + 1e-8
    roi_ratio = roi_act / total_act

    return CAMQualityReport(
        total_activation=total,
        max_activation=mx,
        coverage=pixels_above,
        is_degenerate=is_deg,
        roi_focus_ratio=roi_ratio,
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
        dataset_mean: Optional[Union[np.ndarray, List[float]]] = None,
        dataset_std: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.architecture = architecture
        self.device = device
        self.img_size = img_size
        self.dataset_mean = np.array(dataset_mean, dtype=np.float32) if dataset_mean is not None else IMAGENET_MEAN
        self.dataset_std = np.array(dataset_std, dtype=np.float32) if dataset_std is not None else IMAGENET_STD
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
        method: str = "gradcam++",
    ) -> Dict[str, Any]:
        """Generate heatmap. Accepts preprocessed Tensor or raw Numpy array.

        Args:
            method: ``"gradcam++"`` (default), ``"scorecam"``, or
                ``"attention_rollout"`` (recommended for ViT models).
        """
        
        # Look up expected anatomical regions based on the target class' clinical rules
        class_name = ""
        expected_regions = None
        if target_class is not None and target_class < len(self.class_names):
            class_name = self.class_names[target_class]
            rules = get_clinical_rules().get(class_name.lower(), {})
            expected_regions = rules.get("expected_regions") or (
                [rules["expected_region_key"]] if rules.get("expected_region_key") else None
            )

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
                "quality": assess_cam_quality(empty_cam, expected_regions=expected_regions),
                "target_class": target_class if target_class is not None else 0,
                "method": f"{method} (fallback)",
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

        if method == "attention_rollout":
            cam_gen = AttentionRollout(self.model)
        elif method == "scorecam":
            cam_gen = ScoreCAM(self.model, self._target_layer)
        else:
            cam_gen = GradCAMPlusPlus(self.model, self._target_layer)
        try:
            grayscale_cam, used_class = cam_gen.generate(input_tensor, target_class=target_class)
        finally:
            cam_gen.remove_hooks()
            del cam_gen  # Aggressive memory cleanup

        if vis_image.dtype == np.uint8:
            vis_image = vis_image.astype(np.float32) / 255.0
        if vis_image.shape[:2] != grayscale_cam.shape[:2]:
            vis_image = cv2.resize(vis_image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

        overlay = create_overlay(vis_image, grayscale_cam, alpha=alpha)
        region_scores = score_lung_regions(grayscale_cam)
        quality = assess_cam_quality(grayscale_cam, expected_regions=expected_regions)

        return {
            "heatmap": overlay,
            "grayscale_cam": grayscale_cam,
            "region_scores": region_scores,
            "quality": quality,
            "target_class": used_class,
            "method": method,
        }

    def explain_prediction(
        self,
        image: np.ndarray,
        prediction: int,
        confidence: float,
        method: str = "gradcam++",
    ) -> Dict[str, Any]:
        heatmap_result = self.generate_heatmap(image, target_class=prediction, method=method)
        
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

        result: Dict[str, Any] = {
            "prediction": prediction,
            "class_name": class_name,
            "confidence": confidence,
            "visualization": heatmap_result,
            "key_findings": findings,
            "clinical_plausibility": plausibility,
            "quality": heatmap_result["quality"],
        }

        # Conditional Score-CAM for low-confidence / low-quality predictions
        if (
            confidence < 0.6
            or not heatmap_result["quality"].passes_qc
            or plausibility < 0.6
        ):
            try:
                scorecam_result = self.generate_heatmap(
                    image, target_class=prediction, method="scorecam",
                )
                result["scorecam_visualization"] = scorecam_result
            except Exception as exc:
                logger.warning("Score-CAM fallback failed: %s", exc)

        return result

    def explain_multilabel_prediction(
        self,
        image: np.ndarray,
        positive_indices: List[int],
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate heatmaps for ALL positive labels in a single multilabel image.

        Args:
            image: Raw input image (H, W) or (H, W, C).
            positive_indices: List of class indices predicted positive.
            probabilities: Full probability vector, shape (num_classes,).

        Returns:
            Dict with per-class explanations under ``"per_class"`` key.
        """
        per_class: Dict[str, Dict[str, Any]] = {}

        for idx in positive_indices:
            class_name = (
                self.class_names[idx]
                if idx < len(self.class_names)
                else f"class_{idx}"
            )
            confidence = float(probabilities[idx])

            heatmap_result = self.generate_heatmap(image, target_class=idx)
            findings = extract_findings(
                heatmap_result["region_scores"],
                class_name=class_name,
                confidence=confidence,
            )
            plausibility = clinical_plausibility_score(
                heatmap_result["region_scores"],
                class_name=class_name,
            )
            entry: Dict[str, Any] = {
                "class_index": idx,
                "confidence": confidence,
                "visualization": heatmap_result,
                "key_findings": findings,
                "clinical_plausibility": plausibility,
                "quality": heatmap_result["quality"],
            }

            # Conditional Score-CAM for uncertain / low-quality per-class results
            if (
                confidence < 0.6
                or not heatmap_result["quality"].passes_qc
                or plausibility < 0.6
            ):
                try:
                    scorecam_result = self.generate_heatmap(
                        image, target_class=idx, method="scorecam",
                    )
                    entry["scorecam_visualization"] = scorecam_result
                except Exception as exc:
                    logger.warning("Score-CAM fallback failed for class %s: %s", class_name, exc)

            per_class[class_name] = entry

        return {
            "positive_classes": [
                self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                for i in positive_indices
            ],
            "per_class": per_class,
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
        dataset_mean: Optional[Union[np.ndarray, List[float]]] = None,
        dataset_std: Optional[Union[np.ndarray, List[float]]] = None,
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
            dataset_mean=dataset_mean,
            dataset_std=dataset_std,
        )

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalizes a tensor dynamically based on the exact normalization stats."""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        # Correctly inverse the (x - mean) / std operation
        img = (img * self.engine.dataset_std) + self.engine.dataset_mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def process_dataset(
        self,
        dataloader: Any,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        from xclinvision.config import is_multilabel

        _multilabel = is_multilabel()
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
                    tensor_batch = images[i : i + 1].to(self.device)

                    if _multilabel:
                        # labels[i] is a binary vector — iterate over all positive labels
                        label_vec = labels[i]
                        positive_indices = (label_vec > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
                        if isinstance(positive_indices, int):
                            positive_indices = [positive_indices]
                        if not positive_indices:
                            # Safe fallback to the first class
                            positive_indices = [0]
                    else:
                        positive_indices = [int(labels[i].item())]

                    for label_idx in positive_indices:
                        result = self.engine.generate_heatmap(
                            input_data=tensor_batch,
                            original_image=img_np,
                            target_class=label_idx,
                        )

                        region_scores = result["region_scores"]
                        quality: CAMQualityReport = result["quality"]
                        class_name = self.class_names[label_idx] if label_idx < len(self.class_names) else f"class_{label_idx}"

                        plaus = clinical_plausibility_score(region_scores, class_name)
                        plausibility_scores.append(plaus)
                        per_class[label_idx].append(plaus)

                        if not quality.passes_qc:
                            qc_failures += 1

                        # Score-CAM fallback for QC failures / low plausibility
                        if not quality.passes_qc or plaus < 0.6:
                            try:
                                scorecam_result = self.engine.generate_heatmap(
                                    input_data=tensor_batch,
                                    original_image=img_np,
                                    target_class=label_idx,
                                    method="scorecam",
                                )
                                sc_quality: CAMQualityReport = scorecam_result["quality"]
                                if sc_quality.passes_qc:
                                    region_scores = scorecam_result["region_scores"]
                                    plaus = clinical_plausibility_score(region_scores, class_name)
                                    plausibility_scores[-1] = plaus
                                    per_class[label_idx][-1] = plaus
                                    if not quality.passes_qc and sc_quality.passes_qc:
                                        qc_failures -= 1
                            except Exception as sc_exc:
                                logger.warning(
                                    "Score-CAM fallback failed for sample %d class %s: %s",
                                    processed, class_name, sc_exc,
                                )

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
    img_size: int = 1024,
    dataset_mean: Optional[Union[np.ndarray, List[float]]] = None,
    dataset_std: Optional[Union[np.ndarray, List[float]]] = None,
    method: str = "gradcam++",
) -> Dict[str, Any]:
    engine = ExplainabilityEngine(
        model,
        class_names=class_names,
        architecture=architecture,
        device=device,
        img_size=img_size,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
    )
    return engine.explain_prediction(image, prediction, confidence, method=method)


