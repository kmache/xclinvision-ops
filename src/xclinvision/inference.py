"""Inference pipeline for model predictions with uncertainty estimation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from xclinvision.evaluator import TemperatureScaler
from xclinvision.xai import DEFAULT_CLASS_NAMES, generate_explanation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Return a torch.device, falling back to CPU when CUDA is unavailable."""
    if isinstance(device, torch.device):
        return device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable – falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _enable_mc_dropout(model: nn.Module) -> None:
    """Set Dropout layers to train mode to enable MC-Dropout inference."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout)):
            m.train()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """End-to-end inference pipeline for chest X-ray classification.

    Features
    --------
    - Single-image and mini-batched prediction
    - Configurable image size (must match training resolution)
    - Optional temperature-scaling for calibrated probabilities
    - MC-Dropout epistemic uncertainty estimation
    - Optional Grad-CAM / XAI explanation generation
    """

    def __init__(
        self,
        model: nn.Module,
        architecture: str = "unknown",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature_scaler: Optional[TemperatureScaler] = None,
        mc_samples: int = 10,
        image_size: int = 384,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Trained classification model (any nn.Module).
            architecture: Architecture name used to select the Grad-CAM target
                layer (e.g. "efficientnet_b2", "swin_t").
            device: Torch device string ("cuda" or "cpu").
            temperature_scaler: Fitted TemperatureScaler for calibrated probs.
            mc_samples: Number of stochastic forward passes for MC-Dropout.
            image_size: Input resolution matching the training pipeline.
            class_names: Human-readable class labels (label-index order).
        """
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.architecture = architecture
        self.temperature_scaler = temperature_scaler
        self.mc_samples = mc_samples
        self.image_size = image_size
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    # -----------------------------------------------------------------------
    # Pre-processing
    # -----------------------------------------------------------------------s
    def preprocess(
        self,
        image: Union[np.ndarray, str, Image.Image],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess a single image into a normalised tensor + raw RGB array.

        Args:
            image: A file-path string, PIL Image, or NumPy array (any channel
                count).

        Returns:
            Tuple of (tensor [1,C,H,W], vis_image [H,W,3 uint8]).
        """
        if isinstance(image, str):
            path = image
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"Cannot read image from path: {path}")
            img = np.stack([raw, raw, raw], axis=-1)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = np.asarray(image)

        # Normalise channel count to 3
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        vis_image = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # ImageNet normalisation (matches training pipeline)
        norm = vis_image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std

        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device), vis_image

    def preprocess_batch(
        self,
        images: List[Union[np.ndarray, str, Image.Image]],
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Preprocess a list of images into a batched tensor (N,C,H,W).

        Returns:
            Tuple of (batched tensor, list of vis_images).
        """
        if not images:
            return torch.empty(0, 3, self.image_size, self.image_size, device=self.device), []
        tensors, vis_images = zip(*[self.preprocess(img) for img in images])
        return torch.cat(list(tensors), dim=0), list(vis_images)

    # -----------------------------------------------------------------------
    # Core prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        image: Union[np.ndarray, str, Image.Image],
        return_uncertainty: bool = True,
        return_explanation: bool = True,
    ) -> Dict[str, Any]:
        """Run full inference on a single image.

        Args:
            image: Input image (path, PIL Image, or numpy array).
            return_uncertainty: Compute MC-Dropout epistemic uncertainty.
            return_explanation: Generate Grad-CAM / attention explanation.

        Returns:
            Dict with keys:
                prediction (int), class_name (str),
                probabilities (list[float]), confidence (float),
                class_names (list[str]),
                and optionally: uncertainty (dict), uncertainty_level (str),
                explanation (dict).
        """
        x, vis_image = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(x)

            if self.temperature_scaler is not None:
                scaled = self.temperature_scaler.scale(logits.cpu().numpy())
                logits = torch.from_numpy(scaled).to(self.device)

            probs = F.softmax(logits, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_class].item())

        result: Dict[str, Any] = {
            "prediction": pred_class,
            "class_name": self.class_names[pred_class]
            if pred_class < len(self.class_names)
            else "Unknown",
            "probabilities": probs[0].cpu().numpy().tolist(),
            "confidence": confidence,
            "class_names": self.class_names,
        }

        if return_uncertainty:
            uncertainty = self.compute_uncertainty(x)
            result["uncertainty"] = uncertainty
            result["uncertainty_level"] = self._get_uncertainty_level(uncertainty)

        if return_explanation:
            exp_res = generate_explanation(
                model=self.model,
                image=vis_image,
                prediction=pred_class,
                confidence=confidence,
                class_names=self.class_names,
                architecture=self.architecture,
                device=str(self.device),
                img_size=self.image_size,
            )
            result["explanation"] = exp_res

        return result

    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Image.Image]],
        return_uncertainty: bool = False,
        return_explanation: bool = False,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """Run inference on multiple images with GPU mini-batching.

        Forward passes are grouped into mini-batches for efficiency.
        Uncertainty (MC-Dropout) and explanations are computed per-image
        when requested.

        Args:
            images: List of input images.
            return_uncertainty: Compute per-image epistemic uncertainty.
            return_explanation: Generate per-image XAI explanation.
            batch_size: Mini-batch size for batched forward passes.

        Returns:
            List of result dicts (same schema as predict()).
        """
        results: List[Dict[str, Any]] = []

        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            x_batch, vis_images = self.preprocess_batch(chunk)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_batch)
                if self.temperature_scaler is not None:
                    scaled = self.temperature_scaler.scale(logits.cpu().numpy())
                    logits = torch.from_numpy(scaled).to(self.device)
                probs_batch = F.softmax(logits, dim=1).cpu().numpy()

            for i, (vis_image, prob) in enumerate(zip(vis_images, probs_batch)):
                pred_class = int(np.argmax(prob))
                result: Dict[str, Any] = {
                    "prediction": pred_class,
                    "class_name": self.class_names[pred_class]
                    if pred_class < len(self.class_names)
                    else "Unknown",
                    "probabilities": prob.tolist(),
                    "confidence": float(prob[pred_class]),
                    "class_names": self.class_names,
                }
                if return_uncertainty:
                    x_single = x_batch[i : i + 1]
                    unc = self.compute_uncertainty(x_single)
                    result["uncertainty"] = unc
                    result["uncertainty_level"] = self._get_uncertainty_level(unc)
                if return_explanation:
                    result["explanation"] = generate_explanation(
                        model=self.model,
                        image=vis_image,
                        prediction=pred_class,
                        confidence=float(prob[pred_class]),
                        class_names=self.class_names,
                        architecture=self.architecture,
                        device=str(self.device),
                        img_size=self.image_size,
                    )
                results.append(result)

        return results

    # -----------------------------------------------------------------------
    # Uncertainty estimation
    # -----------------------------------------------------------------------

    def enable_mc_dropout(self) -> None:
        """Enable dropout layers for MC-Dropout inference (in-place)."""
        _enable_mc_dropout(self.model)

    def compute_uncertainty(self, x: torch.Tensor) -> Dict[str, Any]:
        """Estimate epistemic uncertainty via MC Dropout.

        Runs `mc_samples` stochastic forward passes and measures variance
        across the predictive distributions.

        Args:
            x: Pre-processed 4-D tensor (1, C, H, W).

        Returns:
            Dict:
                epistemic (float): Mean variance across MC samples.
                predictive_entropy (float): Entropy of the mean distribution.
                mc_samples (int): Number of stochastic samples used.
        """
        _enable_mc_dropout(self.model)

        mc_preds: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if self.temperature_scaler is not None:
                    scaled = self.temperature_scaler.scale(logits.cpu().numpy())
                    logits = torch.from_numpy(scaled).to(self.device)
                mc_preds.append(F.softmax(logits, dim=1).cpu().numpy())

        self.model.eval()  # restore deterministic mode

        preds = np.array(mc_preds)        # (mc_samples, 1, num_classes)
        mean_pred = preds.mean(axis=0)    # (1, num_classes)

        epistemic = float(preds.var(axis=0).mean())

        eps = 1e-10
        predictive_entropy = float(-np.sum(mean_pred * np.log(mean_pred + eps), axis=-1))

        return {
            "epistemic": epistemic,
            "predictive_entropy": predictive_entropy,
            "mc_samples": self.mc_samples,
        }

    def _get_uncertainty_level(self, uncertainty: Dict[str, Any]) -> str:
        """Map epistemic variance to a human-readable level."""
        v = uncertainty["epistemic"]
        if v < 0.01:
            return "low"
        elif v < 0.05:
            return "medium"
        return "high"


# ---------------------------------------------------------------------------
# Convenience top-level function (consumed by xclinvision.__init__)
# ---------------------------------------------------------------------------

def predict(
    model: nn.Module,
    image: Union[np.ndarray, str, Image.Image],
    architecture: str = "unknown",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 384,
    temperature_scaler: Optional[TemperatureScaler] = None,
    return_uncertainty: bool = True,
    return_explanation: bool = True,
) -> Dict[str, Any]:
    """Stateless convenience wrapper around InferencePipeline.

    A new ``InferencePipeline`` is instantiated on every call, so this
    function is convenient for one-off predictions.  For repeated calls
    (e.g. inside a loop), prefer creating a single ``InferencePipeline``
    instance and reusing it.

    Args:
        model: Trained classification model.
        image: Input image (file path, PIL Image, or numpy array).
        architecture: Model architecture name for XAI target selection.
        device: Torch device string.
        img_size: Input resolution used during training.
        temperature_scaler: Optional fitted TemperatureScaler.
        return_uncertainty: Estimate epistemic uncertainty via MC Dropout.
        return_explanation: Generate Grad-CAM / attention explanation.

    Returns:
        Prediction dictionary (same schema as InferencePipeline.predict()).
    """
    pipeline = InferencePipeline(
        model=model,
        architecture=architecture,
        device=device,
        temperature_scaler=temperature_scaler,
        image_size=img_size,
    )
    return pipeline.predict(
        image,
        return_uncertainty=return_uncertainty,
        return_explanation=return_explanation,
    )
