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
from xclinvision.processing import process_and_filter_xray, read_image_grayscale
from xclinvision.config import get_class_names, is_multilabel
from xclinvision.xai import generate_explanation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Return a torch.device, falling back to CPU when CUDA is unavailable."""
    if isinstance(device, torch.device):
        return device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable falling back to CPU.")
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
        dataset_mean: Optional[List[float]] = None,
        dataset_std: Optional[List[float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
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
            dataset_mean: Optional mean values for input normalization.
            dataset_std: Optional std values for input normalization.
        """
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.architecture = architecture
        self.temperature_scaler = temperature_scaler
        self.mc_samples = mc_samples
        self.image_size = image_size
        self.class_names = class_names or get_class_names()
        self.dataset_mean = dataset_mean if dataset_mean is not None else [0.485, 0.456, 0.406]
        self.dataset_std = dataset_std if dataset_std is not None else [0.229, 0.224, 0.225]
        self.multilabel = is_multilabel()
        # Per-class thresholds: dict mapping class_name -> threshold (default 0.5)
        if thresholds is not None:
            self._thresholds = thresholds
        else:
            self._thresholds = {n: 0.5 for n in self.class_names}
        self._threshold_array = np.array([self._thresholds.get(n, 0.5) for n in self.class_names])
        # Cache a torch tensor version on the correct device for GPU-side comparison
        self._threshold_tensor = torch.tensor(
            self._threshold_array, device=self.device, dtype=torch.float32
        )

    # -----------------------------------------------------------------------
    # Temperature scaling helper (DRY — used by predict, predict_batch, compute_uncertainty)
    # -----------------------------------------------------------------------
    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to raw logits BEFORE activation.

        Temperature scaling divides logits by T (a positive scalar) to
        sharpen or flatten the probability distribution *before* sigmoid
        or softmax is applied.  This must happen before any activation.

        Supports:
        - TemperatureScaler objects (.temperature float attribute)
        - Raw float/int temperature values
        - Torch Tensor temperature values
        """
        if self.temperature_scaler is None:
            return logits

        # Extract the scalar T from whichever representation we have
        if isinstance(self.temperature_scaler, (int, float)):
            T = float(self.temperature_scaler)
        elif hasattr(self.temperature_scaler, "temperature"):
            T = float(self.temperature_scaler.temperature)
        elif hasattr(self.temperature_scaler, "T") and isinstance(self.temperature_scaler.T, torch.Tensor):
            return logits / self.temperature_scaler.T.to(logits.device)
        else:
            logger.warning(
                "temperature_scaler has no recognised temperature attribute; "
                "returning unscaled logits."
            )
            return logits

        if T <= 0:
            logger.warning("Temperature T=%.4f is non-positive; clamping to 0.01", T)
            T = 0.01

        return logits / T

    # -----------------------------------------------------------------------
    # Pre-processing
    # -----------------------------------------------------------------------s
    def preprocess(
        self,
        image: Union[np.ndarray, str, Image.Image],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess a single image into a normalised tensor + raw RGB array.

        Replicates the exact training preprocessing chain:
          1. Load image as grayscale.
          2. process_and_filter_xray(): auto-crop, dark-overlay cleaning, CLAHE,
             letterbox-pad to (image_size × image_size) — identical to training.
          3. Convert to 3-channel RGB.
          4. ImageNet normalisation + convert to tensor.

        Args:
            image: A file-path string, PIL Image, or NumPy array (any channel
                count).

        Returns:
            Tuple of (tensor [1,C,H,W], vis_image [H,W,3 uint8]).
        """
        # --- Load and convert to grayscale (matches training pipeline) -----
        if isinstance(image, str):
            gray = read_image_grayscale(image)
            if gray is None:
                raise FileNotFoundError(f"Cannot read image from path: {image}")
        elif isinstance(image, Image.Image):
            arr = np.array(image.convert("L"))
            gray = arr
        else:
            arr = np.asarray(image)
            if arr.ndim == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY if arr.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY)
            else:
                gray = arr

        # --- Apply full preprocessing pipeline (CLAHE, crop, letterbox) ---
        proc_result = process_and_filter_xray(gray, target_size=self.image_size)
        processed = proc_result.image
        if processed is None:
            logger.warning(
                "process_and_filter_xray rejected image (%s); "
                "falling back to plain resize. Predictions may be less reliable.",
                proc_result.reason,
            )
            processed = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        vis_image = img.copy()

        # --- Normalise and convert to tensor --------------------------------
        norm = img.astype(np.float32) / 255.0
        mean = np.array(self.dataset_mean, dtype=np.float32)
        std  = np.array(self.dataset_std, dtype=np.float32)
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
            logits = self._apply_temperature(self.model(x))

            if self.multilabel:
                probs = torch.sigmoid(logits)
                preds_binary = (probs >= self._threshold_tensor).int()[0]  # (num_classes,)
                active_indices = preds_binary.nonzero(as_tuple=True)[0].tolist()
                pred_class = active_indices[0] if active_indices else int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, pred_class].item())
                predicted_names = [
                    self.class_names[i] for i in active_indices
                ] if active_indices else [self.class_names[pred_class]]
            else:
                probs = F.softmax(logits, dim=1)
                pred_class = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, pred_class].item())
                predicted_names = None

        result: Dict[str, Any] = {
            "prediction": pred_class,
            "class_name": self.class_names[pred_class]
            if pred_class < len(self.class_names)
            else "Unknown",
            "probabilities": probs[0].cpu().numpy().tolist(),
            "confidence": confidence,
            "class_names": self.class_names,
        }
        if self.multilabel:
            result["predictions_multilabel"] = preds_binary.cpu().tolist()
            result["class_names_predicted"] = predicted_names

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
                logits = self._apply_temperature(self.model(x_batch))
                if self.multilabel:
                    probs_batch = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs_batch = F.softmax(logits, dim=1).cpu().numpy()
            
            batched_mc_preds = None
            if return_uncertainty:
                self.enable_mc_dropout()
                mc_preds_list = []
                with torch.no_grad():
                    for _ in range(self.mc_samples):
                        mc_logits = self._apply_temperature(self.model(x_batch))
                        if self.multilabel:
                            mc_preds_list.append(torch.sigmoid(mc_logits).cpu().numpy())
                        else:
                            mc_preds_list.append(F.softmax(mc_logits, dim=1).cpu().numpy())
                self.model.eval()
                batched_mc_preds = np.array(mc_preds_list) # Shape: (mc_samples, batch_size, num_classes)

            for i, (vis_image, prob) in enumerate(zip(vis_images, probs_batch)):
                if self.multilabel:
                    preds_binary = (prob >= self._threshold_array).astype(int)
                    active_indices = np.where(preds_binary)[0].tolist()
                    pred_class = active_indices[0] if active_indices else int(np.argmax(prob))
                    predicted_names = [
                        self.class_names[j] for j in active_indices
                    ] if active_indices else [self.class_names[pred_class]]
                else:
                    pred_class = int(np.argmax(prob))
                    preds_binary = None
                    predicted_names = None

                result: Dict[str, Any] = {
                    "prediction": pred_class,
                    "class_name": self.class_names[pred_class]
                    if pred_class < len(self.class_names)
                    else "Unknown",
                    "probabilities": prob.tolist(),
                    "confidence": float(prob[pred_class]),
                    "class_names": self.class_names,
                }
                if self.multilabel:
                    result["predictions_multilabel"] = preds_binary.tolist()
                    result["class_names_predicted"] = predicted_names
                if return_uncertainty:
                    item_preds = batched_mc_preds[:, i, :] # Shape: (mc_samples, num_classes)
                    mean_pred = item_preds.mean(axis=0)
                    
                    epistemic = float(item_preds.var(axis=0).mean())
                    predictive_entropy = float(-np.sum(mean_pred * np.log(mean_pred + 1e-10)))
                    
                    result["uncertainty"] = {
                        "epistemic": epistemic,
                        "predictive_entropy": predictive_entropy,
                        "mc_samples": self.mc_samples,
                    }
                    result["uncertainty_level"] = self._get_uncertainty_level(result["uncertainty"])
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
                logits = self._apply_temperature(self.model(x))
                if self.multilabel:
                    mc_preds.append(torch.sigmoid(logits).cpu().numpy())
                else:
                    mc_preds.append(F.softmax(logits, dim=1).cpu().numpy())

        self.model.eval()

        preds = np.array(mc_preds)     
        mean_pred = preds.mean(axis=0) 

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
