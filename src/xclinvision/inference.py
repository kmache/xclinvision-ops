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


#: Tier ordering used to pick the headline finding when several labels are
#: positive; lower sorts first, and probability breaks ties *within* a tier.
#: See :meth:`InferencePipeline._rank_active` for the rule and its rationale.
_PRIORITY_RANK: Dict[str, int] = {"critical": 0, "urgent": 1, "routine": 2}

#: Cached tier map, resolved on first use when no priority_map was injected.
_FALLBACK_PRIORITY_MAP: Optional[Dict[str, str]] = None


def _default_priority_map() -> Dict[str, str]:
    """Tier map derived from guardrails' CRITICAL/URGENT condition sets.

    Imported lazily and cached: ``xclinvision.agent.guardrails`` runs the agent
    package __init__, which pulls chromadb + sentence-transformers (~9.7s), so
    a module-level import here would tax every ``import xclinvision.inference``
    including the training and evaluation scripts. Callers that inject a
    ``priority_map`` (the backend does, from the ThresholdProfile) never reach
    this.
    """
    global _FALLBACK_PRIORITY_MAP
    if _FALLBACK_PRIORITY_MAP is None:
        try:
            from xclinvision.agent.guardrails import (
                CRITICAL_CONDITIONS,
                URGENT_CONDITIONS,
            )
            resolved = {name: "critical" for name in CRITICAL_CONDITIONS}
            resolved.update({name: "urgent" for name in URGENT_CONDITIONS})
            _FALLBACK_PRIORITY_MAP = resolved
        except Exception as exc:
            logger.warning(
                "Could not load clinical priority tiers (%s); multilabel "
                "headline ranking falls back to probability order.", exc,
            )
            _FALLBACK_PRIORITY_MAP = {}
    return _FALLBACK_PRIORITY_MAP


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
        priority_map: Optional[Dict[str, str]] = None,
        calibration_status: Optional[str] = None,
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
        # class_name -> "critical" | "urgent" | "routine". Absent entries sort
        # as "routine", so an unsupplied map degrades to probability ranking.
        self._priority_map = priority_map or {}
        #: One warning per pipeline, not per inference.
        self._warned_uncalibrated = False
        #: Explicit verdict from the fit, carried in the checkpoint payload.
        #: A fitted scaler is necessary but NOT sufficient to call the output
        #: calibrated: the fit also has to have worked. Checkpoints whose
        #: held-out ECE stayed above the quality bar keep "uncalibrated" even
        #: though they carry parameters, so the report layer keeps hedging the
        #: language. None falls back to "a scaler exists", which is what a
        #: caller constructing a pipeline by hand means.
        self._calibration_status = calibration_status
        # Cache a torch tensor version on the correct device for GPU-side comparison
        self._threshold_tensor = torch.tensor(
            self._threshold_array, device=self.device, dtype=torch.float32
        )

    def _rank_active(self, active_indices: List[int], probs_row) -> List[int]:
        """Order positive labels by clinical priority tier, then by probability.

        THE RULE
        --------
        Sort key is ``(tier_rank, -probability)``: CRITICAL before URGENT before
        routine, and within one tier the higher probability first. Tiers come from
        the injected ``priority_map`` — the backend supplies one built from its
        ``ThresholdProfile`` — or, for direct ``InferencePipeline`` users, from
        :func:`_default_priority_map`, which reads ``CRITICAL_CONDITIONS`` and
        ``URGENT_CONDITIONS`` in ``xclinvision.agent.guardrails``. Any label in
        neither set is ``routine``. With no tier information at all the key
        degrades to pure probability order.

        THE CONSEQUENCE
        ---------------
        **A finding in a higher tier outranks a lower-tier finding that has a
        higher probability.** The headline is therefore routinely *not* the
        model's most confident label, and that is the policy rather than a bug.

        Worked example — all three above their thresholds::

            Pulmonary fibrosis  p=0.95  routine
            Cardiomegaly        p=0.88  urgent
            Pneumothorax        p=0.62  critical

            ranked -> ["Pneumothorax", "Cardiomegaly", "Pulmonary fibrosis"]
            headline = Pneumothorax at 0.62, the *lowest*-probability positive.

        Within a single tier probability decides, unchanged::

            two routine findings at p=0.70 and p=0.95 -> the 0.95 one leads.

        WHY THE BIAS IS DELIBERATE
        --------------------------
        Exactly one label becomes ``class_name``, and that single label drives the
        LLM summary, the exported report and the urgency gate. Ranking by
        confidence buries a moderate-probability emergency behind a confident
        chronic finding; under-triaging a pneumothorax costs more clinically than
        over-triaging a fibrosis. Every positive is still returned in
        ``class_names_predicted``, so nothing is hidden — only the ordering is
        opinionated.

        This replaced ``active_indices[0]`` — the lowest class *index*, i.e.
        whatever ``configs/system.yaml`` happened to list first — under which a
        pneumothorax at 0.55 was reported behind a cardiomegaly at 0.97 purely
        because of YAML ordering.

        NOTE — depends on the configured class list, not on this function.
        With ``configs/system.yaml`` as shipped, only Cardiomegaly is tiered
        (urgent); Aortic enlargement, Pleural thickening and Pulmonary fibrosis
        are all routine, and the critical tier is unreachable because neither
        Pneumothorax nor Consolidation is a served class. The rule therefore
        currently reduces to "Cardiomegaly first when positive, otherwise
        probability order". Adding a critical class to ``model.class_names``
        changes that without any change here.
        """
        priority = self._priority_map or _default_priority_map()
        return sorted(
            active_indices,
            key=lambda i: (
                _PRIORITY_RANK.get(priority.get(self.class_names[i], "routine"), 2),
                -float(probs_row[i]),
            ),
        )

    # -----------------------------------------------------------------------
    # Temperature scaling helper (DRY — used by predict, predict_batch, compute_uncertainty)
    # -----------------------------------------------------------------------
    def _is_calibrated(self) -> bool:
        """Whether served probabilities may be described as calibrated.

        Requires both a fitted scaler and, when the checkpoint states one, a
        calibration_status of "calibrated". A checkpoint whose fit failed to
        clear the ECE bar carries its parameters but stays "uncalibrated".
        """
        if self.temperature_scaler is None:
            return False
        if self._calibration_status is not None:
            return self._calibration_status == "calibrated"
        return True

    def _per_class_calibration(self):
        """Return ``(T, bias)`` tensors when calibration is per-class, else None.

        Accepts the three shapes a checkpoint or caller can supply: a fitted
        ``TemperatureScaler`` whose ``temperature`` is a sequence, a bare
        sequence of temperatures, or a ``{"temperature": [...], "bias": [...]}``
        mapping as stored in the exported ``.pth`` payload.
        """
        scaler = self.temperature_scaler
        temps = bias = None

        if isinstance(scaler, dict):
            temps, bias = scaler.get("temperature"), scaler.get("bias")
        elif isinstance(scaler, (list, tuple)):
            temps, bias = scaler, None
        elif hasattr(scaler, "temperature") and isinstance(
            getattr(scaler, "temperature"), (list, tuple)
        ):
            temps, bias = scaler.temperature, getattr(scaler, "bias", None)

        if not isinstance(temps, (list, tuple)):
            return None

        n = len(self.class_names)
        if len(temps) != n:
            logger.warning(
                "Per-class temperature has %d entries but the model serves %d "
                "classes; ignoring calibration.", len(temps), n,
            )
            return None

        t = torch.tensor([max(float(x), 1e-4) for x in temps], dtype=torch.float32)
        b = torch.zeros(n, dtype=torch.float32)
        if isinstance(bias, (list, tuple)) and len(bias) == n:
            b = torch.tensor([float(x) for x in bias], dtype=torch.float32)
        self._calibration_is_per_class = True
        return t, b

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to raw logits BEFORE activation.

        Temperature scaling divides logits by T (a positive scalar) to
        sharpen or flatten the probability distribution *before* sigmoid
        or softmax is applied.  This must happen before any activation.

        Supports:
        - TemperatureScaler objects (.temperature float attribute, optional .bias)
        - Raw float/int temperature values
        - Per-class sequences: ``[T0, T1, ...]`` or ``{"temperature": [...],
          "bias": [...]}`` — applied elementwise as ``z / T + b``
        - Torch Tensor temperature values

        The bias term matters here. Temperature alone can only pull logits
        toward or away from zero, so it cannot remove a systematic offset;
        these checkpoints trained with pos_weight 5.1-15.2 and carry exactly
        such an offset. See ``TemperatureScaler.fit_per_class``.
        """
        if self.temperature_scaler is None:
            if not self._warned_uncalibrated:
                self._warned_uncalibrated = True
                logger.warning(
                    "Temperature scaling is a no-op for %s; served probabilities "
                    "are uncalibrated.", self.architecture,
                )
            return logits

        # --- per-class affine: z / T + b, one T and b per class ------------
        vec = self._per_class_calibration()
        if vec is not None:
            t_vec, b_vec = vec
            return logits / t_vec.to(logits.device) + b_vec.to(logits.device)

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
        xai_method: str = "gradcam++",
        target_class: Optional[int] = None,
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

            In multilabel mode ``class_name`` is the *ranked* headline, not the
            highest-probability label: positives are ordered by clinical tier
            first and probability only within a tier, so a lower-probability
            urgent finding leads a higher-probability routine one. Every
            positive is listed in ``class_names_predicted``. See
            :meth:`_rank_active` for the rule and why it is biased that way.
        """
        x, vis_image = self.preprocess(image)

        with torch.no_grad():
            logits = self._apply_temperature(self.model(x))

            if self.multilabel:
                probs = torch.sigmoid(logits)
                preds_binary = (probs >= self._threshold_tensor).int()[0]  # (num_classes,)
                active_indices = preds_binary.nonzero(as_tuple=True)[0].tolist()
                active_indices = self._rank_active(active_indices, probs[0])
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
            # Honest name for the top-1 score. Until a temperature is fitted
            # this is a raw sigmoid/softmax output, not a calibrated
            # probability, so calling it "confidence" overstates it.
            "raw_probability": confidence,
            # Deprecated alias, kept populated so existing consumers (frontend,
            # agent tools, stored analyses) keep working. Prefer
            # raw_probability.
            "confidence": confidence,
            "class_names": self.class_names,
            "thresholds": [float(t) for t in self._threshold_array],
            # A fitted scaler is necessary but not sufficient: the checkpoint
            # also has to have passed the held-out ECE bar at fit time. See
            # self._calibration_status.
            "calibrated": self._is_calibrated(),
            "calibration_status": ("calibrated" if self._is_calibrated() else "uncalibrated"),
        }
        if self.multilabel:
            result["predictions_multilabel"] = preds_binary.cpu().tolist()
            result["class_names_predicted"] = predicted_names

        if return_uncertainty:
            uncertainty = self.compute_uncertainty(x)
            result["uncertainty"] = uncertainty
            result["uncertainty_level"] = self._get_uncertainty_level(uncertainty)

        if return_explanation:
            xai_target = target_class if target_class is not None else pred_class
            exp_res = generate_explanation(
                model=self.model,
                image=vis_image,
                prediction=xai_target,
                confidence=confidence,
                class_names=self.class_names,
                architecture=self.architecture,
                device=str(self.device),
                img_size=self.image_size,
                method=xai_method,
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
                try:
                    with torch.no_grad():
                        for _ in range(self.mc_samples):
                            mc_logits = self._apply_temperature(self.model(x_batch))
                            if self.multilabel:
                                mc_preds_list.append(torch.sigmoid(mc_logits).cpu().numpy())
                            else:
                                mc_preds_list.append(F.softmax(mc_logits, dim=1).cpu().numpy())
                finally:
                    self.model.eval()
                batched_mc_preds = np.array(mc_preds_list) # Shape: (mc_samples, batch_size, num_classes)

            for i, (vis_image, prob) in enumerate(zip(vis_images, probs_batch)):
                if self.multilabel:
                    preds_binary = (prob >= self._threshold_array).astype(int)
                    active_indices = self._rank_active(np.where(preds_binary)[0].tolist(), prob)
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
                    "raw_probability": float(prob[pred_class]),
                    "confidence": float(prob[pred_class]),  # deprecated alias
                    "class_names": self.class_names,
                    "calibrated": self.temperature_scaler is not None,
                }
                if self.multilabel:
                    result["predictions_multilabel"] = preds_binary.tolist()
                    result["class_names_predicted"] = predicted_names
                if return_uncertainty:
                    item_preds = batched_mc_preds[:, i, :] # Shape: (mc_samples, num_classes)
                    mean_pred = item_preds.mean(axis=0)
                    
                    epistemic = float(item_preds.var(axis=0).mean())
                    predictive_entropy = self._predictive_entropy(mean_pred, self.multilabel)
                    
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

    @staticmethod
    def _predictive_entropy(mean_pred: np.ndarray, multilabel: bool) -> float:
        """Predictive entropy of an averaged prediction distribution.

        Multiclass softmax outputs sum to 1, so categorical Shannon entropy
        ``-Σ p·log p`` is appropriate.

        Multilabel sigmoid outputs are independent Bernoulli per label and
        do *not* sum to 1; applying the categorical formula to them gives a
        meaningless value (issue #2). The correct quantity is per-label
        binary entropy ``-(p·log p + (1-p)·log(1-p))``, here averaged
        across labels so the result remains bounded in ``[0, log 2]`` and
        is comparable across configurations with different label counts.
        """
        eps = 1e-10
        if multilabel:
            binary = -(
                mean_pred * np.log(mean_pred + eps)
                + (1.0 - mean_pred) * np.log(1.0 - mean_pred + eps)
            )
            return float(np.mean(binary))
        # np.sum on shape (1, K) returns a (1,) array; squeeze before float()
        # to avoid NumPy 1.25 deprecation warning.
        return float(np.squeeze(-np.sum(mean_pred * np.log(mean_pred + eps), axis=-1)))

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
        try:
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    logits = self._apply_temperature(self.model(x))
                    if self.multilabel:
                        mc_preds.append(torch.sigmoid(logits).cpu().numpy())
                    else:
                        mc_preds.append(F.softmax(logits, dim=1).cpu().numpy())
        finally:
            self.model.eval()

        preds = np.array(mc_preds)
        mean_pred = preds.mean(axis=0)

        epistemic = float(preds.var(axis=0).mean())

        predictive_entropy = self._predictive_entropy(mean_pred, self.multilabel)

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
    xai_method: str = "gradcam++",
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
        xai_method=xai_method,
    )
