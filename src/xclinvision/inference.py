"""Inference pipeline for model predictions with uncertainty estimation."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from xclinvision.architecture import BaseModel
from xclinvision.evaluator import TemperatureScaler


class InferencePipeline:
    """End-to-end inference pipeline for chest X-ray classification."""
    
    def __init__(
        self,
        model: BaseModel,
        device: str = "cuda",
        temperature_scaler: Optional[TemperatureScaler] = None,
        mc_samples: int = 10,
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.temperature_scaler = temperature_scaler
        self.mc_samples = mc_samples
        
    def preprocess(self, image: Union[np.ndarray, str, Image.Image]) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        # Resize and normalize
        image = cv2.resize(image, (384, 384))
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).unsqueeze(0).to(self.device)
    
    def predict(
        self,
        image: Union[np.ndarray, str, Image.Image],
        return_uncertainty: bool = True,
    ) -> Dict:
        """Run inference on a single image."""
        x = self.preprocess(image)
        
        with torch.no_grad():
            logits = self.model(x)
            
            # Apply temperature scaling if available
            if self.temperature_scaler is not None:
                logits = self.temperature_scaler.scale(logits.cpu().numpy())
                logits = torch.from_numpy(logits).to(self.device)
            
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            
        result = {
            "prediction": pred_class,
            "probabilities": probs[0].cpu().numpy(),
            "confidence": confidence,
            "class_names": ["Normal", "Pneumonia", "Tuberculosis"],
        }
        
        # Compute uncertainty if requested
        if return_uncertainty:
            uncertainty = self.compute_uncertainty(x)
            result["uncertainty"] = uncertainty
            result["uncertainty_level"] = self._get_uncertainty_level(uncertainty)
            
        return result
    
    def compute_uncertainty(self, x: torch.Tensor) -> Dict:
        """Compute epistemic uncertainty using MC Dropout."""
        self.model.enable_mc_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
                
        predictions = np.array(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(axis=0)
        
        # Epistemic uncertainty: variance across MC samples
        epistemic_unc = predictions.var(axis=0).mean()
        
        # Predictive uncertainty: entropy of mean prediction
        eps = 1e-10
        predictive_unc = -np.sum(mean_pred * np.log(mean_pred + eps))
        
        self.model.eval()
        
        return {
            "epistemic": float(epistemic_unc),
            "predictive_entropy": float(predictive_unc),
            "mc_samples": self.mc_samples,
        }
    
    def _get_uncertainty_level(self, uncertainty: Dict) -> str:
        """Convert uncertainty to categorical level."""
        epistemic = uncertainty["epistemic"]
        
        if epistemic < 0.01:
            return "low"
        elif epistemic < 0.05:
            return "medium"
        else:
            return "high"
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Image.Image]],
        return_uncertainty: bool = True,
    ) -> List[Dict]:
        """Run inference on a batch of images."""
        return [self.predict(img, return_uncertainty) for img in images]