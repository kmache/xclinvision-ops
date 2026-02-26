"""Explainability (XAI) module for generating clinical heatmaps."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import cv2
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


class ExplainabilityEngine:
    """Generate clinical explanations for model predictions."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        method: str = "gradcam++",
        target_layer: Optional[str] = None,
    ):
        self.model = model
        self.method = method
        self.target_layer = target_layer
        
        if not GRADCAM_AVAILABLE:
            raise ImportError("pytorch-grad-cam is required for explainability")
            
    def _get_target_layer(self) -> torch.nn.Module:
        """Get the target layer for gradient computation."""
        if self.target_layer is None:
            # Auto-detect based on model type
            if hasattr(self.model, 'backbone'):
                if 'efficientnet' in str(type(self.model.backbone)).lower():
                    return self.model.backbone.blocks[-1]
                elif 'resnet' in str(type(self.model.backbone)).lower():
                    return self.model.backbone.layer4[-1]
                elif 'swin' in str(type(self.model.backbone)).lower():
                    return self.model.backbone.layers[-1]
        
        # Try to find layer by name
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return module
                
        # Default: last convolutional layer
        for m in reversed(list(self.model.modules())):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                return m
                
        raise ValueError("Could not find suitable target layer")
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Dict:
        """Generate Grad-CAM++ heatmap for the image."""
        target_layer = self._get_target_layer()
        
        # Initialize GradCAM
        if self.method == "gradcam++":
            cam = GradCAMPlusPlus(
                model=self.model,
                target_layers=[target_layer],
            )
        else:
            cam = GradCAM(
                model=self.model,
                target_layers=[target_layer],
            )
            
        # Prepare input
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                
        # Generate CAM
        grayscale_cam = cam(
            input_tensor=self._preprocess(image),
            targets=self._get_cam_target(target_class),
        )
        
        # Overlay on original image
        heatmap = show_cam_on_image(
            image,
            grayscale_cam[0],
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
        )
        
        # Compute region importance scores
        region_scores = self._compute_region_scores(grayscale_cam[0])
        
        return {
            "heatmap": heatmap,
            "grayscale_cam": grayscale_cam[0],
            "region_scores": region_scores,
            "target_class": target_class,
            "method": self.method,
        }
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for CAM generation."""
        # Resize if needed
        if image.shape[:2] != (384, 384):
            image = cv2.resize(image, (384, 384))
            
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (image - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.float()
    
    def _get_cam_target(self, target_class: Optional[int]):
        """Get CAM target for specified class."""
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        if target_class is None:
            # Use predicted class
            return None
        return [ClassifierOutputTarget(target_class)]
    
    def _compute_region_scores(self, grayscale_cam: np.ndarray) -> Dict:
        """Compute importance scores for image regions."""
        h, w = grayscale_cam.shape
        
        # Divide into regions
        regions = {
            "left_upper": grayscale_cam[:h//2, :w//2].mean(),
            "right_upper": grayscale_cam[:h//2, w//2:].mean(),
            "left_lower": grayscale_cam[h//2:, :w//2].mean(),
            "right_lower": grayscale_cam[h//2:, w//2:].mean(),
            "center": grayscale_cam[h//4:3*h//4, w//4:3*w//4].mean(),
        }
        
        return regions
    
    def generate_attention_rollout(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Dict:
        """Generate attention rollout for transformer models."""
        # Implementation for Swin/ViT models
        # This requires specific attention extraction
        
        return {
            "heatmap": None,
            "method": "attention_rollout",
            "note": "Attention rollout requires transformer-specific implementation",
        }
    
    def explain_prediction(
        self,
        image: np.ndarray,
        prediction: int,
        confidence: float,
    ) -> Dict:
        """Generate complete explanation for a prediction."""
        # Generate heatmap
        heatmap_result = self.generate_heatmap(image, target_class=prediction)
        
        # Build explanation
        explanation = {
            "prediction": prediction,
            "confidence": confidence,
            "class_name": ["Normal", "Pneumonia", "Tuberculosis"][prediction],
            "visualization": heatmap_result,
            "key_findings": self._extract_key_findings(heatmap_result),
        }
        
        return explanation
    
    def _extract_key_findings(self, heatmap_result: Dict) -> List[str]:
        """Extract key findings from heatmap analysis."""
        findings = []
        scores = heatmap_result["region_scores"]
        
        # Find most activated regions
        max_region = max(scores, key=scores.get)
        max_score = scores[max_region]
        
        if max_score > 0.3:
            findings.append(f"High activation in {max_region.replace('_', ' ')} region")
            
        if scores["center"] > 0.4:
            findings.append("Significant findings in central lung fields")
            
        # Check for bilateral patterns
        left_total = scores["left_upper"] + scores["left_lower"]
        right_total = scores["right_upper"] + scores["right_lower"]
        
        if abs(left_total - right_total) < 0.1 and (left_total + right_total) > 0.5:
            findings.append("Bilateral pattern detected")
        elif left_total > right_total * 2:
            findings.append("Left-sided predominance")
        elif right_total > left_total * 2:
            findings.append("Right-sided predominance")
            
        return findings


def generate_explanation(
    model: torch.nn.Module,
    image: np.ndarray,
    prediction: int,
    confidence: float,
    method: str = "gradcam++",
) -> Dict:
    """Convenience function to generate explanation."""
    engine = ExplainabilityEngine(model, method=method)
    return engine.explain_prediction(image, prediction, confidence)
