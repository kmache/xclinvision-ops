"""Model architecture definitions and factory functions."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer


class BaseModel(nn.Module):
    """Base class for all XClinVision models."""
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.feature_dim = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer."""
        raise NotImplementedError
        
    def enable_mc_dropout(self):
        """Enable dropout for Monte Carlo sampling."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class EfficientNetModel(BaseModel):
    """EfficientNet-B2 model for chest X-ray classification."""
    
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__(num_classes, dropout_rate)
        
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,  # Remove default head
        )
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        logits = self.classifier(features)
        return logits
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        # Return pooled features before final linear layers
        return nn.AdaptiveAvgPool2d((1, 1))(features).flatten(1)


class ResNetModel(BaseModel):
    """ResNet-50 baseline model for chest X-ray classification."""
    
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__(num_classes, dropout_rate)
        
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        logits = self.classifier(features)
        return logits
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return nn.AdaptiveAvgPool2d((1, 1))(features).flatten(1)


class SwinTransformerModel(BaseModel):
    """Swin Transformer model for chest X-ray classification."""
    
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__(num_classes, dropout_rate)
        
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        logits = self.classifier(features)
        return logits
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


class BiomedCLIPModel(BaseModel):
    """BiomedCLIP foundation model for chest X-ray classification."""
    
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__(num_classes, dropout_rate)
        
        from transformers import CLIPModel, CLIPProcessor
        
        self.clip_model = CLIPModel.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.processor = CLIPProcessor.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        
        # Get vision encoder output dimension
        self.feature_dim = 512  # BiomedCLIP vision dimension
        
        # Freeze text encoder (not used for classification)
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
            
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get vision encoder features
        vision_outputs = self.clip_model.vision_model(x)
        features = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]
        
        logits = self.classifier(features)
        return logits
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(x)
        return vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]


MODEL_REGISTRY = {
    "efficientnet_b2": EfficientNetModel,
    "resnet50": ResNetModel,
    "swin_t": SwinTransformerModel,
    "biomedclip": BiomedCLIPModel,
}


def create_model(
    model_name: str,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
    **kwargs,
) -> BaseModel:
    """Factory function to create models by name.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
        
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        **kwargs,
    )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model architecture.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model metadata
    """
    info = {
        "efficientnet_b2": {
            "type": "CNN (Modern)",
            "params": "9.2M",
            "input_size": (384, 384),
            "strengths": ["Best accuracy-compute trade-off", "Commonly used in production"],
            "weaknesses": ["Slightly slower than ResNet-50"],
            "expected_recall": 0.92,
            "expected_ece": 0.05,
        },
        "resnet50": {
            "type": "CNN (Legacy)",
            "params": "25.6M",
            "input_size": (384, 384),
            "strengths": ["Stable training", "Fast inference"],
            "weaknesses": ["Moderate calibration", "Blurry heatmaps"],
            "expected_recall": 0.88,
            "expected_ece": 0.12,
        },
        "swin_t": {
            "type": "Transformer",
            "params": "28.3M",
            "input_size": (224, 224),
            "strengths": ["Excellent localization", "Captures global context"],
            "weaknesses": ["Complex", "Needs more data"],
            "expected_recall": 0.91,
            "expected_ece": 0.18,
        },
        "biomedclip": {
            "type": "Foundation Model",
            "params": "86M",
            "input_size": (224, 224),
            "strengths": ["Pre-trained on medical images", "High semantic alignment"],
            "weaknesses": ["Heavy", "Requires careful fine-tuning"],
            "expected_recall": 0.90,
            "expected_ece": 0.09,
        },
    }
    return info.get(model_name, {})
