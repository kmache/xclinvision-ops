"""Model architectures for Chest X-ray Classification."""

import torch
import torch.nn as nn
import timm
import logging

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Dictionary mapping our friendly names to the official 'timm' model names
TIMM_MODEL_MAP = {
    "densenet": "densenet121",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2",
    "convnext_tiny": "convnext_tiny",
    "vit_small": "vit_small_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224"
}

class BiomedCLIPClassifier(nn.Module):
    """
    Custom wrapper for Microsoft's BiomedCLIP.
    Extracts the pre-trained Vision Transformer from the CLIP model 
    and adds a linear classification head for our 3 classes.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required for BiomedCLIP. "
                "Install with: pip install open_clip_torch"
            )
        
        logger.info("Loading Microsoft BiomedCLIP foundation model...")
        # Load the BiomedCLIP model
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        
        # We only need the visual encoder (ViT) for image classification
        self.vision_encoder = model.visual
        
        # BiomedCLIP's vision encoder outputs a 512-dimensional vector
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
        # If not pretrained, we randomly initialize the vision encoder (rarely done)
        if not pretrained:
            # Reset vision encoder parameters
            if hasattr(self.vision_encoder, 'init_parameters'):
                self.vision_encoder.init_parameters()
            else:
                # Fallback: manually reinitialize using PyTorch's default init
                for param in self.vision_encoder.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)

    def forward(self, x):
        # Extract image features
        features = self.vision_encoder(x)
        # Classify
        return self.head(features)


def build_model(model_name="densenet", num_classes=3, pretrained=True):
    """
    Factory function to build any of the requested state-of-the-art architectures.
    
    Args:
        model_name (str): The name of the model to load.
                          Options: 'densenet', 'resnet50', 'efficientnet_b0', 
                                   'efficientnet_b2', 'convnext_tiny', 'vit_small', 
                                   'swin_t', 'biomedclip'
        num_classes (int): Number of output classes (Default: 3 for Normal, Pneum, TB)
        pretrained (bool): Whether to load pre-trained weights
    """
    model_name = model_name.lower()
    
    # 1. Handle BiomedCLIP (Special Case)
    if model_name == "biomedclip":
        return BiomedCLIPClassifier(num_classes=num_classes, pretrained=pretrained)
    
    # 2. Handle Standard Models via `timm`
    if model_name not in TIMM_MODEL_MAP:
        raise ValueError(f"Model '{model_name}' not supported. "
                         f"Choose from: {list(TIMM_MODEL_MAP.keys())} or 'biomedclip'")
    
    timm_name = TIMM_MODEL_MAP[model_name]
    logger.info(f"Loading {timm_name} via timm (Pretrained={pretrained})...")
    
    # timm automatically handles replacing the classification head 
    # when you pass `num_classes`.
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.2  # Adds dropout for regularization automatically
    )
    
    return model


def freeze_backbone(model: nn.Module, unfreeze_head: bool = True) -> None:
    """
    Freeze all layers except the classification head for transfer learning.
    """
    # 1. Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    if unfreeze_head:
        # 2. Unfreeze the classification head safely
        
        # Handle our custom BiomedCLIP
        if isinstance(model, BiomedCLIPClassifier):
            for param in model.head.parameters():
                param.requires_grad = True
                
        # Handle `timm` models natively
        elif hasattr(model, 'get_classifier') and callable(model.get_classifier):
            classifier = model.get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True
                
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
                
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
                
        else:
            # Last resort: iterate through parameters of the last child module
            last_module = list(model.children())[-1]
            for param in last_module.parameters():
                param.requires_grad = True
            
    logger.info(f"Frozen backbone, unfrozen_head={unfreeze_head}")


def unfreeze_layers(model: nn.Module, num_layers: int = 0) -> None:
    """
    Gradually unfreeze layers for fine-tuning.
    """
    children = list(model.children())
    
    if num_layers == 0 or num_layers >= len(children):
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all layers for fine-tuning")
    else:
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True
        logger.info(f"Unfrozen the last {num_layers} layer blocks")


# --- Quick Test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # List of all requested models
    models_to_test =[
        "densenet", 
        "resnet50", 
        "efficientnet_b0", 
        "efficientnet_b2", 
        "convnext_tiny", 
        "vit_small", 
        "swin_t",
        "biomedclip"
    ]
    
    # Dummy image tensor (Batch Size = 2, Channels = 3, Height = 224, Width = 224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    for name in models_to_test:
        print(f"\n--- Testing {name.upper()} ---")
        try:
            model = build_model(model_name=name, num_classes=3, pretrained=False) # False for faster local testing
            output = model(dummy_input)
            print(f"Success! Output shape: {output.shape} (Expected: [2, 3])")
        except Exception as e:
            print(f"Failed to build {name}: {e}")