"""Model architectures for Chest X-ray Classification.
modeling.py - Defines all model architectures, including custom BiomedCLIP wrapper, 
ensemble logic, and advanced progressive unfreezing utilities.
"""

import logging
import types
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import timm

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Standardized mapping for timm architectures
TIMM_MODEL_MAP = {
    "densenet": "densenet121",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
    "swin_s": "swin_small_patch4_window7_224",
    "swin_b": "swin_base_patch4_window7_224",
}

class BiomedCLIPClassifier(nn.Module):
    """
    Custom wrapper for Microsoft's BiomedCLIP.
    Extracts the pre-trained Vision Transformer from the CLIP model 
    and adds a linear classification head for our 3 classes.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required for BiomedCLIP. "
                "Install with: pip install open_clip_torch"
            )
        
        logger.info("Loading Microsoft BiomedCLIP foundation model...")
        hub_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        
        try:
            if pretrained:
                model, _, _ = open_clip.create_model_and_transforms(hub_id)
            else:
                model = open_clip.create_model('ViT-B-16', pretrained=False)
            self.vision_encoder = model.visual
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            raise
        
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feat_dim = self.vision_encoder(dummy_input).shape[-1]
            
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes)
        )
        
        self._target_layer = self._find_target_layer()
        
        if not pretrained:
            self._random_init()
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the last attention block for visualization."""
        try:
            if hasattr(self.vision_encoder, 'transformer'):
                blocks = list(self.vision_encoder.transformer.resblocks.children())
                if blocks:
                    last_block = blocks[-1]
                    return getattr(last_block, 'attn', last_block)
            
            elif hasattr(self.vision_encoder, 'blocks'):
                last_block = self.vision_encoder.blocks[-1]
                return getattr(last_block, 'attn', last_block)
            
            for name, module in reversed(list(self.vision_encoder.named_modules())):
                if 'attn' in name.lower() and hasattr(module, 'qkv'):
                    return module
            return None
        except Exception as e:
            logger.warning(f"Error finding target layer for BiomedCLIP: {e}")
            return None
    
    def _random_init(self):
        """Randomly initialize vision encoder."""
        if hasattr(self.vision_encoder, 'init_parameters'):
            self.vision_encoder.init_parameters()
        else:
            for param in self.vision_encoder.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        features = self.vision_encoder(x)
        return self.head(features)
    
    def get_gradcam_target(self) -> Optional[nn.Module]:
        return self._target_layer


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple models for robust predictions.
    Supports 'average' (logits) and 'vote' (probabilities) methods.
    """
    def __init__(self, models: List[nn.Module], method: str = 'average', weights: Optional[List[float]] = None, img_size: int = 224):
        super().__init__()
        if not models:
            raise ValueError("At least one model required for ensemble")
        
        self.models = nn.ModuleList(models)
        self.method = method.lower()
        self.weights = weights
        self.img_size = img_size
        
        if self.method not in ['average', 'vote', 'weighted']:
            raise ValueError(f"Method must be 'average', 'vote', or 'weighted', got {method}")

        if self.method == 'weighted' and weights is None:
            raise ValueError("weights must be provided when method='weighted'")
        
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match models ({len(models)})")
            if abs(sum(weights) - 1.0) > 1e-4:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.4f}")
        
        self._validate_models()
    
    def _validate_models(self):
        first_model = self.models[0]
        device = next(first_model.parameters()).device if list(first_model.parameters()) else torch.device('cpu')
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size, device=device)
        
        with torch.no_grad():
            first_output = first_model(dummy_input)
        
        self.num_classes = first_output.shape[-1]
        
        for i, model in enumerate(self.models[1:], 1):
            with torch.no_grad():
                output = model(dummy_input)
            if output.shape[-1] != self.num_classes:
                raise ValueError(f"Model {i} output size {output.shape[-1]} != {self.num_classes}")
        
        logger.info(f"Ensemble validated: {len(self.models)} models, {self.num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model(x) for model in self.models]
        
        if self.method == 'average':
            stacked = torch.stack(predictions, dim=0)
            return stacked.mean(dim=0)
        
        elif self.method == 'weighted' and self.weights is not None:
            stacked = torch.stack(predictions, dim=0)
            weights_tensor = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
            return (stacked * weights_tensor).sum(dim=0)
        
        else:  # vote
            probs = torch.stack([torch.softmax(p, dim=1) for p in predictions], dim=0)
            avg_probs = probs.mean(dim=0)
            return torch.log(avg_probs + 1e-10)
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [model(x) for model in self.models]


def build_model(
    model_name: str = "densenet", 
    num_classes: int = 3, 
    pretrained: bool = True,
    dropout: float = 0.2,
    drop_path_rate: Optional[float] = None,
    img_size: int = 224
) -> nn.Module:
    """Factory function to build state-of-the-art architectures."""
    model_name = model_name.lower().strip()
    
    if model_name == "biomedclip":
        return BiomedCLIPClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    
    if model_name not in TIMM_MODEL_MAP:
        available = list(TIMM_MODEL_MAP.keys()) + ['biomedclip']
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {available}")
    
    timm_name = TIMM_MODEL_MAP[model_name]
    logger.info(f"Building {timm_name} (pretrained={pretrained})...")
    
    model_kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_rate": dropout,
    }
    
    is_transformer = any(x in model_name for x in ['vit', 'swin', 'deit', 'convnext'])
    if is_transformer:
        if drop_path_rate is None:
            drop_path_rate = 0.1
        model_kwargs["drop_path_rate"] = drop_path_rate
        logger.info(f"Using drop_path_rate={drop_path_rate} for {model_name}")
        
    if any(x in model_name for x in ['vit', 'swin', 'deit']):
        model_kwargs["img_size"] = img_size
    
    try:
        model = timm.create_model(timm_name, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to create {timm_name}: {e}")
        raise
    
    model.num_classes = num_classes
    _add_gradcam_support(model, model_name)
    
    return model


def _add_gradcam_support(model: nn.Module, architecture: str):
    """Monkey-patch Grad-CAM support securely using types.MethodType."""
    target = None
    if any(x in architecture for x in ['resnet', 'densenet', 'efficientnet', 'convnext', 'mobilenet']):
        target = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)
    elif any(x in architecture for x in ['vit', 'swin']):
        target = _find_transformer_attention(model)

    if target:
        model.get_gradcam_target = types.MethodType(lambda self: target, model)
        logger.debug(f"Grad-CAM target bound to {target.__class__.__name__}")
    else:
        logger.warning(f"No Grad-CAM target found for {architecture}")


def _find_transformer_attention(model: nn.Module) -> Optional[nn.Module]:
    """Find last attention module in transformer architecture."""
    patterns = [
        lambda m: m.blocks[-1].attn if hasattr(m, 'blocks') else None,
        lambda m: m.layers[-1].blocks[-1].attn if hasattr(m, 'layers') else None,
    ]
    for pattern in patterns:
        try:
            result = pattern(model)
            if result is not None and hasattr(result, 'qkv'):
                return result
        except Exception:
            continue
    
    for name, module in reversed(list(model.named_modules())):
        if 'attn' in name.lower() and hasattr(module, 'qkv'):
            return module
    return None


def get_param_counts(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Get parameter counts for model analysis."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_pct': 100.0 * trainable / total if total > 0 else 0.0,
        'frozen_pct': 100.0 * frozen / total if total > 0 else 0.0,
        'total_m': total / 1e6,
        'trainable_m': trainable / 1e6,
        'frozen_m': frozen / 1e6,
    }


def get_model_normalization(model: nn.Module, model_name: str) -> Dict[str, tuple]:
    """Retrieve the correct normalization mean/std for the architecture."""
    if model_name.lower() == "biomedclip":
        return {
            "mean": (0.48145466, 0.4578275, 0.40821073),
            "std":  (0.26862954, 0.26130258, 0.27577711)
        }
    if hasattr(model, 'default_cfg'):
        return {
            "mean": model.default_cfg.get('mean', (0.485, 0.456, 0.406)),
            "std":  model.default_cfg.get('std', (0.229, 0.224, 0.225))
        }
    # Fallback standard ImageNet
    return {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def freeze_backbone(model: nn.Module, unfreeze_head: bool = True) -> None:
    """Freeze all layers except the classification head."""
    for param in model.parameters():
        param.requires_grad = False
    
    if not unfreeze_head:
        logger.info("Frozen entire model")
        return
    
    if isinstance(model, EnsembleClassifier):
        for sub_model in model.models:
            freeze_backbone(sub_model, unfreeze_head=unfreeze_head)
        return
    
    unfrozen = False
    
    if isinstance(model, BiomedCLIPClassifier):
        for param in model.head.parameters(): param.requires_grad = True
        unfrozen = True
    elif hasattr(model, 'get_classifier') and callable(model.get_classifier):
        classifier = model.get_classifier()
        if classifier is not None:
            for param in classifier.parameters(): param.requires_grad = True
            unfrozen = True
    elif hasattr(model, 'fc') and model.fc is not None:
        for param in model.fc.parameters(): param.requires_grad = True
        unfrozen = True
    elif hasattr(model, 'head') and model.head is not None:
        for param in model.head.parameters(): param.requires_grad = True
        unfrozen = True
    elif hasattr(model, 'classifier') and model.classifier is not None:
        for param in model.classifier.parameters(): param.requires_grad = True
        unfrozen = True
    
    if not unfrozen:
        last_module = list(model.children())[-1]
        for param in last_module.parameters():
            param.requires_grad = True
        logger.warning(f"Unfroze last module as fallback: {type(last_module).__name__}")
    
    counts = get_param_counts(model)
    logger.info(f"Frozen backbone: {counts['frozen_m']:.2f}M frozen, {counts['trainable_m']:.2f}M trainable ({counts['trainable_pct']:.1f}%)")


def unfreeze_layers(model: nn.Module, num_layers: int = 0) -> None:
    """
    Gradually unfreeze layers for progressive fine-tuning.
    Smartly hunts for feature blocks in TIMM architectures.
    """
    if num_layers == 0:
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all layers")
        return
    
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")
    
    freeze_backbone(model, unfreeze_head=True)
    
    target_seq = None
    if hasattr(model, 'blocks') and isinstance(model.blocks, nn.Sequential):
        target_seq = model.blocks
    elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        target_seq = model.features
    elif hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential):
        target_seq = model.layers
    # Fix #22: timm ConvNext exposes feature blocks as model.stages (nn.Sequential).
    # Without this branch, the fallback (target_seq = model) iterates all top-level
    # children including the head, effectively unfreezing the whole model at once.
    elif hasattr(model, 'stages') and isinstance(model.stages, nn.Sequential):
        target_seq = model.stages
    else:
        target_seq = model 
        
    valid_blocks = []
    for child in target_seq.children():
        # Only consider blocks that actually have learnable parameters
        if list(child.parameters()) and \
           not isinstance(child, (nn.Linear, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Dropout)):
            valid_blocks.append(child)
    
    blocks_to_unfreeze = valid_blocks[-num_layers:] if num_layers < len(valid_blocks) else valid_blocks
    
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True
    
    counts = get_param_counts(model)
    logger.info(f"Progressive unfreeze (last {num_layers} blocks): {counts['trainable_m']:.2f}M trainable ({counts['trainable_pct']:.1f}%)")


def get_target_layer(model: nn.Module, architecture: str) -> Optional[nn.Module]:
    if hasattr(model, 'get_gradcam_target') and callable(model.get_gradcam_target):
        return model.get_gradcam_target()
    
    arch = architecture.lower()
    if any(x in arch for x in ['resnet', 'densenet', 'efficientnet', 'convnext']):
        return get_last_conv_layer(model)
    elif any(x in arch for x in ['vit', 'swin', 'biomedclip']):
        return _find_transformer_attention(model)
    return None


def get_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


# =============================================================================
# TESTING
# =============================================================================

def _test_model(model_name: str, num_classes: int = 3, batch_size: int = 2, device: str = 'cpu', img_size: int = 384) -> Dict:
    try:
        model = build_model(model_name, num_classes=num_classes, pretrained=False, img_size=img_size)
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (batch_size, num_classes), f"Bad output shape: {output.shape}"
        
        counts = get_param_counts(model)
        target = get_target_layer(model, model_name)
        
        return {
            'name': model_name, 'status': '✓ PASS', 'output_shape': tuple(output.shape),
            'params_m': counts['total_m'], 'trainable_m': counts['trainable_m'],
            'target_type': type(target).__name__ if target else 'None', 'error': None
        }
    except Exception as e:
        return {
            'name': model_name, 'status': '✗ FAIL', 'output_shape': None,
            'params_m': 0, 'trainable_m': 0, 'target_type': 'N/A', 'error': str(e)
        }


def _test_ensemble(models: List[nn.Module], method: str = 'average', weights: Optional[List[float]] = None, img_size: int = 384) -> Dict:
    try:
        ensemble = EnsembleClassifier(models, method=method, weights=weights, img_size=img_size)
        device = next(models[0].parameters()).device
        dummy_input = torch.randn(2, 3, img_size, img_size, device=device)
        
        with torch.no_grad():
            output = ensemble(dummy_input)
        
        return {'method': method, 'status': '✓ PASS', 'output_shape': tuple(output.shape), 'n_models': len(models), 'error': None}
    except Exception as e:
        return {'method': method, 'status': '✗ FAIL', 'output_shape': None, 'n_models': len(models), 'error': str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    
    TEST_MODELS = [
        "densenet", "resnet50", "efficientnet_b0", "efficientnet_b2",
        "convnext_tiny", "vit_small", "swin_t",
        "biomedclip",  # Uncomment if open_clip installed
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}\nMODEL TESTING SUITE | Device: {device}\n{'='*70}")
    
    results = []
    for name in TEST_MODELS:
        print(f"\nTesting: {name.upper()}")
        result = _test_model(name, device=device)
        results.append(result)
        if result['status'] == '✓ PASS':
            print(f"  Output: {result['output_shape']}\n  Params: {result['params_m']:.2f}M (trainable: {result['trainable_m']:.2f}M)\n  XAI target: {result['target_type']}")
        else:
            print(f"  ERROR: {result['error']}")
    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}\n{'Model':<20} {'Status':<8} {'Params':<12} {'XAI Target'}\n{'-'*70}")
    for r in results:
        params = f"{r['params_m']:.2f}M" if r['params_m'] > 0 else "N/A"
        print(f"{r['name']:<20} {r['status']:<8} {params:<12} {r['target_type']}")
    
    successful_models = [build_model(r['name'], pretrained=False, img_size=384).to(device) for r in results if r['status'] == '✓ PASS'][:2]
    
    if len(successful_models) >= 2:
        print(f"\n{'='*70}\nENSEMBLE TESTS\n{'='*70}")
        for method in ['average', 'vote', 'weighted']:
            weights = [1.0 / len(successful_models)] * len(successful_models) if method == 'weighted' else None
            result = _test_ensemble(successful_models, method=method, weights=weights)
            status = "✓" if result['status'] == '✓ PASS' else "✗"
            print(f"{status} Ensemble ({method}): {result.get('output_shape', 'FAILED')}")
    
    print(f"\n{'='*70}")