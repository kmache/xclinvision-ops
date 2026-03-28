"""Model architectures for Chest X-ray Classification.
modeling.py - Defines all TIMM model architectures, 
ensemble logic, and advanced progressive unfreezing utilities.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generalized Mean (GeM) Pooling
# ---------------------------------------------------------------------------
class GeM(nn.Module):
    """Generalized Mean Pooling (Radenovic et al., 2018).

    Acts as a learnable slider between Average Pooling (p=1) and
    Max Pooling (p→∞).  Higher *p* amplifies strong local activations,
    which is critical for detecting small pathologies (fibrosis,
    pleural thickening) that occupy <5 % of the feature map.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  →  (B, C, 1, 1)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1,
        ).pow(1.0 / self.p)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p.data.item():.2f})"


# Model registry with architecture-specific metadata
MODEL_REGISTRY = {
    "densenet": {
        "timm_name": "densenet121.ra_in1k",
        "default_img_size": 224,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "features",
    },
    "resnet50": {
        "timm_name": "resnet50.a1_in1k",
        "default_img_size": 224,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "layer4",
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0.ra_in1k",
        "default_img_size": 224,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks",
    },
    "efficientnet_b2": {
        "timm_name": "efficientnet_b2.ra_in1k",
        "default_img_size": 260,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks",
    },
    "efficientnet_b3": {
        "timm_name": "efficientnet_b3.ra2_in1k",
        "default_img_size": 300,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks",
    },
    "efficientnet_b4": {
        "timm_name": "efficientnet_b4.ra2_in1k",
        "default_img_size": 380,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks",
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k",
        "default_img_size": 224,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "stages[-1].blocks[-1]",
    },
    "convnext_small": {
        "timm_name": "convnext_small.fb_in22k_ft_in1k_384",
        "default_img_size": 384,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "stages[-1].blocks[-1]",
    },
    "vit_tiny": {
        "timm_name": "vit_tiny_patch16_384.augreg_in21k_ft_in1k",
        "default_img_size": 384,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks[-1].attn.qkv",
    },
    "vit_small": {
        "timm_name": "vit_small_patch16_384.augreg_in21k_ft_in1k",
        "default_img_size": 384,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks[-1].attn.qkv",
    },
    "vit_base": {
        "timm_name": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        "default_img_size": 384,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "blocks[-1].attn.qkv",
    },
    "swin_t": {
        "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "default_img_size": 224,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "layers[-1].blocks[-1].attn.qkv",
    },
    "swin_s": {
        "timm_name": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        "default_img_size": 224,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "layers[-1].blocks[-1].attn.qkv",
    },
    "swin_b": {
        "timm_name": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
        "default_img_size": 384,
        "head_type": "linear",
        "progressive": True,
        "gradcam_target": "layers[-1].blocks[-1].attn.qkv",
    },
}

# Future-proofing: per-architecture normalization overrides
CUSTOM_NORMS = {
    "cxr_custom": {"mean": (0.505,), "std": (0.252,)}  # Example for 1-channel CXR
}

class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple models for robust predictions.
    Outputs logits for consistent loss and metric computation.
    """
    def __init__(self, models: List[nn.Module], method: str = 'average', 
                 weights: Optional[List[float]] = None):
        super().__init__()
        if not models:
            raise ValueError("At least one model required for ensemble")
        
        self.models = nn.ModuleList(models)
        self.method = method.lower()
        self.weights = weights
        
        if self.method not in ['average', 'weighted']:
            raise ValueError(f"Method must be 'average' or 'weighted', got '{method}'")

        if self.method == 'weighted' and not weights:
            raise ValueError("weights must be provided when method='weighted'")
        
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match models ({len(models)})")
            if abs(sum(weights) - 1.0) > 1e-4:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.4f}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model(x) for model in self.models]
        stacked = torch.stack(predictions, dim=0)

        if self.method == 'average':
            return stacked.mean(dim=0)
        elif self.method == 'weighted' and self.weights is not None:
            shape = [-1] + [1] * (stacked.ndim - 1)
            weights_tensor = torch.tensor(self.weights, device=x.device, dtype=stacked.dtype).view(*shape)
            return (stacked * weights_tensor).sum(dim=0)
        
        # Fallback
        return stacked.mean(dim=0)
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [model(x) for model in self.models]


def build_model(
    model_name: str = "densenet", 
    num_classes: int = 4, 
    pretrained: bool = True,
    dropout: float = 0.4,
    drop_path_rate: Optional[float] = None,
    img_size: Optional[int] = None,
    classification_mode: str = "multilabel",
    pooling: str = "gem",
) -> nn.Module:
    """Factory function to build state-of-the-art architectures.
    
    Supports both multilabel and multiclass classification:
      - multilabel: classifier bias initialised to -2.0 so sigmoid outputs
        start low (~0.12), preventing dead gradients with Focal / BCE loss.
      - multiclass: default (zero) bias is kept — softmax is shift-invariant
        so the value has no effect, and zero is the cleaner convention.
    """
    model_name = model_name.lower().strip()
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {available}")
    
    metadata = MODEL_REGISTRY[model_name]
    timm_name = metadata["timm_name"]
    final_img_size = img_size or metadata["default_img_size"]

    logger.info(f"Building {timm_name} (pretrained={pretrained}, img_size={final_img_size})...")
    
    model_kwargs: Dict[str, Any] = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        # drop_rate MUST be 0 — the custom classifier head already
        # contains Dropout(dropout).  timm applies drop_rate as a
        # SECOND dropout in forward_head / ClassifierHead, giving an
        # effective ~64% drop rate that cripples learning.
        "drop_rate": 0.0,
    }
    
    # Heuristics for modern architectures utilizing drop_path_rate
    modern_archs = ['vit', 'swin', 'deit', 'convnext', 'xcit', 'cait']
    is_modern = any(x in model_name for x in modern_archs)
    
    if is_modern:
        # If not explicitly set, use a conservative fraction of the dropout rate
        model_kwargs["drop_path_rate"] = drop_path_rate if drop_path_rate is not None else (dropout * 0.1)
        logger.info(f"Using drop_path_rate={model_kwargs['drop_path_rate']} for {model_name}")
        
    # Certain timm models (ViT/Swin/DeiT) accept img_size on creation to interpolate positional embeddings natively
    if any(x in model_name for x in ['vit', 'swin', 'deit']):
        model_kwargs["img_size"] = final_img_size
    
    try:
        model = timm.create_model(timm_name, **model_kwargs)

        # Build a custom multi-layer classifier and swap it into the
        # existing head structure.  This preserves any pooling / norm
        # layers that timm wraps around the final Linear (critical for
        # ConvNeXt, Swin, etc. whose ClassifierHead does global-pool
        # before the fc layer).
        in_features = model.num_features
        custom_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        _replace_classifier(model, custom_fc)

        # Replace Global Average Pooling with GeM if requested
        if pooling == "gem":
            _replace_global_pool(model)
            logger.info(f"Replaced GAP with GeM pooling for {model_name}")

        # Resolve what TIMM expects vs what we asked for (Crucial for CNNs that don't take img_size on init)
        try:
            data_cfg = resolve_data_config(model.default_cfg, model=model)
            native_size = data_cfg.get('input_size', (3, 224, 224))[-1]
            if final_img_size != native_size:
                logger.warning(
                    f"Requested img_size {final_img_size} differs from native resolution {native_size}. "
                    "Ensure Datasets/Transforms resize the images appropriately."
                )
        except Exception as e:
            logger.debug(f"Could not resolve data config for {timm_name}: {e}")

    except Exception as e:
        logger.error(f"Failed to create {timm_name}: {e}")
        raise
    
    model.num_classes = num_classes
    model._img_size = final_img_size
    model._metadata = metadata

    # NOTE: Classifier bias is NOT initialised here.  The training module
    # (XClinVisionModel) applies the bias init AFTER the loss function is
    # known, because the optimal bias depends on the loss type:
    #   - BCE / BCEWithLogitsLoss: bias = -2.0  (sigmoid starts ~0.12,
    #     avoids dead gradients when most targets are 0).
    #   - Focal loss: bias = 0.0  (sigmoid starts at 0.5, ensuring
    #     equal focal weighting for positive and negative samples;
    #     bias=-2.0 would suppress 96% of the negative gradient).

    return model


def _replace_classifier(model: nn.Module, new_classifier: nn.Module) -> None:
    """Replace only the final classifier inside a timm model's head.

    Handles the different head layouts across timm architectures:
      - ClassifierHead with ``.fc``  (ConvNeXt, Swin, newer ViT)
      - Plain ``nn.Linear`` as ``model.head``  (older ViT / DeiT)
      - ``model.fc``  (ResNet)
      - ``model.classifier``  (EfficientNet, DenseNet)
    """
    if hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            model.head.fc = new_classifier
            return
        if isinstance(model.head, nn.Linear):
            model.head = new_classifier
            return
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = new_classifier
        return
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = new_classifier
        return
    raise RuntimeError(
        f"Could not locate classifier layer in {type(model).__name__}. "
        "Manual integration required."
    )


def _replace_global_pool(model: nn.Module) -> None:
    """Replace the global average pool in a timm model's head with GeM.

    Works with:
      - NormMlpClassifierHead (ConvNeXt, Swin) → ``head.global_pool``
      - ClassifierHead (EfficientNet) → ``head.global_pool``
      - ResNet → ``global_pool``
      - DenseNet → ``global_pool``
    """
    gem = GeM(p=3.0)

    # Timm head-based models (ConvNeXt, Swin, EfficientNet)
    if hasattr(model, 'head') and hasattr(model.head, 'global_pool'):
        model.head.global_pool = gem
        logger.info("Replaced head.global_pool with GeM")
        return

    # ResNet / DenseNet style
    if hasattr(model, 'global_pool'):
        model.global_pool = gem
        logger.info("Replaced model.global_pool with GeM")
        return

    logger.warning("Could not find global_pool to replace with GeM — using default pooling.")


def _init_classifier_bias(model: nn.Module, bias_value: float = -2.0) -> None:
    """Initialize the final classifier bias of the custom head."""
    last_linear = None

    # Locate the classifier module (may be a Sequential or a plain Linear)
    clf = None
    try:
        clf = model.get_classifier() if hasattr(model, 'get_classifier') else None
    except (AttributeError, TypeError):
        pass
    if clf is None:
        for attr in ('head', 'fc', 'classifier'):
            candidate = getattr(model, attr, None)
            if candidate is not None:
                clf = candidate
                break

    if clf is not None:
        if isinstance(clf, nn.Linear):
            last_linear = clf
        else:
            for sub in reversed(list(clf.modules())):
                if isinstance(sub, nn.Linear):
                    last_linear = sub
                    break

    if last_linear is not None and last_linear.bias is not None:
        nn.init.constant_(last_linear.bias, bias_value)
        logger.info(f"Initialized classifier bias to {bias_value}")
    else:
        logger.warning("Could not find the final linear layer bias to initialize.")


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


def get_model_normalization(model: nn.Module, model_name: Optional[str] = None) -> Dict[str, tuple]:
    """Retrieve the correct normalization mean/std for the architecture."""
    if model_name and model_name in CUSTOM_NORMS:
        return CUSTOM_NORMS[model_name]
        
    if hasattr(model, 'default_cfg'):
        return {
            "mean": model.default_cfg.get('mean', (0.485, 0.456, 0.406)),
            "std":  model.default_cfg.get('std', (0.229, 0.224, 0.225))
        }
    return {
        "mean": (0.485, 0.456, 0.406), 
        "std": (0.229, 0.224, 0.225)
        }


def freeze_backbone(model: nn.Module, unfreeze_head: bool = True) -> None:
    """Freeze all layers except the classification head."""
    if isinstance(model, EnsembleClassifier):
        for sub_model in model.models:
            freeze_backbone(sub_model, unfreeze_head=unfreeze_head)
        return

    # Freeze all 
    for param in model.parameters():
        param.requires_grad = False
    
    if not unfreeze_head:
        logger.info("Frozen entire model")
        return
    
    unfrozen = False

    # Unfreeze the entire head module (pooling + norm + classifier).
    # Walk the standard timm attribute names; the first one that carries
    # parameters is the head we want.
    for attr in ('head', 'fc', 'classifier'):
        head_module = getattr(model, attr, None)
        if head_module is not None and any(True for _ in head_module.parameters()):
            for param in head_module.parameters():
                param.requires_grad = True
            unfrozen = True
            logger.info("Unfroze head module (model.%s)", attr)
            break

    if not unfrozen:
        logger.warning("Could not reliably unfreeze classification head.")
    
    counts = get_param_counts(model)
    logger.info(f"Frozen backbone: {counts['frozen_m']:.2f}M frozen, {counts['trainable_m']:.2f}M trainable ({counts['trainable_pct']:.1f}%)")


def unfreeze_layers(model: nn.Module, num_layers: int = 0) -> None:
    """Gradually unfreeze layers for progressive fine-tuning.
    
    Args:
        model: The model to unfreeze layers in
        num_layers: Number of layer blocks to unfreeze from the end. 
                   0 means unfreeze all layers.
    """
    metadata = getattr(model, '_metadata', {})
    if not metadata.get('progressive', True):
        logger.warning("Model architecture not flagged for progressive unfreezing; proceeding anyway.")

    if num_layers == 0:
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all layers")
        return

    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")

    # First freeze everything, then unfreeze head + requested blocks
    freeze_backbone(model, unfreeze_head=True)

    # Collect all trainable blocks (excluding head)
    blocks_found = []
    
    # Look for common block naming patterns in timm models
    block_patterns = [
        'blocks', 'layers', 'stages',  # EfficientNet, ResNet, ConvNeXt
        'features',  # DenseNet
    ]
    
    found_blocks = False
    for pattern in block_patterns:
        if hasattr(model, pattern):
            module = getattr(model, pattern)
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                blocks_found.extend(list(module.children()))
                found_blocks = True
                logger.debug(f"Found blocks in model.{pattern}")
                break
            elif isinstance(module, nn.Module):
                # Try to get children if it's a container
                children = list(module.children())
                if children:
                    blocks_found.extend(children)
                    found_blocks = True
                    logger.debug(f"Found blocks in model.{pattern} children")
                    break
    
    # Fallback: try to find blocks anywhere in named modules
    if not found_blocks:
        for name, module in model.named_children():
            if name in ('head', 'fc', 'classifier', 'global_pool', 'norm'):
                continue
            # Check if this module has trainable parameters
            has_params = any(True for _ in module.parameters())
            if has_params:
                blocks_found.append(module)

    if not blocks_found:
        logger.warning("Could not identify layer blocks; unfreezing top-level children.")
        blocks_found = [m for n, m in model.named_children() 
                       if n not in ('head', 'fc', 'classifier', 'global_pool', 'norm')]

    # Filter to blocks with parameters
    valid_blocks = [b for b in blocks_found if any(True for _ in b.parameters())]
    
    if not valid_blocks:
        logger.warning("No valid blocks found to unfreeze")
        return

    # Unfreeze the last num_layers blocks
    blocks_to_unfreeze = valid_blocks[-num_layers:] if num_layers < len(valid_blocks) else valid_blocks

    unfrozen_count = 0
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            if param.requires_grad == False:
                param.requires_grad = True
                unfrozen_count += 1

    counts = get_param_counts(model)
    logger.info(
        f"Progressive unfreeze (last {len(blocks_to_unfreeze)} blocks, {unfrozen_count} newly unfrozen): "
        f"{counts['trainable_m']:.2f}M trainable ({counts['trainable_pct']:.1f}%)"
    )


def get_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Lookup layer for Grad-CAM reliably based on model metadata lookup.
    Properly parses index accesses like `stages[-1].blocks[-1]`.
    """
    metadata = getattr(model, '_metadata', None)
    if not metadata or 'gradcam_target' not in metadata:
        return get_last_conv_layer(model)
        
    target_str = metadata['gradcam_target']
    
    try:
        target = model
        parts = target_str.split('.')
        for part in parts:
            if not part: continue
            if '[' in part:
                attr, idx_str = part.split('[')
                idx = int(idx_str.rstrip(']'))

                if attr:
                    target = getattr(target, attr)[idx]
                else:
                    target = target[idx]
            else:
                target = getattr(target, part)
        return target
    except Exception as e:
        logger.debug(f"Metadata Grad-CAM path {target_str} resolution failed: {e}")
        return get_last_conv_layer(model)


def get_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


# =============================================================================
# TESTING
# =============================================================================
def _test_model(model_name: str, num_classes: int = 3, batch_size: int = 2, device: str = 'cpu') -> Dict:
    try:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
        model = model.to(device)
        model.eval()
        
        # Test 1: Shape / Native Execution
        img_size = getattr(model, '_img_size', 224) 
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, num_classes), f"Bad output shape: {output.shape}"
        
        # Test 2: GradCAM Path Resolution
        target = get_target_layer(model)
        assert target is not None, "GradCAM target resolution failed to find a valid module"

        # Test 3: Progressive / Head Freezing Behavior
        freeze_backbone(model, unfreeze_head=True)
        counts = get_param_counts(model)
        assert counts['trainable_pct'] < 10.0, f"Too many trainable params after head-only unfreeze: {counts['trainable_pct']:.1f}%"
        
        return {
            'name': model_name, 'status': '✓ PASS', 'output_shape': tuple(output.shape),
            'params_m': counts['total_m'], 'trainable_m': counts['trainable_m'],
            'target_type': type(target).__name__, 'error': None,
            'model_ref': model 
        }
    except Exception as e:
        return {
            'name': model_name, 'status': '✗ FAIL', 'output_shape': None,
            'params_m': 0, 'trainable_m': 0, 'target_type': 'N/A', 'error': str(e),
            'model_ref': None
        }


def _test_ensemble(models: List[nn.Module], method: str = 'average', weights: Optional[List[float]] = None) -> Dict:
    try:
        ensemble = EnsembleClassifier(models, method=method, weights=weights)
        device = next(models[0].parameters()).device
        img_size = getattr(models[0], '_img_size', 224)
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
    
    successful_models = [r['model_ref'] for r in results if r['status'] == '✓ PASS' and r['model_ref'] is not None][:2]
    
    if len(successful_models) >= 2:
        print(f"\n{'='*70}\nENSEMBLE TESTS\n{'='*70}")
        for method in ['average', 'weighted']:
            weights = [1.0 / len(successful_models)] * len(successful_models) if method == 'weighted' else None
            result = _test_ensemble(successful_models, method=method, weights=weights)
            status = "✓" if result['status'] == '✓ PASS' else "✗"
            print(f"{status} Ensemble ({method}): {result.get('output_shape', 'FAILED')}")
            if status == "✗":
                print(f"   Reason: {result['error']}")
    
    print(f"\n{'='*70}\n")

