"""BiomedCLIP custom models and wrappers."""

import logging
from typing import Optional

import torch
import torch.nn as nn

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


logger = logging.getLogger(__name__)

class BiomedCLIPClassifier(nn.Module):
    """
    Custom wrapper for Microsoft's BiomedCLIP.
    Extracts the pre-trained Vision Transformer from the CLIP model
    and adds a linear classification head with ``num_classes`` outputs.
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

        # Bias init for the classifier head
        from .modeling import _init_classifier_bias
        _init_classifier_bias(self, bias_value=-2.0)

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
    
    REQUIRED_SIZE = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.REQUIRED_SIZE or x.shape[-2] != self.REQUIRED_SIZE:
            logger.warning(
                "BiomedCLIP received %dx%d input; resizing to %dx%d. "
                "Set input.size: [224, 224] in the model config to avoid "
                "lossy runtime interpolation.",
                x.shape[-2], x.shape[-1],
                self.REQUIRED_SIZE, self.REQUIRED_SIZE,
            )
            x = torch.nn.functional.interpolate(
                x, size=(self.REQUIRED_SIZE, self.REQUIRED_SIZE),
                mode='bicubic', align_corners=False,
            )
        features = self.vision_encoder(x)
        return self.head(features)
    
    def get_gradcam_target(self) -> Optional[nn.Module]:
        return self._target_layer
    




# """Model architectures for Chest X-ray Classification.
# modeling.py - Defines all TIMM model architectures, 
# ensemble logic, and advanced progressive unfreezing utilities.
# """

# import logging
# from typing import Dict, List, Optional, Union, Any

# import torch
# import torch.nn as nn
# import timm

# logger = logging.getLogger(__name__)

# # Model registry with architecture-specific metadata
# MODEL_REGISTRY = {
#     "densenet": {
#         "timm_name": "densenet121",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "features",
#     },
#     "resnet50": {
#         "timm_name": "resnet50",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "layer4",
#     },
#     "efficientnet_b0": {
#         "timm_name": "efficientnet_b0",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks",
#     },
#     "efficientnet_b2": {
#         "timm_name": "efficientnet_b2",
#         "default_img_size": 260,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks",
#     },
#     "efficientnet_b3": {
#         "timm_name": "efficientnet_b3",
#         "default_img_size": 300,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks",
#     },
#     "efficientnet_b4": {
#         "timm_name": "efficientnet_b4",
#         "default_img_size": 380,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks",
#     },
#     "convnext_tiny": {
#         "timm_name": "convnext_tiny",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "stages[-1].blocks[-1]",
#     },
#     "convnext_small": {
#         "timm_name": "convnext_small",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "stages[-1].blocks[-1]",
#     },
#     "vit_tiny": {
#         "timm_name": "vit_tiny_patch16_224",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks[-1].attn.qkv",
#     },
#     "vit_small": {
#         "timm_name": "vit_small_patch16_224",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks[-1].attn.qkv",
#     },
#     "vit_base": {
#         "timm_name": "vit_base_patch16_224",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "blocks[-1].attn.qkv",
#     },
#     "swin_t": {
#         "timm_name": "swin_tiny_patch4_window7_224",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "layers[-1].blocks[-1].attn.qkv",
#     },
#     "swin_s": {
#         "timm_name": "swin_small_patch4_window7_224",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "layers[-1].blocks[-1].attn.qkv",
#     },
#     "swin_b": {
#         "timm_name": "swin_base_patch4_window7_224",
#         "default_img_size": 224,
#         "head_type": "linear",
#         "progressive": True,
#         "gradcam_target": "layers[-1].blocks[-1].attn.qkv",
#     },
# }


# class EnsembleClassifier(nn.Module):
#     """
#     Ensemble of multiple models for robust predictions.
#     Outputs logits for consistent loss and metric computation.
#     """
#     def __init__(self, models: List[nn.Module], method: str = 'average', 
#                  weights: Optional[List[float]] = None):
#         super().__init__()
#         if not models:
#             raise ValueError("At least one model required for ensemble")
        
#         self.models = nn.ModuleList(models)
#         self.method = method.lower()
#         self.weights = weights
        
#         if self.method not in ['average', 'weighted']:
#             raise ValueError(f"Method must be 'average' or 'weighted', got '{method}'")

#         if self.method == 'weighted' and not weights:
#             raise ValueError("weights must be provided when method='weighted'")
        
#         if weights is not None:
#             if len(weights) != len(models):
#                 raise ValueError(f"Number of weights ({len(weights)}) must match models ({len(models)})")
#             if abs(sum(weights) - 1.0) > 1e-4:
#                 raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.4f}")
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         predictions = [model(x) for model in self.models]
#         stacked = torch.stack(predictions, dim=0)

#         if self.method == 'average':
#             return stacked.mean(dim=0)
#         elif self.method == 'weighted' and self.weights is not None:
#             # Safe reshaping allowing broadcasting irrespective of output rank
#             shape = [-1] + [1] * (stacked.ndim - 1)
#             weights_tensor = torch.tensor(self.weights, device=x.device, dtype=stacked.dtype).view(*shape)
#             return (stacked * weights_tensor).sum(dim=0)
        
#         # Fallback
#         return stacked.mean(dim=0)
    
#     def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
#         return [model(x) for model in self.models]


# def build_model(
#     model_name: str = "densenet", 
#     num_classes: int = 3, 
#     pretrained: bool = True,
#     dropout: float = 0.2,
#     drop_path_rate: Optional[float] = None,
#     img_size: Optional[int] = None,
#     classification_mode: str = "multilabel",
# ) -> nn.Module:
#     """Factory function to build state-of-the-art architectures.
    
#     Supports both multilabel and multiclass classification:
#       - multilabel: classifier bias initialised to -2.0 so sigmoid outputs
#         start low (~0.12), preventing dead gradients with Focal / BCE loss.
#       - multiclass: default (zero) bias is kept — softmax is shift-invariant
#         so the value has no effect, and zero is the cleaner convention.

#     Input resolution is made explicit: img_size overrides model defaults.
#     """
#     model_name = model_name.lower().strip()
    
#     if model_name not in MODEL_REGISTRY:
#         available = list(MODEL_REGISTRY.keys())
#         raise ValueError(f"Model '{model_name}' not supported. Choose from: {available}")
    
#     metadata = MODEL_REGISTRY[model_name]
#     timm_name = metadata["timm_name"]
#     final_img_size = img_size or metadata["default_img_size"]

#     logger.info(f"Building {timm_name} (pretrained={pretrained}, img_size={final_img_size})...")
    
#     model_kwargs: Dict[str, Any] = {
#         "pretrained": pretrained,
#         "num_classes": num_classes,
#         "drop_rate": dropout,
#     }
    
#     is_transformer = any(x in model_name for x in ['vit', 'swin', 'deit', 'convnext'])
#     if is_transformer:
#         if drop_path_rate is None:
#             drop_path_rate = 0.1
#         model_kwargs["drop_path_rate"] = drop_path_rate
#         logger.info(f"Using drop_path_rate={drop_path_rate} for {model_name}")
        
#     # Some timm models accept img_size on creation
#     if any(x in model_name for x in ['vit', 'swin', 'deit']):
#         model_kwargs["img_size"] = final_img_size
    
#     try:
#         model = timm.create_model(timm_name, **model_kwargs)
#     except Exception as e:
#         logger.error(f"Failed to create {timm_name}: {e}")
#         raise
    
#     model.num_classes = num_classes
#     model._img_size = final_img_size
#     model._metadata = metadata

#     if classification_mode == "multilabel":
#         _init_classifier_bias(model, bias_value=-2.0)
#         logger.info("Applied multilabel bias init (-2.0) to classifier head")
#     else:
#         logger.info("Multiclass mode: using default classifier bias (0.0)")

#     return model


# def _init_classifier_bias(model: nn.Module, bias_value: float = -2.0) -> None:
#     """Initialize the final classifier bias to a negative value.

#     Intended for **multilabel** (sigmoid) classification:
#     sigmoid(-2.0) ≈ 0.12, so the model starts with low positive
#     predictions — critical for imbalanced data and Focal Loss where
#     near-1.0 initial sigmoid outputs collapse (1 - pt)^gamma to zero.

#     For **multiclass** (softmax) this function should be skipped:
#     softmax is shift-invariant so a uniform bias has no effect, but
#     skipping keeps the intent explicit.
#     """
#     last_linear = None

#     # Use timm's get_classifier() which correctly points to the task head
#     if hasattr(model, "get_classifier") and callable(model.get_classifier):
#         clf = model.get_classifier()
#         if isinstance(clf, nn.Linear):
#             last_linear = clf
#         elif isinstance(clf, (nn.Sequential, nn.ModuleList)):
#             for sub in reversed(list(clf.modules())):
#                 if isinstance(sub, nn.Linear) and sub.out_features == getattr(model, 'num_classes', 0):
#                     last_linear = sub
#                     break
#             # Fallback if num_classes doesn't exactly match or isn't set yet
#             if last_linear is None:
#                 for sub in reversed(list(clf.modules())):
#                     if isinstance(sub, nn.Linear):
#                         last_linear = sub
#                         break

#     if last_linear is not None and last_linear.bias is not None:
#         nn.init.constant_(last_linear.bias, bias_value)
#         logger.info(
#             f"Initialized classifier bias to {bias_value} "
#             f"(layer: {last_linear.__class__.__name__}, "
#             f"out_features={last_linear.out_features})"
#         )
#     else:
#         logger.warning(
#             "Could not find or reliably set the final linear layer bias initialization."
#         )


# def get_param_counts(model: nn.Module) -> Dict[str, Union[int, float]]:
#     """Get parameter counts for model analysis."""
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     frozen = total - trainable
    
#     return {
#         'total': total,
#         'trainable': trainable,
#         'frozen': frozen,
#         'trainable_pct': 100.0 * trainable / total if total > 0 else 0.0,
#         'frozen_pct': 100.0 * frozen / total if total > 0 else 0.0,
#         'total_m': total / 1e6,
#         'trainable_m': trainable / 1e6,
#         'frozen_m': frozen / 1e6,
#     }


# def get_model_normalization(model: nn.Module, model_name: Optional[str] = None) -> Dict[str, tuple]:
#     """Retrieve the correct normalization mean/std for the architecture.

#     Args:
#         model: The model instance (checked for ``default_cfg``).
#         model_name: Optional architecture key — reserved for future
#             per-architecture overrides; currently unused.
#     """
#     if hasattr(model, 'default_cfg'):
#         return {
#             "mean": model.default_cfg.get('mean', (0.485, 0.456, 0.406)),
#             "std":  model.default_cfg.get('std', (0.229, 0.224, 0.225))
#         }
#     # Fallback standard ImageNet
#     return {"mean": (0.485, 0.456, 0.406), 
#             "std": (0.229, 0.224, 0.225)}


# def freeze_backbone(model: nn.Module, unfreeze_head: bool = True) -> None:
#     """
#     Freeze all layers except the classification head.
#     More robustly accesses timm's generic accessors.
#     """
#     if isinstance(model, EnsembleClassifier):
#         for sub_model in model.models:
#             freeze_backbone(sub_model, unfreeze_head=unfreeze_head)
#         return

#     # Freeze all 
#     for param in model.parameters():
#         param.requires_grad = False
    
#     if not unfreeze_head:
#         logger.info("Frozen entire model")
#         return
    
#     unfrozen = False
    
#     if hasattr(model, 'get_classifier') and callable(model.get_classifier):
#         classifier = model.get_classifier()
#         if classifier is not None:
#             for param in classifier.parameters(): param.requires_grad = True
#             unfrozen = True
#             logger.info("Unfroze backbone using model.get_classifier()")
            
#     if not unfrozen:
#         logger.warning("Could not reliably unfreeze classification head using generic methods.")
    
#     counts = get_param_counts(model)
#     logger.info(f"Frozen backbone: {counts['frozen_m']:.2f}M frozen, {counts['trainable_m']:.2f}M trainable ({counts['trainable_pct']:.1f}%)")


# def unfreeze_layers(model: nn.Module, num_layers: int = 0) -> None:
#     """
#     Gradually unfreeze layers for progressive fine-tuning.
#     Instead of guessing properties, relies dynamically on named modules backwards
#     to find sequential blocks that act as feature extractors.
#     """
#     if num_layers == 0:
#         for param in model.parameters():
#             param.requires_grad = True
#         logger.info("Unfrozen all layers")
#         return
    
#     if num_layers < 0:
#         raise ValueError(f"num_layers must be >= 0, got {num_layers}")
    
#     freeze_backbone(model, unfreeze_head=True)
    
#     blocks_found = []
    
#     for name, module in model.named_children():
#         if name in ('head', 'fc', 'classifier'):
#             continue # already handled
            
#         if isinstance(module, (nn.Sequential, nn.ModuleList)):
#             # If it's a sequence wrapper, grab its direct children backwards
#             blocks_found.extend(list(module.children()))
#         elif len(list(module.children())) > 0:
#             blocks_found.append(module)
            
#     if not blocks_found:
#         logger.warning("Model architecture not easily splittable for blocks; unfreezing top layers sequentially.")
#         blocks_found = list(model.children())[:-1] # Exclude head

#     # Filter out layers that do not possess parameters
#     valid_blocks = [b for b in blocks_found if sum(1 for p in b.parameters() if p.requires_grad is not None) > 0]
#     blocks_to_unfreeze = valid_blocks[-num_layers:] if num_layers < len(valid_blocks) else valid_blocks
    
#     for block in blocks_to_unfreeze:
#         for param in block.parameters():
#             param.requires_grad = True
            
#     counts = get_param_counts(model)
#     logger.info(f"Progressive unfreeze (last {num_layers} layer structures): {counts['trainable_m']:.2f}M trainable ({counts['trainable_pct']:.1f}%)")


# def get_target_layer(model: nn.Module) -> Optional[nn.Module]:
#     """
#     Lookup layer for Grad-CAM reliably based on model metadata lookup.
#     Provided as best-effort since certain ViTs / nested wrappers are complex.
#     """
#     metadata = getattr(model, '_metadata', None)
#     if not metadata or 'gradcam_target' not in metadata:
#         return get_last_conv_layer(model)
        
#     target_str = metadata['gradcam_target']
    
#     # Attempt to eval string paths safely
#     try:
#         target = model
#         for segment in target_str.replace(']', '').split('['):
#             for part in segment.split('.'):
#                 if not part: continue
#                 # Is it an index?
#                 if part.lstrip('-').isdigit():
#                     target = target[int(part)]
#                 else:
#                     target = getattr(target, part)
#         return target
#     except Exception as e:
#         logger.debug(f"Metadata Grad-CAM path {target_str} resolution failed: {e}")
#         return get_last_conv_layer(model)

# def get_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
#     last_conv = None
#     for module in model.modules():
#         if isinstance(module, nn.Conv2d):
#             last_conv = module
#     return last_conv


# # =============================================================================
# # TESTING
# # =============================================================================
# def _test_model(model_name: str, num_classes: int = 3, batch_size: int = 2, device: str = 'cpu') -> Dict:
#     try:
#         model = build_model(model_name, num_classes=num_classes, pretrained=False)
#         model = model.to(device)
#         model.eval()
        
#         img_size = getattr(model, '_img_size', 224) # Fallback if missing
#         dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
#         with torch.no_grad():
#             output = model(dummy_input)
        
#         assert output.shape == (batch_size, num_classes), f"Bad output shape: {output.shape}"
        
#         counts = get_param_counts(model)
#         target = get_target_layer(model)
        
#         return {
#             'name': model_name, 'status': '✓ PASS', 'output_shape': tuple(output.shape),
#             'params_m': counts['total_m'], 'trainable_m': counts['trainable_m'],
#             'target_type': type(target).__name__ if target else 'None', 'error': None,
#             'model_ref': model # pass reference to use for ensemble tests
#         }
#     except Exception as e:
#         return {
#             'name': model_name, 'status': '✗ FAIL', 'output_shape': None,
#             'params_m': 0, 'trainable_m': 0, 'target_type': 'N/A', 'error': str(e),
#             'model_ref': None
#         }


# def _test_ensemble(models: List[nn.Module], method: str = 'average', weights: Optional[List[float]] = None) -> Dict:
#     try:
#         ensemble = EnsembleClassifier(models, method=method, weights=weights)
#         device = next(models[0].parameters()).device
#         img_size = getattr(models[0], '_img_size', 224)
#         dummy_input = torch.randn(2, 3, img_size, img_size, device=device)
        
#         with torch.no_grad():
#             output = ensemble(dummy_input)
        
#         return {'method': method, 'status': '✓ PASS', 'output_shape': tuple(output.shape), 'n_models': len(models), 'error': None}
#     except Exception as e:
#         return {'method': method, 'status': '✗ FAIL', 'output_shape': None, 'n_models': len(models), 'error': str(e)}


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    
#     TEST_MODELS = [
#         "densenet", "resnet50", "efficientnet_b0", "efficientnet_b2",
#         "convnext_tiny", "vit_small", "swin_t",
#     ]
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"\n{'='*70}\nMODEL TESTING SUITE | Device: {device}\n{'='*70}")
    
#     results = []
#     for name in TEST_MODELS:
#         print(f"\nTesting: {name.upper()}")
#         result = _test_model(name, device=device)
#         results.append(result)
#         if result['status'] == '✓ PASS':
#             print(f"  Output: {result['output_shape']}\n  Params: {result['params_m']:.2f}M (trainable: {result['trainable_m']:.2f}M)\n  XAI target: {result['target_type']}")
#         else:
#             print(f"  ERROR: {result['error']}")
    
#     print(f"\n{'='*70}\nSUMMARY\n{'='*70}\n{'Model':<20} {'Status':<8} {'Params':<12} {'XAI Target'}\n{'-'*70}")
#     for r in results:
#         params = f"{r['params_m']:.2f}M" if r['params_m'] > 0 else "N/A"
#         print(f"{r['name']:<20} {r['status']:<8} {params:<12} {r['target_type']}")
    
#     # Reuse tested model instances instead of rebuilding them (avoid discrepancy bugs)
#     successful_models = [r['model_ref'] for r in results if r['status'] == '✓ PASS' and r['model_ref'] is not None][:2]
    
#     if len(successful_models) >= 2:
#         print(f"\n{'='*70}\nENSEMBLE TESTS\n{'='*70}")
#         # Note: vote was explicitly removed and simplified to strictly linear logs/logits
#         for method in ['average', 'weighted']:
#             weights = [1.0 / len(successful_models)] * len(successful_models) if method == 'weighted' else None
#             result = _test_ensemble(successful_models, method=method, weights=weights)
#             status = "✓" if result['status'] == '✓ PASS' else "✗"
#             print(f"{status} Ensemble ({method}): {result.get('output_shape', 'FAILED')}")
#             if status == "✗":
#                 print(f"   Reason: {result['error']}")
    
#     print(f"\n{'='*70}")
