"""Reliability analysis and robustness testing."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import cv2
from xclinvision.dataset import get_val_transforms


class ReliabilityAnalyzer:
    """Analyze model reliability and robustness."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        
    def compute_feature_embeddings(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Extract feature embeddings from model."""
        self.model.eval()
        embeddings = []
        
        # Determine image size from model if possible, default to 224
        img_size = 224
        if hasattr(self.model, "default_cfg") and "input_size" in self.model.default_cfg:
            img_size = self.model.default_cfg["input_size"][-1]
            
        transform = get_val_transforms(image_size=img_size)
        
        if isinstance(images, np.ndarray):
            # If it's a single image, wrap it in a list. If it's a batch, list(images) will work too.
            if images.ndim == 3 or images.ndim == 2:
                images = [images]
            else:
                images = list(images)
                
        with torch.no_grad():
            for img in images:
                # Ensure input is an RGB image (H, W, 3) 
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    
                # Apply normalization and shape alignment [C, H, W]
                transformed = transform(image=img)
                img_tensor = transformed["image"].unsqueeze(0).to(self.device).float()
                
                # Extract features safely depending on the model's architecture
                if hasattr(self.model, "vision_encoder"):
                    # BiomedCLIP specific
                    features = self.model.vision_encoder(img_tensor)
                elif hasattr(self.model, "forward_features"):
                    # Native timm models
                    features = self.model.forward_features(img_tensor)
                    if hasattr(self.model, "global_pool") and features.ndim > 2:
                        features = self.model.global_pool(features)
                    elif features.ndim > 2:
                        features = features.mean(dim=[-2, -1]) if features.ndim == 4 else features.mean(dim=1)
                elif hasattr(self.model, "get_features"):
                    # General get_features method
                    features = self.model.get_features(img_tensor)
                else:
                    # Fallback (returns output from base forward if no better method is found)
                    features = self.model(img_tensor)
                    if features.ndim > 2:
                        features = features.mean(dim=[-2, -1]) if features.ndim == 4 else features.mean(dim=1)
                        
                embeddings.append(features.cpu().numpy().flatten())
                
        return np.array(embeddings)
    
    def detect_out_of_distribution(
        self,
        test_embeddings: np.ndarray,
        train_embeddings: np.ndarray,
        threshold_percentile: float = 95.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect OOD samples using Mahalanobis distance.

        The decision threshold is derived from *training* distances so it
        reflects the learned distribution rather than the test set itself.
        """
        # Compute mean and covariance of training embeddings
        train_mean = np.mean(train_embeddings, axis=0)
        train_cov = np.cov(train_embeddings.T)
        
        # H-3 fix: 1e-6 is far too small for high-dimensional embeddings (e.g.
        # 768-dim ViT/Swin features) where the covariance matrix is rank-deficient
        # when n_samples < feature_dim.  Use 1e-2 to ensure stable inversion.
        train_cov += np.eye(train_cov.shape[0]) * 1e-2
        
        # Compute inverse covariance
        try:
            # Prefer pseudo-inverse: robust to rank-deficiency even after regularization
            inv_cov = np.linalg.pinv(train_cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(train_cov.shape[0])
            logger.warning("Covariance inversion failed; using identity matrix (OOD distances are unreliable).")
        
        def _mahalanobis(embeddings: np.ndarray) -> np.ndarray:
            diffs = embeddings - train_mean
            # Vectorised: sqrt(diag(diffs @ inv_cov @ diffs.T))
            return np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))

        # Threshold derived from the training set
        train_distances = _mahalanobis(train_embeddings)
        threshold = np.percentile(train_distances, threshold_percentile)

        # Compute test distances and flag OOD samples
        test_distances = _mahalanobis(test_embeddings)
        is_ood = test_distances > threshold
        
        return is_ood, test_distances
    
    def compute_embedding_drift(
        self,
        test_embeddings: np.ndarray,
        train_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """Compute drift between training and test embeddings."""
        # Frechet Inception Distance (FID-like)
        mu_train = np.mean(train_embeddings, axis=0)
        sigma_train = np.cov(train_embeddings.T)
        
        mu_test = np.mean(test_embeddings, axis=0)
        sigma_test = np.cov(test_embeddings.T)
        
        # Compute FID
        diff = mu_train - mu_test
        covmean = self._sqrtm(sigma_train @ sigma_test)
        
        if np.isnan(covmean).any():
            fid = np.sum(diff ** 2)
        else:
            fid = np.sum(diff ** 2) + np.trace(sigma_train + sigma_test - 2 * covmean)
            
        # Cosine distance between means (guarded against zero-norm)
        norm_product = np.linalg.norm(mu_train) * np.linalg.norm(mu_test)
        if norm_product == 0:
            cosine_dist = 1.0  # maximally dissimilar when a mean is zero
        else:
            cosine_dist = 1 - np.dot(mu_train, mu_test) / norm_product
        
        return {
            "frechet_distance": float(fid),
            "cosine_distance": float(cosine_dist),
            "mean_l2_distance": float(np.linalg.norm(diff)),
        }
    
    def _sqrtm(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    
    def test_robustness(
        self,
        image: np.ndarray,
        pipeline,
        perturbations: Optional[List[str]] = None,
    ) -> Dict:
        """Test model robustness to various perturbations."""
        if perturbations is None:
            perturbations = ["noise", "blur", "brightness", "contrast"]
            
        results = {}
        
        # Original prediction
        orig_result = pipeline.predict(image)
        orig_pred = orig_result["prediction"]
        orig_conf = orig_result["confidence"]
        
        results["original"] = {
            "prediction": orig_pred,
            "confidence": orig_conf,
        }
        
        # Test each perturbation
        for perturb in perturbations:
            perturbed = self._apply_perturbation(image, perturb)
            pert_result = pipeline.predict(perturbed)
            
            results[perturb] = {
                "prediction": pert_result["prediction"],
                "confidence": pert_result["confidence"],
                "prediction_changed": pert_result["prediction"] != orig_pred,
                "confidence_drop": orig_conf - pert_result["confidence"],
            }
            
        return results
    
    def _apply_perturbation(
        self,
        image: np.ndarray,
        perturbation: str,
        severity: float = 0.1,
    ) -> np.ndarray:
        """Apply perturbation to image."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for robustness perturbations. "
                "Install it with: pip install opencv-python"
            ) from exc
        
        perturbed = image.copy()
        
        if perturbation == "noise":
            noise = np.random.normal(0, severity * 255, image.shape)
            perturbed = np.clip(image + noise, 0, 255).astype(np.uint8)
            
        elif perturbation == "blur":
            kernel_size = int(5 + severity * 10) // 2 * 2 + 1
            perturbed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif perturbation == "brightness":
            perturbed = np.clip(image * (1 + severity), 0, 255).astype(np.uint8)
            
        elif perturbation == "contrast":
            mean = np.mean(image)
            perturbed = np.clip((image - mean) * (1 + severity) + mean, 0, 255).astype(np.uint8)
            
        return perturbed


class FailureAnalyzer:
    """Analyze model failure modes."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["Normal", "Pneumonia", "Tuberculosis"]
        
    def analyze_failures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict:
        """Analyze failure patterns."""
        failures = {
            "false_positives": {},
            "false_negatives": {},
            "high_confidence_errors": {},
        }
        
        for i, class_name in enumerate(self.class_names):
            # False positives
            fp_mask = (y_pred == i) & (y_true != i)
            failures["false_positives"][class_name] = {
                "count": int(np.sum(fp_mask)),
                "indices": np.where(fp_mask)[0].tolist(),
            }
            
            # False negatives
            fn_mask = (y_true == i) & (y_pred != i)
            failures["false_negatives"][class_name] = {
                "count": int(np.sum(fn_mask)),
                "indices": np.where(fn_mask)[0].tolist(),
            }
            
        # High confidence errors
        max_probs = np.max(y_probs, axis=1)
        incorrect = y_pred != y_true
        high_conf_incorrect = incorrect & (max_probs > 0.8)
        
        failures["high_confidence_errors"] = {
            "count": int(np.sum(high_conf_incorrect)),
            "indices": np.where(high_conf_incorrect)[0].tolist(),
            "avg_confidence": float(np.mean(max_probs[high_conf_incorrect])) if np.sum(high_conf_incorrect) > 0 else 0.0,
        }
        
        return failures
