"""Reliability analysis and robustness testing."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import pairwise_distances


class ReliabilityAnalyzer:
    """Analyze model reliability and robustness."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        
    def compute_feature_embeddings(
        self,
        images: np.ndarray,
    ) -> np.ndarray:
        """Extract feature embeddings from model."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for img in images:
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
                features = self.model.get_features(img_tensor)
                embeddings.append(features.cpu().numpy().flatten())
                
        return np.array(embeddings)
    
    def detect_out_of_distribution(
        self,
        test_embeddings: np.ndarray,
        train_embeddings: np.ndarray,
        threshold_percentile: float = 95.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect OOD samples using Mahalanobis distance."""
        # Compute mean and covariance of training embeddings
        train_mean = np.mean(train_embeddings, axis=0)
        train_cov = np.cov(train_embeddings.T)
        
        # Add small regularization for numerical stability
        train_cov += np.eye(train_cov.shape[0]) * 1e-6
        
        # Compute inverse covariance
        try:
            inv_cov = np.linalg.inv(train_cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(train_cov)
        
        # Compute Mahalanobis distances
        distances = []
        for emb in test_embeddings:
            diff = emb - train_mean
            dist = np.sqrt(diff @ inv_cov @ diff)
            distances.append(dist)
            
        distances = np.array(distances)
        
        # Determine threshold
        threshold = np.percentile(distances, threshold_percentile)
        
        # Flag OOD samples
        is_ood = distances > threshold
        
        return is_ood, distances
    
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
            
        # Cosine distance between means
        cosine_dist = 1 - np.dot(mu_train, mu_test) / (np.linalg.norm(mu_train) * np.linalg.norm(mu_test))
        
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
        perturbations: List[str] = None,
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
        import cv2
        
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
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ["Normal", "Pneumonia", "Tuberculosis"]
        
    def analyze_failures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        images: Optional[List] = None,
    ) -> Dict:
        """Analyze failure patterns."""
        failures = {
            "false_positives": {},
            "false_negatives": {},
            "high_confidence_errors": [],
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
