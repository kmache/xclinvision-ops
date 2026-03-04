"""MLOps monitoring and experiment tracking."""

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelRegistry:
    """Manage model versions and staging."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri
        
        if MLFLOW_AVAILABLE and tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Register a model in the registry."""
        if MLFLOW_AVAILABLE:
            result = mlflow.register_model(model_uri, name)
            
            # Add tags if provided
            if tags:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                version = result.version
                
                for key, value in tags.items():
                    client.set_model_version_tag(name, version, key, value)
                    
            return result
        return None
        
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: str,  # "Staging", "Production", "Archived"
    ):
        """Transition model to a new stage.

        Note: transition_model_version_stage is deprecated in MLflow >= 2.0.
        For MLflow >= 2.0, prefer using aliases:
            client.set_registered_model_alias(name, alias, version)
        """
        if MLFLOW_AVAILABLE:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            try:
                # MLflow >= 2.0 alias-based approach
                alias = stage.lower()  # e.g. "production", "staging"
                client.set_registered_model_alias(name, alias, str(version))
            except AttributeError:
                # Fallback for older MLflow versions
                client.transition_model_version_stage(name, version, stage)


class PredictionLogger:
    """Log predictions for audit and monitoring."""
    
    def __init__(self, log_dir: str = "./logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(
        self,
        image_hash: str,
        prediction: int,
        probabilities: List[float],
        confidence: float,
        uncertainty: Optional[Dict] = None,
        model_version: str = "unknown",
        timestamp: Optional[str] = None,
    ):
        """Log a single prediction."""
        # Use a single now() call to avoid a midnight race condition between
        # the timestamp string and the daily log filename.
        now = datetime.now()
        if timestamp is None:
            timestamp = now.isoformat()

        entry = {
            "timestamp": timestamp,
            "image_hash": image_hash,
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "model_version": model_version,
        }

        # Append to daily log file
        date_str = now.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"predictions_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def get_prediction_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve prediction history."""
        predictions = []
        
        for log_file in sorted(self.log_dir.glob("predictions_*.jsonl")):
            with open(log_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    
                    # Filter by date if specified
                    if start_date and entry["timestamp"] < start_date:
                        continue
                    if end_date and entry["timestamp"] > end_date:
                        continue
                        
                    predictions.append(entry)
                    
        return predictions


class DriftDetector:
    """Detect feature-space drift using a two-sample Kolmogorov-Smirnov test.

    Pass *feature embeddings* (not raw pixels) for meaningful semantic drift
    detection.  Use ReliabilityAnalyzer.compute_feature_embeddings() to
    extract them before calling set_baseline / detect_drift.
    """

    def __init__(self, alert_threshold: float = 0.05):
        self.alert_threshold = alert_threshold
        self._baseline_samples: Optional[Any] = None  # stores actual baseline array

    def set_baseline(self, reference_data: Any) -> None:
        """Store reference feature embeddings as the drift baseline.

        Args:
            reference_data: 2-D numpy array of shape (N, feature_dim).
        """
        import numpy as np

        if not isinstance(reference_data, np.ndarray):
            raise TypeError(
                f"reference_data must be a numpy array, got {type(reference_data).__name__}"
            )
        if reference_data.ndim != 2:
            raise ValueError(
                f"reference_data must be 2-D (N, feature_dim), got shape {reference_data.shape}"
            )
        self._baseline_samples = reference_data.copy()

    def detect_drift(self, new_data: Any) -> Dict[str, Any]:
        """Run a per-feature two-sample KS test between baseline and new data.

        Args:
            new_data: 2-D numpy array of shape (M, feature_dim).

        Returns:
            Dict with keys: drift_detected, drift_score, threshold,
            per_feature_ks, per_feature_pvalue.
        """
        import numpy as np
        from scipy import stats

        if self._baseline_samples is None:
            raise RuntimeError("Baseline not set — call set_baseline() first")

        if not isinstance(new_data, np.ndarray) or new_data.ndim != 2:
            raise TypeError("new_data must be a 2-D numpy array")

        if new_data.shape[1] != self._baseline_samples.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: baseline has {self._baseline_samples.shape[1]} "
                f"features but new_data has {new_data.shape[1]}"
            )

        n_features = self._baseline_samples.shape[1]
        ks_stats: List[float] = []
        p_values: List[float] = []

        for i in range(n_features):
            ks_stat, p_value = stats.ks_2samp(
                self._baseline_samples[:, i], new_data[:, i]
            )
            ks_stats.append(float(ks_stat))
            p_values.append(float(p_value))

        avg_drift = float(np.mean(ks_stats))

        return {
            "drift_detected": avg_drift > self.alert_threshold,
            "drift_score": avg_drift,
            "threshold": self.alert_threshold,
            "per_feature_ks": ks_stats,
            "per_feature_pvalue": p_values,
        }


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: Deque[Dict] = deque(maxlen=window_size)
        
    def add_prediction(
        self,
        prediction: int,
        ground_truth: Optional[int] = None,
        confidence: Optional[float] = None,
    ):
        """Add a prediction to the monitoring window."""
        self.predictions.append({
            "prediction": prediction,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })
        # deque(maxlen=window_size) evicts the oldest entry automatically
            
    def get_window_metrics(self) -> Dict[str, Any]:
        """Compute metrics for the current window."""
        if not self.predictions:
            return {}
            
        # Average confidence
        confidences = [p["confidence"] for p in self.predictions if p["confidence"] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Prediction distribution
        predictions = [p["prediction"] for p in self.predictions]
        distribution = Counter(predictions)
        
        metrics = {
            "avg_confidence": avg_confidence,
            "prediction_count": len(predictions),
            "class_distribution": dict(distribution),
        }
        
        # Accuracy if ground truth available
        labeled = [(p["prediction"], p["ground_truth"]) for p in self.predictions if p["ground_truth"] is not None]
        if labeled:
            correct = sum(1 for pred, gt in labeled if pred == gt)
            metrics["accuracy"] = correct / len(labeled)
            
        return metrics
