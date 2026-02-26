"""MLOps monitoring and experiment tracking."""

from typing import Dict, List, Optional, Any
import os
import json
from datetime import datetime
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Track experiments with MLflow."""
    
    def __init__(
        self,
        experiment_name: str = "xclinvision",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.tags = tags or {}
        
        if MLFLOW_AVAILABLE:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        else:
            print("Warning: MLflow not available. Logging to local files only.")
            
    def start_run(self, run_name: Optional[str] = None):
        """Start a new experiment run."""
        if MLFLOW_AVAILABLE:
            return mlflow.start_run(run_name=run_name)
        return None
        
    def end_run(self):
        """End current run."""
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if MLFLOW_AVAILABLE:
            for key, value in params.items():
                mlflow.log_param(key, value)
                
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
                
    def log_model(self, model, artifact_path: str = "model"):
        """Log model artifact."""
        if MLFLOW_AVAILABLE:
            mlflow.pytorch.log_model(model, artifact_path)
            
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log file artifact."""
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(local_path, artifact_path)


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
        stage: str,  # Staging, Production, Archived
    ):
        """Transition model to a new stage."""
        if MLFLOW_AVAILABLE:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
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
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
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
        date_str = datetime.now().strftime("%Y-%m-%d")
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
    """Detect data and performance drift."""
    
    def __init__(self, alert_threshold: float = 0.05):
        self.alert_threshold = alert_threshold
        self.baseline_stats = None
        
    def set_baseline(self, reference_data: Any):
        """Set baseline distribution from reference data."""
        import numpy as np
        
        if isinstance(reference_data, np.ndarray):
            self.baseline_stats = {
                "mean": np.mean(reference_data, axis=0),
                "std": np.std(reference_data, axis=0),
                "percentiles": np.percentile(reference_data, [10, 25, 50, 75, 90], axis=0),
            }
            
    def detect_drift(self, new_data: Any) -> Dict[str, Any]:
        """Detect drift in new data."""
        import numpy as np
        from scipy import stats
        
        if self.baseline_stats is None:
            return {"error": "Baseline not set"}
            
        if isinstance(new_data, np.ndarray):
            # Compute KS test for each feature
            drift_scores = []
            
            for i in range(new_data.shape[1]):
                # Two-sample KS test
                baseline_samples = np.random.normal(
                    self.baseline_stats["mean"][i],
                    self.baseline_stats["std"][i],
                    1000,
                )
                ks_stat, p_value = stats.ks_2samp(baseline_samples, new_data[:, i])
                drift_scores.append(ks_stat)
                
            avg_drift = np.mean(drift_scores)
            
            return {
                "drift_detected": avg_drift > self.alert_threshold,
                "drift_score": float(avg_drift),
                "threshold": self.alert_threshold,
            }
            
        return {"error": "Unsupported data type"}


class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = []
        self.ground_truth = []
        
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
        
        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            
    def get_window_metrics(self) -> Dict[str, float]:
        """Compute metrics for the current window."""
        if not self.predictions:
            return {}
            
        # Average confidence
        confidences = [p["confidence"] for p in self.predictions if p["confidence"] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Prediction distribution
        predictions = [p["prediction"] for p in self.predictions]
        from collections import Counter
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
