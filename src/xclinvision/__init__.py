"""XClinVision: Explainable Medical Imaging AI Platform with Clinical Decision Support.

This package provides a comprehensive framework for medical image analysis,
combining explainable deep learning with RAG-enhanced LLM-based clinical
decision support.
"""

__version__ = "0.1.0"
__author__ = "XClinVision Team"
__license__ = "MIT"


def __getattr__(name: str):
    """Lazy imports so lightweight modules (e.g. processing) load without heavy deps."""
    if name == "build_model":
        from xclinvision.modeling import build_model
        return build_model
    if name == "predict":
        from xclinvision.inference import predict
        return predict
    if name == "generate_explanation":
        from xclinvision.xai import generate_explanation
        return generate_explanation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_model",
    "predict",
    "generate_explanation",
]
