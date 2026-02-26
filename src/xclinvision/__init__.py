"""XClinVision: Explainable Medical Imaging AI Platform with Clinical Decision Support.

This package provides a comprehensive framework for medical image analysis,
combining explainable deep learning with RAG-enhanced LLM-based clinical
decision support.
"""

__version__ = "0.1.0"
__author__ = "XClinVision Team"
__license__ = "MIT"

from xclinvision.architecture import create_model
from xclinvision.inference import predict
from xclinvision.xai import generate_explanation

__all__ = [
    "create_model",
    "predict",
    "generate_explanation",
]
