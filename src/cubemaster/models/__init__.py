"""Model architectures for Rubik's Cube color classification."""

from .base import BaseColorClassifier
from .mlp import MLPClassifier
from .shallow_cnn import ShallowCNNClassifier
from .mobilenet import MobileNetV3Classifier

MODEL_REGISTRY = {
    "mlp": MLPClassifier,
    "shallow_cnn": ShallowCNNClassifier,
    "mobilenet": MobileNetV3Classifier,
}

__all__ = [
    "BaseColorClassifier",
    "MLPClassifier",
    "ShallowCNNClassifier",
    "MobileNetV3Classifier",
    "MODEL_REGISTRY",
]
