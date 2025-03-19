from typing import Any

from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ._base_metric import Metric


class Accuracy(Metric):
    @staticmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        score: float = accuracy_score(y_true, y_pred)
        return score


class Precision(Metric):
    @staticmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        score: float = precision_score(y_true, y_pred)
        return score


class Recall(Metric):
    @staticmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        score: float = recall_score(y_true, y_pred)
        return score


METRICS = [Accuracy()]

from enum import Enum
from functools import partial

from ._base_model import BaseModel
from .clip_classifier import ClipClassifier


class Model(Enum):
    CLIP_VIT_BASE_PATCH16_PRETRAINED = partial(lambda: ClipClassifier("ViT-B/16"))

    @classmethod
    def from_str(cls, name: str) -> BaseModel:
        model: BaseModel = cls[name.upper()].value()
        return model
