from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Metric(ABC):
    """
    所有度量指标类的抽象基类。

    该类定义了一个抽象方法 `calculate`，用于计算特定的度量指标，
    并且重写了 `__str__` 方法，以便在打印度量指标对象时返回其类名。
    """

    @staticmethod
    @abstractmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        """
        计算度量指标。

        此方法是一个抽象方法，需要在子类中实现。它接受真实标签和预测标签作为输入，
        并返回一个浮点数表示计算得到的度量指标值。

        参数:
            y_true (NDArray[Any]): 真实标签的数组。
            y_pred (NDArray[Any]): 预测标签的数组。

        返回:
            float: 计算得到的度量指标值。
        """
        pass

    def __str__(self) -> str:
        """
        返回度量指标类的名称。

        此方法重写了 `__str__` 方法，当打印度量指标对象时，将返回其类名。

        返回:
            str: 度量指标类的名称。
        """
        return self.__class__.__name__