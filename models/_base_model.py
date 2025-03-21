from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class BaseModel(ABC):
    @abstractmethod
    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        为评估目的预测模型的输出。

        该方法接收一个数据加载器作为输入，用于提供待预测的数据。它将返回一个元组，
        元组的第一个元素是真实标签（y_true）的数组，第二个元素是模型预测标签（y_pred）的数组。
        此方法是一个抽象方法，需要在具体的子类中实现。

        参数:
            x (DataLoader[Any]): 用于提供待预测数据的数据加载器。

        返回:
            tuple[NDArray[Any], NDArray[Any]]: 一个元组，包含真实标签数组和预测标签数组。
        """
        pass

    @property
    @abstractmethod
    def transforms(self) -> Compose:
        """
        获取用于推理和训练的图像变换组合。

        该属性返回一个 `Compose` 对象，其中包含了一系列用于图像预处理的变换操作。
        这些变换操作将在数据输入模型之前对图像进行处理，例如缩放、裁剪、归一化等。
        此属性是一个抽象属性，需要在具体的子类中实现。

        返回:
            Compose: 包含图像变换操作的 `Compose` 对象。
        """
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        获取用于推理和训练的批量大小。

        该属性返回一个整数，表示在推理和训练过程中每次处理的样本数量。
        批量大小是一个重要的超参数，它会影响模型的训练速度和内存使用情况。
        此属性是一个抽象属性，需要在具体的子类中实现。

        返回:
            int: 用于推理和训练的批量大小。
        """
        pass

    @abstractmethod
    def reconfig_labels(
        self,
        labels: list[str],
    ) -> None:
        """
        为给定的数据集重新配置模型。

        该方法接收一个字符串列表作为输入，列表中的每个元素表示一个类别标签。
        它将根据这些标签对模型进行重新配置，例如更新模型的输出层以适应新的类别数量。
        此方法是一个抽象方法，需要在具体的子类中实现。

        参数:
            labels (list[str]): 包含类别标签的字符串列表。
        """
        pass