from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class ZeroShotDataset(ABC):
    def __init__(self, dataset: Dataset) -> None:
        """
        初始化ZeroShotDataset类的实例。

        参数:
            dataset (Dataset): 作为基础数据集的torch.utils.data.Dataset对象。
        """
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        """
        获取ZeroShotDataset实例所包装的基础数据集。

        返回:
            Dataset: 基础数据集对象。
        """
        return self._dataset

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """
        抽象方法，用于获取数据集中所有样本对应的标签列表。
        该方法需要在子类中被具体实现。

        返回:
            list[str]: 包含数据集中所有样本对应标签的字符串列表。
        """
        pass