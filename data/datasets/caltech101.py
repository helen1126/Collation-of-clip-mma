import json
import os
from typing import Any, Callable, Optional, Union

import requests
from PIL import Image
from torchvision.datasets import Caltech101 as _Caltech101
from torchvision.datasets.utils import verify_str_arg


class Caltech101(_Caltech101):
    """
    自定义的 Caltech 101 数据集类，继承自 torchvision 的 Caltech101 类。
    该类提供了对 Caltech 101 数据集的加载和处理功能，支持自定义分割和目标类型。
    """
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        """
        从指定的 URL 下载数据集的分割信息。

        返回:
            dict[str, list[int]]: 包含训练集和测试集分割信息的字典。
        """
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[list[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        初始化 Caltech 101 数据集实例。

        参数:
            root (str): 数据集存储的根目录。
            split (str): 数据集的分割方式，可选值为 "train" 或 "test"，默认为 "train"。
            target_type (Union[list[str], str]): 目标类型，可选值为 "category" 或 "annotation"，默认为 "category"。
            transform (Optional[Callable]): 应用于图像数据的转换函数，默认为 None。
            target_transform (Optional[Callable]): 应用于目标数据的转换函数，默认为 None。
            download (bool): 是否下载数据集，默认为 False。
        """
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.split = verify_str_arg(split, "split", ("train", "test"))
        self._split_dict: dict[str, list] = self._download_split()
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._categories = dict()

        self.data = []
        self.targets = []

        for filepath, label, category in self._split_dict[self.split]:
            self.data.append(filepath)
            self.targets.append(label)
            self._categories.setdefault(label, category)

        # extract dict _categories to list categories sorted by label
        self.categories = [self._categories[label] for label in sorted(self._categories.keys())]

    def __len__(self) -> int:
        """
        获取数据集的长度，即数据样本的数量。

        返回:
            int: 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        根据索引获取数据集中的一个样本。

        参数:
            index (int): 样本的索引。

        返回:
            tuple[Any, Any]: 包含图像和目标的元组，目标的类型由 target_type 指定。
        """
        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.data[index],
            )
        )

        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target