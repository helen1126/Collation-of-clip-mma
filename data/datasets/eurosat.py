import json
import os
import ssl
from typing import Any, Callable, Optional, Tuple

import requests
from PIL import Image
from torchvision.datasets import EuroSAT as _EuroSAT

ssl._create_default_https_context = ssl._create_unverified_context


class EuroSAT(_EuroSAT):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        """
        从指定的URL下载数据集的分割信息。

        返回:
            dict[str, list[int]]: 包含数据集分割信息的字典，键为分割类型（如'train'、'test'），值为对应的索引列表。
        """
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        初始化EuroSAT数据集实例。

        参数:
            root (str): 数据集存储的根目录。
            split (str): 数据集的分割方式，可选值为 "train" 或其他自定义分割，默认为 "train"。
            transform (Optional[Callable]): 应用于图像数据的转换函数，默认为 None。
            target_transform (Optional[Callable]): 应用于目标数据的转换函数，默认为 None。
            download (bool): 是否下载数据集，默认为 False。
        """
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)

        self.split = split
        self._split_dict = self._download_split()

        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.root = os.path.expanduser(root)

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

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        根据索引获取数据集中的一个样本。

        参数:
            idx (int): 样本的索引。

        返回:
            Tuple[Any, Any]: 包含图像和目标的元组，图像为PIL.Image对象，目标为对应的标签。
        """
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(os.path.join(self.root, "eurosat", "2750", image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label