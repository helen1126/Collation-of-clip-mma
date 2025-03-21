import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import requests
import os
from PIL import Image
from torchvision.datasets import SUN397 as _SUN397


class SUN397(_SUN397):
    """
    自定义的SUN397数据集类，继承自torchvision的SUN397类。
    该类用于处理SUN397数据集，支持自定义分割和数据加载。
    """

    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        """
        从指定的URL下载数据集的分割信息。

        该函数会向存储分割信息的URL发送请求，获取响应内容并将其解析为JSON格式。

        返回:
            dict[str, list[int]]: 包含数据集分割信息的字典，键为分割类型（如 'train', 'test' 等），值为对应的索引列表。
        """
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: str = "train",
        download: bool = False,
    ) -> None:
        """
        初始化SUN397数据集实例。

        该函数会调用父类的构造函数进行基本的初始化操作，然后根据传入的参数设置数据集的分割方式、根目录等。
        接着下载分割信息，如果需要则下载数据集，并检查数据集是否存在。最后，将数据集中的图像文件和标签信息存储在实例属性中。

        参数:
            root (str): 数据集存储的根目录。
            transform (Optional[Callable]): 应用于图像数据的转换函数，默认为 None。
            target_transform (Optional[Callable]): 应用于目标数据（标签）的转换函数，默认为 None。
            split (str): 数据集的分割方式，可选值为 'train' 等，默认为 'train'。
            download (bool): 是否下载数据集，默认为 False。
        """
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)
        self.split = split
        self.root = root

        self._data_dir = Path(self.root) / "SUN397"
        self._split_dict = self._download_split()

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

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
            int: 数据集的长度，由存储图像文件路径的列表的长度决定。
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        根据索引获取数据集中的一个样本。

        该函数会根据给定的索引从存储的图像文件路径列表和标签列表中获取对应的图像文件路径和标签，
        然后打开图像并将其转换为RGB模式。如果存在转换函数，则对图像和标签分别应用相应的转换函数。
        最后返回处理后的图像和标签。

        参数:
            idx (int): 样本的索引。

        返回:
            Tuple[Any, Any]: 包含图像和对应标签的元组，图像为PIL.Image对象，标签为处理后的标签。
        """
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(os.path.join(self.root, "SUN397", image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label