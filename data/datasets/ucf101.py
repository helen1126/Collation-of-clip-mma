import json
import os
import ssl
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import UCF101 as _UCF101
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class UCF101(VisionDataset):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y&export=download&authuser=0"
    _DATASET_URL = "https://drive.usercontent.google.com/download?id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O&export=download&authuser=0&confirm=t&uuid=7543a322-e294-4c7a-9b80-172e644f0b02&at=APZUnTWmj4xoWwpIk0N_qZS5heJF%3A1717005803820"

    def _download_split(self) -> dict[str, list[int]]:
        """
        从指定的URL下载数据集的分割信息。

        该函数会向存储分割信息的URL发送请求，获取响应内容并将其解析为JSON格式。

        返回:
            dict[str, list[int]]: 包含数据集分割信息的字典，键为分割类型（如 'train', 'test'），值为对应的索引列表。
        """
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        初始化UCF101数据集实例。

        参数:
            root (str): 数据集存储的根目录。
            train (bool): 是否使用训练集，默认为True。如果为True，使用训练集；否则使用测试集。
            transform (Optional[Callable]): 应用于图像数据的转换函数，默认为None。
            download (bool): 是否下载数据集，默认为False。

        功能:
            1. 根据train参数确定使用训练集还是测试集。
            2. 存储转换函数和分割信息。
            3. 构建数据集的基础目录路径。
            4. 如果download为True，则下载数据集。
            5. 检查数据集是否存在，如果不存在则抛出异常。
            6. 初始化数据和标签列表，并根据分割信息填充这些列表。
            7. 从字典形式的类别信息构建按标签排序的类别列表。
        """
        split = "train" if train else "test"

        self.transform = transform
        self.split = split
        self._split_dict = self._download_split()

        self.root = os.path.expanduser(root)
        self._data_dir = Path(self.root, "UCF-101-midframes")

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
            int: 数据集的长度，由存储图像文件路径的列表的长度决定。
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        根据索引获取数据集中的一个样本。

        参数:
            idx (int): 样本的索引。

        返回:
            Tuple[Any, Any]: 包含图像和对应标签的元组。图像为PIL.Image对象，标签为对应的类别标签。

        功能:
            1. 根据索引从数据和标签列表中获取对应的图像文件路径和标签。
            2. 打开图像文件并将其转换为RGB模式。
            3. 如果存在转换函数，则对图像应用该转换。
            4. 返回处理后的图像和标签。
        """
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(os.path.join(self._data_dir, image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _check_exists(self) -> bool:
        """
        检查数据集是否已经存在。

        返回:
            bool: 如果数据集目录存在，则返回True；否则返回False。
        """
        return self._data_dir.is_dir()

    def download(self) -> None:
        """
        下载并解压UCF101数据集。

        如果数据集已经存在，则不进行下载操作。
        否则，从指定的URL下载数据集并解压到指定的根目录。
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, filename="ucf101.zip")