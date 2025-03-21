import json
from pathlib import Path
from typing import Callable

import requests
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import verify_str_arg


class OxfordFlowers(Flowers102):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        """
        从指定的URL下载数据集的分割信息。

        该函数会向存储分割信息的URL发送请求，获取响应内容并将其解析为JSON格式。

        返回:
            dict[str, list[int]]: 包含数据集分割信息的字典，键为分割类型（如 'train', 'val', 'test'），值为对应的索引列表。
        """
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        """
        初始化OxfordFlowers数据集实例。

        该函数会调用父类的构造函数进行基本的初始化操作，然后验证分割类型，下载分割信息，
        根据下载标志决定是否下载数据集，检查数据集的完整性，并将数据集中的图像文件和标签信息存储在实例属性中。

        参数:
            root (str): 数据集存储的根目录。
            split (str): 数据集的分割方式，可选值为 "train", "val", "test"，默认为 "train"。
            transform (Callable | None): 应用于图像数据的转换函数，默认为 None。
            target_transform (Callable | None): 应用于目标数据（标签）的转换函数，默认为 None。
            download (bool): 是否下载数据集，默认为 False。
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        # 验证分割类型是否为合法值
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        # 定义数据集的基础文件夹路径
        self._base_folder = Path(self.root) / "flowers-102"
        # 定义图像文件所在的文件夹路径
        self._images_folder = self._base_folder / "jpg"
        # 下载数据集的分割信息
        self._split_dict = self._download_split()

        # 如果download标志为True，则下载数据集
        if download:
            self.download()

        # 检查数据集的完整性，如果不完整则抛出异常
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # 初始化存储标签和图像文件路径的列表
        self._labels = []
        self._image_files = []

        # 遍历分割信息，将图像标签和文件路径添加到相应的列表中
        for image_id, image_label, _ in self._split_dict[self._split]:
            self._labels.append(image_label)
            self._image_files.append(self._images_folder / image_id)