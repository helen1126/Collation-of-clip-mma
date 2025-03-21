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
        ��ָ����URL�������ݼ��ķָ���Ϣ��

        �ú�������洢�ָ���Ϣ��URL�������󣬻�ȡ��Ӧ���ݲ��������ΪJSON��ʽ��

        ����:
            dict[str, list[int]]: �������ݼ��ָ���Ϣ���ֵ䣬��Ϊ�ָ����ͣ��� 'train', 'test'����ֵΪ��Ӧ�������б�
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
        ��ʼ��UCF101���ݼ�ʵ����

        ����:
            root (str): ���ݼ��洢�ĸ�Ŀ¼��
            train (bool): �Ƿ�ʹ��ѵ������Ĭ��ΪTrue�����ΪTrue��ʹ��ѵ����������ʹ�ò��Լ���
            transform (Optional[Callable]): Ӧ����ͼ�����ݵ�ת��������Ĭ��ΪNone��
            download (bool): �Ƿ��������ݼ���Ĭ��ΪFalse��

        ����:
            1. ����train����ȷ��ʹ��ѵ�������ǲ��Լ���
            2. �洢ת�������ͷָ���Ϣ��
            3. �������ݼ��Ļ���Ŀ¼·����
            4. ���downloadΪTrue�����������ݼ���
            5. ������ݼ��Ƿ���ڣ�������������׳��쳣��
            6. ��ʼ�����ݺͱ�ǩ�б������ݷָ���Ϣ�����Щ�б�
            7. ���ֵ���ʽ�������Ϣ��������ǩ���������б�
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
        ��ȡ���ݼ��ĳ��ȣ�������������������

        ����:
            int: ���ݼ��ĳ��ȣ��ɴ洢ͼ���ļ�·�����б�ĳ��Ⱦ�����
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        ����������ȡ���ݼ��е�һ��������

        ����:
            idx (int): ������������

        ����:
            Tuple[Any, Any]: ����ͼ��Ͷ�Ӧ��ǩ��Ԫ�顣ͼ��ΪPIL.Image���󣬱�ǩΪ��Ӧ������ǩ��

        ����:
            1. �������������ݺͱ�ǩ�б��л�ȡ��Ӧ��ͼ���ļ�·���ͱ�ǩ��
            2. ��ͼ���ļ�������ת��ΪRGBģʽ��
            3. �������ת�����������ͼ��Ӧ�ø�ת����
            4. ���ش�����ͼ��ͱ�ǩ��
        """
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(os.path.join(self._data_dir, image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _check_exists(self) -> bool:
        """
        ������ݼ��Ƿ��Ѿ����ڡ�

        ����:
            bool: ������ݼ�Ŀ¼���ڣ��򷵻�True�����򷵻�False��
        """
        return self._data_dir.is_dir()

    def download(self) -> None:
        """
        ���ز���ѹUCF101���ݼ���

        ������ݼ��Ѿ����ڣ��򲻽������ز�����
        ���򣬴�ָ����URL�������ݼ�����ѹ��ָ���ĸ�Ŀ¼��
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, filename="ucf101.zip")