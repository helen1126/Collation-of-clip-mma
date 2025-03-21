import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import requests
import os
from PIL import Image
from torchvision.datasets import SUN397 as _SUN397


class SUN397(_SUN397):
    """
    �Զ����SUN397���ݼ��࣬�̳���torchvision��SUN397�ࡣ
    �������ڴ���SUN397���ݼ���֧���Զ���ָ�����ݼ��ء�
    """

    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        """
        ��ָ����URL�������ݼ��ķָ���Ϣ��

        �ú�������洢�ָ���Ϣ��URL�������󣬻�ȡ��Ӧ���ݲ��������ΪJSON��ʽ��

        ����:
            dict[str, list[int]]: �������ݼ��ָ���Ϣ���ֵ䣬��Ϊ�ָ����ͣ��� 'train', 'test' �ȣ���ֵΪ��Ӧ�������б�
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
        ��ʼ��SUN397���ݼ�ʵ����

        �ú�������ø���Ĺ��캯�����л����ĳ�ʼ��������Ȼ����ݴ���Ĳ����������ݼ��ķָʽ����Ŀ¼�ȡ�
        �������طָ���Ϣ�������Ҫ���������ݼ�����������ݼ��Ƿ���ڡ���󣬽����ݼ��е�ͼ���ļ��ͱ�ǩ��Ϣ�洢��ʵ�������С�

        ����:
            root (str): ���ݼ��洢�ĸ�Ŀ¼��
            transform (Optional[Callable]): Ӧ����ͼ�����ݵ�ת��������Ĭ��Ϊ None��
            target_transform (Optional[Callable]): Ӧ����Ŀ�����ݣ���ǩ����ת��������Ĭ��Ϊ None��
            split (str): ���ݼ��ķָʽ����ѡֵΪ 'train' �ȣ�Ĭ��Ϊ 'train'��
            download (bool): �Ƿ��������ݼ���Ĭ��Ϊ False��
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
        ��ȡ���ݼ��ĳ��ȣ�������������������

        ����:
            int: ���ݼ��ĳ��ȣ��ɴ洢ͼ���ļ�·�����б�ĳ��Ⱦ�����
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        ����������ȡ���ݼ��е�һ��������

        �ú�������ݸ����������Ӵ洢��ͼ���ļ�·���б�ͱ�ǩ�б��л�ȡ��Ӧ��ͼ���ļ�·���ͱ�ǩ��
        Ȼ���ͼ�񲢽���ת��ΪRGBģʽ���������ת�����������ͼ��ͱ�ǩ�ֱ�Ӧ����Ӧ��ת��������
        ��󷵻ش�����ͼ��ͱ�ǩ��

        ����:
            idx (int): ������������

        ����:
            Tuple[Any, Any]: ����ͼ��Ͷ�Ӧ��ǩ��Ԫ�飬ͼ��ΪPIL.Image���󣬱�ǩΪ�����ı�ǩ��
        """
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(os.path.join(self.root, "SUN397", image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label