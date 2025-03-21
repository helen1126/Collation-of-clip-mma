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
        ��ָ����URL�������ݼ��ķָ���Ϣ��

        �ú�������洢�ָ���Ϣ��URL�������󣬻�ȡ��Ӧ���ݲ��������ΪJSON��ʽ��

        ����:
            dict[str, list[int]]: �������ݼ��ָ���Ϣ���ֵ䣬��Ϊ�ָ����ͣ��� 'train', 'val', 'test'����ֵΪ��Ӧ�������б�
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
        ��ʼ��OxfordFlowers���ݼ�ʵ����

        �ú�������ø���Ĺ��캯�����л����ĳ�ʼ��������Ȼ����֤�ָ����ͣ����طָ���Ϣ��
        �������ر�־�����Ƿ��������ݼ���������ݼ��������ԣ��������ݼ��е�ͼ���ļ��ͱ�ǩ��Ϣ�洢��ʵ�������С�

        ����:
            root (str): ���ݼ��洢�ĸ�Ŀ¼��
            split (str): ���ݼ��ķָʽ����ѡֵΪ "train", "val", "test"��Ĭ��Ϊ "train"��
            transform (Callable | None): Ӧ����ͼ�����ݵ�ת��������Ĭ��Ϊ None��
            target_transform (Callable | None): Ӧ����Ŀ�����ݣ���ǩ����ת��������Ĭ��Ϊ None��
            download (bool): �Ƿ��������ݼ���Ĭ��Ϊ False��
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        # ��֤�ָ������Ƿ�Ϊ�Ϸ�ֵ
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        # �������ݼ��Ļ����ļ���·��
        self._base_folder = Path(self.root) / "flowers-102"
        # ����ͼ���ļ����ڵ��ļ���·��
        self._images_folder = self._base_folder / "jpg"
        # �������ݼ��ķָ���Ϣ
        self._split_dict = self._download_split()

        # ���download��־ΪTrue�����������ݼ�
        if download:
            self.download()

        # ������ݼ��������ԣ�������������׳��쳣
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # ��ʼ���洢��ǩ��ͼ���ļ�·�����б�
        self._labels = []
        self._image_files = []

        # �����ָ���Ϣ����ͼ���ǩ���ļ�·����ӵ���Ӧ���б���
        for image_id, image_label, _ in self._split_dict[self._split]:
            self._labels.append(image_label)
            self._image_files.append(self._images_folder / image_id)