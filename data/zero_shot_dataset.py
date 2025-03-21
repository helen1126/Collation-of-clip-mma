from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class ZeroShotDataset(ABC):
    def __init__(self, dataset: Dataset) -> None:
        """
        ��ʼ��ZeroShotDataset���ʵ����

        ����:
            dataset (Dataset): ��Ϊ�������ݼ���torch.utils.data.Dataset����
        """
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        """
        ��ȡZeroShotDatasetʵ������װ�Ļ������ݼ���

        ����:
            Dataset: �������ݼ�����
        """
        return self._dataset

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """
        ���󷽷������ڻ�ȡ���ݼ�������������Ӧ�ı�ǩ�б�
        �÷�����Ҫ�������б�����ʵ�֡�

        ����:
            list[str]: �������ݼ�������������Ӧ��ǩ���ַ����б�
        """
        pass