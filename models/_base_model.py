from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class BaseModel(ABC):
    @abstractmethod
    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Ϊ����Ŀ��Ԥ��ģ�͵������

        �÷�������һ�����ݼ�������Ϊ���룬�����ṩ��Ԥ������ݡ���������һ��Ԫ�飬
        Ԫ��ĵ�һ��Ԫ������ʵ��ǩ��y_true�������飬�ڶ���Ԫ����ģ��Ԥ���ǩ��y_pred�������顣
        �˷�����һ�����󷽷�����Ҫ�ھ����������ʵ�֡�

        ����:
            x (DataLoader[Any]): �����ṩ��Ԥ�����ݵ����ݼ�������

        ����:
            tuple[NDArray[Any], NDArray[Any]]: һ��Ԫ�飬������ʵ��ǩ�����Ԥ���ǩ���顣
        """
        pass

    @property
    @abstractmethod
    def transforms(self) -> Compose:
        """
        ��ȡ���������ѵ����ͼ��任��ϡ�

        �����Է���һ�� `Compose` �������а�����һϵ������ͼ��Ԥ����ı任������
        ��Щ�任����������������ģ��֮ǰ��ͼ����д����������š��ü�����һ���ȡ�
        ��������һ���������ԣ���Ҫ�ھ����������ʵ�֡�

        ����:
            Compose: ����ͼ��任������ `Compose` ����
        """
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        ��ȡ���������ѵ����������С��

        �����Է���һ����������ʾ�������ѵ��������ÿ�δ��������������
        ������С��һ����Ҫ�ĳ�����������Ӱ��ģ�͵�ѵ���ٶȺ��ڴ�ʹ�������
        ��������һ���������ԣ���Ҫ�ھ����������ʵ�֡�

        ����:
            int: ���������ѵ����������С��
        """
        pass

    @abstractmethod
    def reconfig_labels(
        self,
        labels: list[str],
    ) -> None:
        """
        Ϊ���������ݼ���������ģ�͡�

        �÷�������һ���ַ����б���Ϊ���룬�б��е�ÿ��Ԫ�ر�ʾһ������ǩ��
        ����������Щ��ǩ��ģ�ͽ����������ã��������ģ�͵����������Ӧ�µ����������
        �˷�����һ�����󷽷�����Ҫ�ھ����������ʵ�֡�

        ����:
            labels (list[str]): ��������ǩ���ַ����б�
        """
        pass