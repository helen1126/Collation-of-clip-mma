from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Metric(ABC):
    """
    ���ж���ָ����ĳ�����ࡣ

    ���ඨ����һ�����󷽷� `calculate`�����ڼ����ض��Ķ���ָ�꣬
    ������д�� `__str__` �������Ա��ڴ�ӡ����ָ�����ʱ������������
    """

    @staticmethod
    @abstractmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        """
        �������ָ�ꡣ

        �˷�����һ�����󷽷�����Ҫ��������ʵ�֡���������ʵ��ǩ��Ԥ���ǩ��Ϊ���룬
        ������һ����������ʾ����õ��Ķ���ָ��ֵ��

        ����:
            y_true (NDArray[Any]): ��ʵ��ǩ�����顣
            y_pred (NDArray[Any]): Ԥ���ǩ�����顣

        ����:
            float: ����õ��Ķ���ָ��ֵ��
        """
        pass

    def __str__(self) -> str:
        """
        ���ض���ָ��������ơ�

        �˷�����д�� `__str__` ����������ӡ����ָ�����ʱ����������������

        ����:
            str: ����ָ��������ơ�
        """
        return self.__class__.__name__