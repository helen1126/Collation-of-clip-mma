import torch


class AddGaussianNoise(object):
    """
    ��������Ӹ�˹�����ı任�ࡣ

    ����ʵ����һ���ɵ��ö��������������������Ӹ�˹������
    ��˹�����ľ�ֵ�ͱ�׼������ڳ�ʼ��ʱָ����
    """

    def __init__(self, mean=0.0, std=1.0):
        """
        ��ʼ�� AddGaussianNoise ���ʵ����

        ����:
            mean (float, ��ѡ): ��˹�����ľ�ֵ��Ĭ��Ϊ 0.0��
            std (float, ��ѡ): ��˹�����ı�׼�Ĭ��Ϊ 1.0��
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        �������������Ӹ�˹������

        �÷�������һ��������Ϊ���룬����������������ͬ��״�ĸ�˹������
        ��������ӵ����������ϡ�

        ����:
            tensor (torch.Tensor): �����������

        ����:
            torch.Tensor: ����˸�˹������������
        """
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        """
        ���ر�ʾ����ʵ�����ַ�����

        �÷�������һ���ַ��������������Լ���ʼ��ʱָ���ľ�ֵ�ͱ�׼�

        ����:
            str: ��ʾ����ʵ�����ַ�����
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"