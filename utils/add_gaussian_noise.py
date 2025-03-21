import torch


class AddGaussianNoise(object):
    """
    向张量添加高斯噪声的变换类。

    该类实现了一个可调用对象，用于向输入的张量添加高斯噪声。
    高斯噪声的均值和标准差可以在初始化时指定。
    """

    def __init__(self, mean=0.0, std=1.0):
        """
        初始化 AddGaussianNoise 类的实例。

        参数:
            mean (float, 可选): 高斯噪声的均值，默认为 0.0。
            std (float, 可选): 高斯噪声的标准差，默认为 1.0。
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        向输入的张量添加高斯噪声。

        该方法接收一个张量作为输入，生成与输入张量相同形状的高斯噪声，
        并将其添加到输入张量上。

        参数:
            tensor (torch.Tensor): 输入的张量。

        返回:
            torch.Tensor: 添加了高斯噪声的张量。
        """
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        """
        返回表示该类实例的字符串。

        该方法返回一个字符串，包含类名以及初始化时指定的均值和标准差。

        返回:
            str: 表示该类实例的字符串。
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"