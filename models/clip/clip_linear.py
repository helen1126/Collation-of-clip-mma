import torch
from torch import nn

from models.clip.clip_base import ClipBase



class ClipLinear(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化ClipLinear类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类ClipBase的构造函数，将骨干网络和根目录参数传递给父类。
            2. 初始化一个线性层self.image_linear，输入和输出维度均为CLIP模型视觉编码器的输出维度。
        """
        super(ClipLinear, self).__init__(backbone, root=root)

        self.image_linear = nn.Sequential(
            nn.Linear(self._clip.visual.output_dim, self._clip.visual.output_dim)
        )

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 包含可学习参数名称的集合，当前仅包含 "image_linear"。
        """
        return set(["image_linear"])

    def to_cpu(self) -> None:
        """
        将模型移动到CPU设备上。

        功能:
            1. 将CLIP模型移动到CPU设备上。
            2. 将self.image_linear线性层移动到CPU设备上。
            3. 将CLIP模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self.image_linear.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            1. 将self.image_linear线性层移动到CUDA设备上。
            2. 将CLIP模型移动到CUDA设备上。
        """
        self.image_linear.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            images (torch.Tensor): 输入的图像张量。
            prompts (list[str] | None, 可选): 输入的文本提示列表，默认为None。

        返回:
            torch.Tensor: 图像相对于文本的logits张量，形状为 [batch_size, n_classes]。

        功能:
            1. 如果提供了prompts，则调用encode_text方法编码文本提示得到文本特征；
               如果存在预计算的文本提示特征，则使用预计算的特征；
               否则，抛出ValueError异常。
            2. 调用encode_images方法编码输入的图像得到图像特征。
            3. 将图像特征输入到self.image_linear线性层进行处理。
            4. 计算图像特征与文本特征转置的矩阵乘积，并乘以logit_scale，得到logits。
        """
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)

        image_features = self.image_linear(image_features)

        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image