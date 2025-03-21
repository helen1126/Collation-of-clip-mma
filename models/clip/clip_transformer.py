# import ClipBase
from typing import Self

import torch

from models.clip.clip_base import ClipBase
from models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)


class ClipTransformer(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化ClipTransformer类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类ClipBase的构造函数，传递骨干网络和根目录参数。
            2. 初始化一个MaskedMultiheadAttention模块，存储在self.mmha中。
        """
        # pass default arguments to the parent class
        super(ClipTransformer, self).__init__(backbone, root=root)

        self.mmha = MaskedMultiheadAttention()

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 包含可学习参数名称的集合，当前仅包含 "mmha"。
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["mmha"])

    def eval(self) -> Self:
        """
        将模型设置为评估模式。

        返回:
            Self: 返回当前模型实例。

        功能:
            1. 将父类中的_clip模型设置为评估模式。
            2. 将self.mmha模块设置为评估模式。
        """
        self._clip.eval()
        self.mmha.eval()
        return self

    def train_(self) -> Self:
        """
        将模型设置为训练模式。

        返回:
            Self: 返回当前模型实例。

        功能:
            1. 将父类中的_clip模型设置为训练模式。
            2. 将self.mmha模块设置为训练模式。
        """
        self._clip.train()
        self.mmha.train()
        return self

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        将模型移动到CPU设备上。

        功能:
            1. 将父类中的_clip模型移动到CPU设备上。
            2. 将self.mmha模块移动到CPU设备上。
            3. 将_clip模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self.mmha.to(torch.device("cpu"))
        self._clip.float()

    def to_mps(self) -> None:
        """
        将模型移动到MPS（苹果的Metal Performance Shaders）设备上。

        功能:
            1. 将父类中的_clip模型移动到MPS设备上。
            2. 将self.mmha模块移动到MPS设备上。
        """
        self._clip.to(torch.device("mps"))
        self.mmha.to(torch.device("mps"))

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            1. 将self.mmha模块移动到CUDA设备上。
            2. 将父类中的_clip模型移动到CUDA设备上。
        """
        self.mmha.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            images (torch.Tensor): 输入的图像张量。
            prompts (list[str] | None, 可选): 输入的文本提示列表，默认为None。

        返回:
            torch.Tensor: 图像相对于文本的logits张量。

        功能:
            1. 如果提供了prompts，则调用encode_text方法编码文本提示得到文本特征；
               如果存在预计算的文本提示特征，则使用预计算的特征；
               否则，抛出ValueError异常。
            2. 调用encode_images方法编码输入的图像得到图像特征，并调整其维度和数据类型。
            3. 调整文本特征的维度和数据类型。
            4. 计算类别数量。
            5. 将文本特征和图像特征在第0维拼接，得到输入序列。
            6. 将输入序列传入self.mmha模块进行处理。
            7. 处理图像特征和文本特征，计算图像相对于文本的logits。
        """
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed prompt features has to be present.")

        image_features = self.encode_images(images).to(torch.float32).unsqueeze(0)
        text_features = text_features.to(torch.float32).unsqueeze(1).expand(-1, image_features.shape[1], -1)

        num_classes = text_features.shape[0]

        input_seq = torch.cat([text_features, image_features], dim=0)
        tr_outputs = self.mmha.forward(input_seq)

        _image_features = (
            (image_features + tr_outputs[num_classes:])
            .permute(1, 0, 2)
            .transpose(1, 2)
            .squeeze()
            .unsqueeze(1)
        )

        _text_features = text_features.permute(1, 0, 2)

        logits_per_image: torch.Tensor = torch.bmm(_image_features, _text_features.transpose(1, 2)).squeeze(1)

        return logits_per_image