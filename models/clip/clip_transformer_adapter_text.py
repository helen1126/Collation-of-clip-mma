import torch
from torch import nn

from models.clip.clip_base import ClipBase
from models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)
from models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPTransformerAdapterText(ClipBase):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化CLIPTransformerAdapterText类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类ClipBase的构造函数，传递骨干网络和根目录参数。
            2. 从_clip模型中获取图像编码器并存储在self.image_encoder中。
            3. 从_clip模型中获取logit_scale并存储在self.logit_scale中。
            4. 初始化一个MaskedMultiheadAttentionDownsampled模块作为适配器，存储在self.adapter中。
        """
        # pass default arguments to the parent class
        super(CLIPTransformerAdapterText, self).__init__(backbone, root=root)

        self.image_encoder = self._clip.visual
        self.logit_scale = self._clip.logit_scale
        self.adapter = MaskedMultiheadAttentionDownsampled()

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 包含可学习参数名称的集合，当前仅包含 "adapter"。
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["adapter"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        将模型移动到CPU设备上。

        功能:
            1. 将父类中的_clip模型移动到CPU设备上。
            2. 将self.adapter模块移动到CPU设备上。
            3. 将_clip模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self.adapter.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            1. 将self.adapter模块移动到CUDA设备上。
            2. 将父类中的_clip模型移动到CUDA设备上。
        """
        self.adapter.to(torch.device("cuda"))
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
            2. 调用encode_images方法编码输入的图像得到图像特征，并将其数据类型转换为float32。
            3. 调整文本特征和图像特征的维度，使其能够进行拼接。
            4. 将文本特征和图像特征在第0维拼接，得到适配器的输入。
            5. 将适配器输入传入self.adapter模块进行处理，得到适配器输出。
            6. 从适配器输出中分离出图像特征和文本特征。
            7. 对适配器输出的图像特征和文本特征进行归一化处理。
            8. 定义图像特征和文本特征的融合比例。
            9. 对原始的图像特征和文本特征进行归一化处理。
            10. 根据融合比例，将原始特征和适配器输出的特征进行融合。
            11. 再次对融合后的图像特征和文本特征进行归一化处理。
            12. 计算logit_scale的指数值。
            13. 计算图像特征和文本特征的点积，并乘以logit_scale，得到logits。
        """
        # Change the forward method to include the visual_mlp
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images).to(torch.float32)  # [batch_size, rep_dim]
        text_features = text_features.to(torch.float32)  # [n_classes, rep_dim]

        num_classes = text_features.shape[0]

        text_features = text_features.unsqueeze(1).expand(-1, image_features.shape[0], -1)
        image_features = image_features.unsqueeze(0)

        adapter_input = torch.cat(
            [text_features, image_features],
            dim=0,
        )

        adapter_output = self.adapter(adapter_input)

        adapter_image_features = adapter_output[num_classes:]
        adapter_text_features = adapter_output[:num_classes]

        adapter_image_features = adapter_image_features / adapter_image_features.norm(dim=-1, keepdim=True)
        adapter_text_features = adapter_text_features / adapter_text_features.norm(dim=-1, keepdim=True)

        image_ratio = 0.2
        text_ratio = 0.2

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_ratio * adapter_image_features + (1 - image_ratio) * image_features
        text_features = text_ratio * adapter_text_features + (1 - text_ratio) * text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image: torch.Tensor = logit_scale * torch.bmm(
            image_features.permute(1, 0, 2), text_features.permute(1, 2, 0)
        ).squeeze(1)

        return logits_per_image