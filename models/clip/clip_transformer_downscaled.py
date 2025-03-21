import torch
from torch import Tensor, nn

from models.clip.clip_base import ClipBase
from models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)


class ClipTransformerDownscaled(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化 ClipTransformerDownscaled 类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类 ClipBase 的构造函数，传递骨干网络和根目录参数。
            2. 计算降维后的维度 down_dim。
            3. 初始化一个 MaskedMultiheadAttention 模块，存储在 self.mmha 中。
            4. 初始化图像和文本的下采样线性层，分别存储在 self.image_downsample 和 self.text_downsample 中。
            5. 初始化图像和文本的上采样线性层，分别存储在 self.image_upsample 和 self.text_upsample 中。
        """
        # pass default arguments to the parent class
        super(ClipTransformerDownscaled, self).__init__(backbone, root=root)

        down_dim = 512 // 32

        self.mmha = MaskedMultiheadAttention(embed_dim=down_dim)

        # Downsampling from 512 to down_dim
        self.image_downsample = nn.Linear(512, down_dim, bias=False)
        self.text_downsample = nn.Linear(512, down_dim, bias=False)

        # Upsampling from down_dim to 512
        self.image_upsample = nn.Linear(down_dim, 512, bias=False)
        self.text_upsample = nn.Linear(down_dim, 512, bias=False)

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 包含可学习参数名称的集合，包括 "mmha"、"image_downsample"、
                      "text_downsample"、"image_upsample" 和 "text_upsample"。
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(
            [
                "mmha",
                "image_downsample",
                "text_downsample",
                "image_upsample",
                "text_upsample",
            ]
        )

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        将模型移动到 CPU 设备上。

        功能:
            1. 将父类中的 _clip 模型移动到 CPU 设备上，并将其数据类型转换为 float。
            2. 将 self.mmha 模块移动到 CPU 设备上。
            3. 将图像和文本的下采样、上采样线性层移动到 CPU 设备上。
        """
        self._clip.to(torch.device("cpu"))
        self._clip.float()
        self.mmha.to(torch.device("cpu"))
        self.image_downsample.to(torch.device("cpu"))
        self.text_downsample.to(torch.device("cpu"))
        self.image_upsample.to(torch.device("cpu"))
        self.text_upsample.to(torch.device("cpu"))

    def to_mps(self) -> None:
        """
        将模型移动到 MPS（苹果的 Metal Performance Shaders）设备上。

        功能:
            1. 将父类中的 _clip 模型移动到 MPS 设备上。
            2. 将 self.mmha 模块移动到 MPS 设备上。
            3. 将图像和文本的下采样、上采样线性层移动到 MPS 设备上。
        """
        self._clip.to(torch.device("mps"))
        self.mmha.to(torch.device("mps"))
        self.image_downsample.to(torch.device("mps"))
        self.text_downsample.to(torch.device("mps"))
        self.image_upsample.to(torch.device("mps"))
        self.text_upsample.to(torch.device("mps"))

    def to_cuda(self) -> None:
        """
        将模型移动到 CUDA 设备上。

        功能:
            1. 将父类中的 _clip 模型移动到 CUDA 设备上。
            2. 将 self.mmha 模块移动到 CUDA 设备上。
            3. 将图像和文本的下采样、上采样线性层移动到 CUDA 设备上。
        """
        self._clip.to(torch.device("cuda"))
        self.mmha.to(torch.device("cuda"))
        self.image_downsample.to(torch.device("cuda"))
        self.text_downsample.to(torch.device("cuda"))
        self.image_upsample.to(torch.device("cuda"))
        self.text_upsample.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            images (torch.Tensor): 输入的图像张量。
            prompts (list[str] | None, 可选): 输入的文本提示列表，默认为 None。

        返回:
            torch.Tensor: 图像相对于文本的 logits 张量。

        功能:
            1. 如果提供了 prompts，则调用 encode_text 方法编码文本提示得到文本特征；
               如果存在预计算的文本提示特征，则使用预计算的特征；
               否则，抛出 ValueError 异常。
            2. 获取图像的批量大小和类别数量。
            3. 调用 encode_images 方法编码输入的图像得到图像特征，并调整其维度和数据类型。
            4. 调整文本特征的维度和数据类型。
            5. 对图像特征和文本特征进行下采样。
            6. 将下采样后的文本特征和图像特征在第 0 维拼接，得到输入序列。
            7. 将输入序列传入 self.mmha 模块进行处理。
            8. 将输入序列和模块输出相加。
            9. 从相加后的序列中分离出图像特征和文本特征，并调整其维度。
            10. 对分离后的图像特征和文本特征进行上采样。
            11. 调整图像特征的维度。
            12. 将上采样后的特征与原始特征相加。
            13. 计算图像相对于文本的 logits。
        """
        if prompts:
            text_features = self.encode_text(prompts)

        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features

        else:
            raise ValueError("At least one prompts or pre-computed prompt features has to be present.")

        batch_size = images.shape[0]
        num_classes = text_features.shape[0]

        image_features = self.encode_images(images)

        image_features = image_features.to(torch.float32).unsqueeze(0)

        text_features = text_features.to(torch.float32).unsqueeze(1).expand(-1, batch_size, -1)

        _image_features = self.image_downsample(image_features)
        _text_features = self.text_downsample(text_features)

        input_seq = torch.cat([_text_features, _image_features], dim=0)

        tr_outputs = self.mmha.forward(input_seq)

        input_seq = input_seq + tr_outputs

        _image_features = input_seq[num_classes:]  # [1, batch_size, embed_dim]
        _text_features = input_seq[:num_classes]  # [n_classes, batch_size, embed_dim]

        _image_features = _image_features.permute(1, 0, 2)  # [batch_size, 1, embed_dim]
        _text_features = _text_features.permute(1, 0, 2)  # [batch_size, n_classes, embed_dim]

        _image_features = self.image_upsample(_image_features)
        _text_features = self.text_upsample(_text_features)

        _image_features = _image_features.transpose(1, 2)

        text_features = text_features.permute(1, 0, 2) + _text_features
        image_features = image_features.permute(1, 2, 0) + _image_features

        logits_per_image: torch.Tensor = torch.bmm(text_features, image_features).squeeze(2)

        return logits_per_image