import torch
from torch import nn

from models.clip.clip_base import ClipBase


class CLIPMLPHead(ClipBase):
    """Clip with a multimodal adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化CLIPMLPHead类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类ClipBase的构造函数，传递骨干网络和根目录参数。
            2. 获取CLIP模型视觉编码器的输出维度，并计算适配器输入维度（为输出维度的两倍）。
            3. 定义隐藏层大小为16。
            4. 创建一个组合层，包含两个线性层和一个ReLU激活函数。
        """
        # pass default arguments to the parent class
        super(CLIPMLPHead, self).__init__(backbone, root=root)

        # add additional blocks to the model
        representation_dim = self._clip.visual.output_dim
        adapter_input_dim = representation_dim * 2
        hidden_size = 16

        self.combination_layer = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 包含可学习参数名称的集合，当前仅包含 "combination_layer"。
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["combination_layer"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        将模型移动到CPU设备上。

        功能:
            1. 将CLIP模型移动到CPU设备上。
            2. 将组合层移动到CPU设备上。
            3. 将CLIP模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self.combination_layer.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            1. 将组合层移动到CUDA设备上。
            2. 将CLIP模型移动到CUDA设备上。
        """
        self.combination_layer.to(torch.device("cuda"))
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
            3. 将图像特征和文本特征的数据类型转换为float32。
            4. 对图像特征和文本特征进行扩展，使其维度匹配。
            5. 将扩展后的图像特征和文本特征在最后一个维度上拼接。
            6. 将拼接后的特征输入到组合层进行处理，并挤压最后一个维度。
            7. 返回处理后的logits张量。
        """
        # Change the forward method to include the visual_mlp
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)
        image_features = image_features.to(torch.float32)  # [batch_size, rep_dim]
        text_features = text_features.to(torch.float32)  # [n_classes, rep_dim]

        image_features_exp = image_features.unsqueeze(1).repeat(
            1, text_features.shape[0], 1
        )  # [batch_size, n_classes, rep_dim]
        text_features_exp = text_features.unsqueeze(0).repeat(
            image_features.shape[0], 1, 1
        )  # [batch_size, n_classes, rep_dim]

        combined_features = torch.cat(
            (image_features_exp, text_features_exp), dim=2
        )  # [batch_size, n_classes, rep_dim]

        logits_per_image: torch.Tensor = self.combination_layer(combined_features).squeeze(
            -1
        )  # [batch_size, n_classes]

        return logits_per_image