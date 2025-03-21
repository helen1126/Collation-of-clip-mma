import torch
from torch import nn

from models.clip.clip_base import ClipBase


class CLIPMMMLPAdapter(ClipBase):
    """Clip with a multimodal adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化CLIPMMMLPAdapter类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类ClipBase的构造函数，传递骨干网络和根目录参数。
            2. 获取CLIP模型视觉编码器的输出维度，并计算适配器输入维度（为输出维度的两倍）。
            3. 定义输出维度为表示维度，缩减因子为32，并计算隐藏层大小。
            4. 创建两个多层感知机（MLP）：mm_to_visual_mlp和mm_to_text_mlp，用于处理多模态特征。
        """
        # pass default arguments to the parent class
        super(CLIPMMMLPAdapter, self).__init__(backbone, root=root)

        representation_dim = self._clip.visual.output_dim
        adapter_input_dim = representation_dim * 2
        output_dim = representation_dim
        reduction = 32
        hidden_size = adapter_input_dim // reduction

        self.mm_to_visual_mlp = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim, bias=False),
            nn.ReLU(),
        )

        self.mm_to_text_mlp = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim, bias=False),
            nn.ReLU(),
        )

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 包含可学习参数名称的集合，这里是 "mm_to_visual_mlp" 和 "mm_to_text_mlp"。
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["mm_to_visual_mlp", "mm_to_text_mlp"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        将模型移动到CPU设备上。

        功能:
            1. 将CLIP模型移动到CPU设备上。
            2. 将mm_to_visual_mlp和mm_to_text_mlp移动到CPU设备上。
            3. 将CLIP模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self.mm_to_visual_mlp.to(torch.device("cpu"))
        self.mm_to_text_mlp.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            1. 将mm_to_visual_mlp和mm_to_text_mlp移动到CUDA设备上。
            2. 将CLIP模型移动到CUDA设备上。
        """
        self.mm_to_visual_mlp.to(torch.device("cuda"))
        self.mm_to_text_mlp.to(torch.device("cuda"))
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
            6. 定义一个融合比例ratio。
            7. 将拼接后的特征分别输入到mm_to_visual_mlp和mm_to_text_mlp中进行处理。
            8. 根据融合比例，将原始特征和适配器输出进行融合。
            9. 对融合后的图像特征和文本特征进行归一化处理。
            10. 计算图像特征和文本特征的点积，并乘以logit_scale，得到logits。
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
        )  # [batch_size, n_classes, rep_dim * 2]

        ratio = 0.2

        image_adapter_output = self.mm_to_visual_mlp(combined_features)  # [batch_size, n_classes, rep_dim]
        text_adapter_output = self.mm_to_text_mlp(combined_features)  # [batch_size, n_classes, rep_dim]

        image_features_exp = (
            1 - ratio
        ) * image_features_exp + ratio * image_adapter_output  # [batch_size, n_classes, rep_dim]
        text_features_exp = (
            1 - ratio
        ) * text_features_exp + ratio * text_adapter_output  # [batch_size, n_classes, rep_dim]

        image_features_exp = image_features_exp / image_features_exp.norm(dim=-1, keepdim=True)
        text_features_exp = text_features_exp / text_features_exp.norm(dim=-1, keepdim=True)

        logits_per_image: torch.Tensor = self.logit_scale * (image_features_exp * text_features_exp).sum(
            dim=-1
        )

        return logits_per_image