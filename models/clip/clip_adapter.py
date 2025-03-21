import torch
from torch import nn

from models.clip.clip_base import ClipBase

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        """
        初始化Adapter模块。

        参数:
            c_in (int): 输入特征的维度。
            reduction (int, 可选): 降维比例，默认为4。

        功能:
            1. 调用父类nn.Module的构造函数。
            2. 初始化一个包含两个线性层和两个ReLU激活函数的顺序模块self.fc，
               第一个线性层将输入维度c_in降维到c_in // reduction，
               第二个线性层将维度恢复到c_in。
        """
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入的张量。

        返回:
            torch.Tensor: 经过self.fc模块处理后的输出张量。
        """
        x = self.fc(x)
        return x

class CLIPAdapter(ClipBase):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化CLIPAdapter类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类ClipBase的构造函数，将骨干网络和根目录参数传递给父类。
            2. 将CLIP模型的视觉编码器赋值给self.image_encoder。
            3. 将CLIP模型的logit_scale赋值给self.logit_scale。
            4. 初始化一个Adapter模块，输入维度为512，降维比例为4，并将其转换为torch.float32类型。
        """
        # pass default arguments to the parent class
        super(CLIPAdapter, self).__init__(backbone, root=root)

        self.image_encoder = self._clip.visual
        self.logit_scale = self._clip.logit_scale
        self.adapter = Adapter(512, 4).to(torch.float32)

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
            1. 将CLIP模型移动到CPU设备上。
            2. 将Adapter模块移动到CPU设备上。
            3. 将CLIP模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self.adapter.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            1. 将Adapter模块移动到CUDA设备上。
            2. 将CLIP模型移动到CUDA设备上。
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
            torch.Tensor: 图像相对于文本的logits张量，形状为 [batch_size, n_classes]。

        功能:
            1. 根据输入的prompts或预计算的文本特征获取文本特征。
            2. 对输入的图像进行编码，得到图像特征。
            3. 将图像特征输入到Adapter模块中得到处理后的特征x。
            4. 按照一定比例将处理后的特征x与原始图像特征融合。
            5. 对图像特征和文本特征进行归一化处理。
            6. 计算logit_scale的指数。
            7. 计算图像特征与文本特征的转置的矩阵乘积，并乘以logit_scale，得到logits。
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

        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_per_image: torch.Tensor = logit_scale * image_features @ text_features.t()

        return logits_per_image