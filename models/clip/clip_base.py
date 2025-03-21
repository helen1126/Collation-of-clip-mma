from typing import Self

import torch
from clip import clip
from PIL.Image import Image
from torch import nn
from torchvision.transforms import Compose


class ClipBase(nn.Module):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化ClipBase类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类nn.Module的构造函数。
            2. 存储数据根目录到self._root。
            3. 调用私有方法_load_clip_to_cpu加载CLIP模型到CPU，并将模型和图像转换函数分别存储到self._clip和self._transforms。
            4. 计算并存储logit_scale，即CLIP模型的logit_scale的指数并分离梯度。
            5. 初始化预计算的文本提示特征为None。
        """
        super(ClipBase, self).__init__()
        self._root = root
        self._clip, self._transforms = self._load_clip_to_cpu(backbone)

        self.logit_scale = self._clip.logit_scale.exp().detach()

        self._precomputed_prompt_features: torch.Tensor | None = None

    @property
    def learnable_param_names(self) -> set[str]:
        """
        获取模型中可学习参数的名称集合。

        返回:
            set[str]: 一个空集合，表示当前没有可学习的参数。
        """
        return set()

    @property
    def transforms(self) -> Compose:
        """
        获取图像转换函数。

        返回:
            Compose: 用于图像预处理的转换函数组合。
        """
        return self._transforms

    def eval(self) -> Self:
        """
        将模型设置为评估模式。

        返回:
            Self: 返回当前对象的引用，支持链式调用。

        功能:
            将CLIP模型设置为评估模式。
        """
        self._clip.eval()
        return self

    def train_(self) -> Self:
        """
        将模型设置为训练模式。

        返回:
            Self: 返回当前对象的引用，支持链式调用。

        功能:
            将CLIP模型设置为训练模式。
        """
        self._clip.train()
        return self

    def transform(self, images: list[Image]) -> torch.Tensor:
        """
        对输入的图像列表进行转换。

        参数:
            images (list[Image]): 输入的图像列表。

        返回:
            torch.Tensor: 转换后的图像张量。
        """
        output: torch.Tensor = self._transforms(images)
        return output

    def to_cpu(self) -> None:
        """
        将模型移动到CPU设备上。

        功能:
            1. 将CLIP模型移动到CPU设备上。
            2. 将CLIP模型的数据类型转换为float。
        """
        self._clip.to(torch.device("cpu"))
        self._clip.float()

    def to_mps(self) -> None:
        """
        将模型移动到MPS设备上。

        功能:
            将CLIP模型移动到MPS设备上。
        """
        self._clip.to(torch.device("mps"))

    def to_cuda(self) -> None:
        """
        将模型移动到CUDA设备上。

        功能:
            将CLIP模型移动到CUDA设备上。
        """
        self._clip.to(torch.device("cuda"))

    @property
    def device(self) -> torch.device:
        """
        获取模型所在的设备。

        返回:
            torch.device: 模型当前所在的设备。
        """
        return next(self.parameters()).device

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
            3. 计算图像特征与文本特征转置的矩阵乘积，并乘以logit_scale，得到logits。
        """
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)

        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        对输入的图像进行编码。

        参数:
            images (torch.Tensor): 输入的图像张量。

        返回:
            torch.Tensor: 编码后的图像特征张量，特征已进行归一化处理。

        功能:
            1. 在无梯度计算的上下文中，将图像张量移动到模型所在设备，并使用CLIP模型的encode_image方法进行编码。
            2. 对编码后的图像特征进行归一化处理。
        """
        with torch.no_grad():
            image_features: torch.Tensor = self._clip.encode_image(images.to(self.device))
            image_features /= image_features.norm(dim=1, keepdim=True)

        return image_features

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """
        对输入的文本提示进行编码。

        参数:
            prompts (list[str]): 输入的文本提示列表。

        返回:
            torch.Tensor: 编码后的文本特征张量，特征已进行归一化处理。

        功能:
            1. 将输入的文本提示进行分词，并将分词结果拼接成一个张量，然后移动到模型所在设备。
            2. 在无梯度计算的上下文中，使用CLIP模型的encode_text方法对文本进行编码。
            3. 对编码后的文本特征进行归一化处理。
        """
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(self.device)

        with torch.no_grad():
            text_features: torch.Tensor = self._clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def precompute_prompt_features(self, prompts: list[str]) -> None:
        """
        预计算文本提示的特征。

        参数:
            prompts (list[str]): 输入的文本提示列表。

        功能:
            调用encode_text方法对文本提示进行编码，并将编码后的特征存储到self._precomputed_prompt_features。
        """
        self._precomputed_prompt_features = self.encode_text(prompts)

    def _load_clip_to_cpu(self, backbone: str) -> tuple[nn.Module, Compose]:
        """
        加载CLIP模型到CPU。

        参数:
            backbone (str): 模型使用的骨干网络名称。

        返回:
            tuple[nn.Module, Compose]: 加载好的CLIP模型和图像转换函数。

        功能:
            1. 根据骨干网络名称获取模型的下载链接，如果名称无效则抛出KeyError异常。
            2. 下载模型文件到指定的根目录。
            3. 尝试以JIT模式加载模型，如果失败则加载模型的状态字典。
            4. 使用状态字典构建CLIP模型。
            5. 获取模型的输入分辨率，并创建对应的图像转换函数。
        """
        try:
            url = clip._MODELS[backbone]
        except KeyError:
            raise KeyError(f"Invalid backbone {backbone} selected.")

        model_path = clip._download(url, self._root)

        try:
            jit_model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model: nn.Module = clip.build_model(state_dict or jit_model.state_dict())
        return model, clip._transform(jit_model.input_resolution.item())