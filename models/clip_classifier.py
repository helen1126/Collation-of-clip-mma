from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from models._base_model import BaseModel
from models.clip.clip_base import ClipBase


class ClipClassifier(BaseModel):
    def __init__(self, model_base: str, class_template: str | None = None) -> None:
        """
        初始化 ClipClassifier 类的实例。

        参数:
            model_base (str): 基础的 CLIP 模型名称。
            class_template (str | None, 可选): 用于生成类别提示的模板字符串。
                如果未提供，则默认为 "a photo of a {}"。
        """
        self._model = ClipBase(model_base)
        self._model.to_cpu()
        self._model.eval()

        self._class_propmts: list[str] | None = None

        self._class_template = class_template or "a photo of a {}"

    @property
    def batch_size(self) -> int:
        """
        获取用于推理的批量大小。

        返回:
            int: 批量大小，固定为 2。
        """
        return 2

    @property
    def transforms(self) -> Compose:
        """
        获取用于推理和训练的图像变换组合。

        返回:
            Compose: 包含图像变换操作的 `Compose` 对象，
                该对象来自基础的 CLIP 模型。
        """
        return self._model.transforms

    def reconfig_labels(self, labels: list[str]) -> None:
        """
        根据给定的标签重新配置模型的提示特征。

        该方法会根据提供的标签列表生成类别提示，并使用基础 CLIP 模型
        预计算这些提示的特征。

        参数:
            labels (list[str]): 包含类别标签的字符串列表。
        """
        prompts = self._build_class_prompt(labels)
        self._model.precompute_prompt_features(prompts)

    def _build_class_prompt(self, class_names: list[str]) -> list[str]:
        """
        根据类别名称和模板字符串生成类别提示列表。

        该方法使用预定义的类别模板字符串，将每个类别名称填充到模板中，
        生成对应的类别提示。

        参数:
            class_names (list[str]): 包含类别名称的字符串列表。

        返回:
            list[str]: 包含生成的类别提示的字符串列表。
        """
        class_template = self._class_template
        return [class_template.format(class_name) for class_name in class_names]

    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        为评估目的进行预测，并返回真实标签和预测标签。

        该方法接收一个数据加载器作为输入，遍历数据加载器中的每个批次，
        使用基础 CLIP 模型进行预测，并记录真实标签和预测标签。

        参数:
            x (DataLoader[Any]): 用于提供待预测数据的数据加载器。

        返回:
            tuple[NDArray[Any], NDArray[Any]]: 一个元组，
                第一个元素是真实标签的数组，第二个元素是预测标签的数组。
        """
        predictions = []
        targets = []
        for batch in tqdm(x):
            images, batch_targets = batch

            with torch.no_grad():
                logits_per_image = self._model.forward(images)
            probs = logits_per_image.softmax(dim=1)
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            targets.extend(batch_targets)
        return np.array(targets), np.array(predictions)