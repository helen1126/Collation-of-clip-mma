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
        ��ʼ�� ClipClassifier ���ʵ����

        ����:
            model_base (str): ������ CLIP ģ�����ơ�
            class_template (str | None, ��ѡ): �������������ʾ��ģ���ַ�����
                ���δ�ṩ����Ĭ��Ϊ "a photo of a {}"��
        """
        self._model = ClipBase(model_base)
        self._model.to_cpu()
        self._model.eval()

        self._class_propmts: list[str] | None = None

        self._class_template = class_template or "a photo of a {}"

    @property
    def batch_size(self) -> int:
        """
        ��ȡ���������������С��

        ����:
            int: ������С���̶�Ϊ 2��
        """
        return 2

    @property
    def transforms(self) -> Compose:
        """
        ��ȡ���������ѵ����ͼ��任��ϡ�

        ����:
            Compose: ����ͼ��任������ `Compose` ����
                �ö������Ի����� CLIP ģ�͡�
        """
        return self._model.transforms

    def reconfig_labels(self, labels: list[str]) -> None:
        """
        ���ݸ����ı�ǩ��������ģ�͵���ʾ������

        �÷���������ṩ�ı�ǩ�б����������ʾ����ʹ�û��� CLIP ģ��
        Ԥ������Щ��ʾ��������

        ����:
            labels (list[str]): ��������ǩ���ַ����б�
        """
        prompts = self._build_class_prompt(labels)
        self._model.precompute_prompt_features(prompts)

    def _build_class_prompt(self, class_names: list[str]) -> list[str]:
        """
        ����������ƺ�ģ���ַ������������ʾ�б�

        �÷���ʹ��Ԥ��������ģ���ַ�������ÿ�����������䵽ģ���У�
        ���ɶ�Ӧ�������ʾ��

        ����:
            class_names (list[str]): ����������Ƶ��ַ����б�

        ����:
            list[str]: �������ɵ������ʾ���ַ����б�
        """
        class_template = self._class_template
        return [class_template.format(class_name) for class_name in class_names]

    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Ϊ����Ŀ�Ľ���Ԥ�⣬��������ʵ��ǩ��Ԥ���ǩ��

        �÷�������һ�����ݼ�������Ϊ���룬�������ݼ������е�ÿ�����Σ�
        ʹ�û��� CLIP ģ�ͽ���Ԥ�⣬����¼��ʵ��ǩ��Ԥ���ǩ��

        ����:
            x (DataLoader[Any]): �����ṩ��Ԥ�����ݵ����ݼ�������

        ����:
            tuple[NDArray[Any], NDArray[Any]]: һ��Ԫ�飬
                ��һ��Ԫ������ʵ��ǩ�����飬�ڶ���Ԫ����Ԥ���ǩ�����顣
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