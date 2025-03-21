from typing import Self

import torch
from clip import clip
from PIL.Image import Image
from torch import nn
from torchvision.transforms import Compose


class ClipBase(nn.Module):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��ClipBase���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���nn.Module�Ĺ��캯����
            2. �洢���ݸ�Ŀ¼��self._root��
            3. ����˽�з���_load_clip_to_cpu����CLIPģ�͵�CPU������ģ�ͺ�ͼ��ת�������ֱ�洢��self._clip��self._transforms��
            4. ���㲢�洢logit_scale����CLIPģ�͵�logit_scale��ָ���������ݶȡ�
            5. ��ʼ��Ԥ������ı���ʾ����ΪNone��
        """
        super(ClipBase, self).__init__()
        self._root = root
        self._clip, self._transforms = self._load_clip_to_cpu(backbone)

        self.logit_scale = self._clip.logit_scale.exp().detach()

        self._precomputed_prompt_features: torch.Tensor | None = None

    @property
    def learnable_param_names(self) -> set[str]:
        """
        ��ȡģ���п�ѧϰ���������Ƽ��ϡ�

        ����:
            set[str]: һ���ռ��ϣ���ʾ��ǰû�п�ѧϰ�Ĳ�����
        """
        return set()

    @property
    def transforms(self) -> Compose:
        """
        ��ȡͼ��ת��������

        ����:
            Compose: ����ͼ��Ԥ�����ת��������ϡ�
        """
        return self._transforms

    def eval(self) -> Self:
        """
        ��ģ������Ϊ����ģʽ��

        ����:
            Self: ���ص�ǰ��������ã�֧����ʽ���á�

        ����:
            ��CLIPģ������Ϊ����ģʽ��
        """
        self._clip.eval()
        return self

    def train_(self) -> Self:
        """
        ��ģ������Ϊѵ��ģʽ��

        ����:
            Self: ���ص�ǰ��������ã�֧����ʽ���á�

        ����:
            ��CLIPģ������Ϊѵ��ģʽ��
        """
        self._clip.train()
        return self

    def transform(self, images: list[Image]) -> torch.Tensor:
        """
        �������ͼ���б����ת����

        ����:
            images (list[Image]): �����ͼ���б�

        ����:
            torch.Tensor: ת�����ͼ��������
        """
        output: torch.Tensor = self._transforms(images)
        return output

    def to_cpu(self) -> None:
        """
        ��ģ���ƶ���CPU�豸�ϡ�

        ����:
            1. ��CLIPģ���ƶ���CPU�豸�ϡ�
            2. ��CLIPģ�͵���������ת��Ϊfloat��
        """
        self._clip.to(torch.device("cpu"))
        self._clip.float()

    def to_mps(self) -> None:
        """
        ��ģ���ƶ���MPS�豸�ϡ�

        ����:
            ��CLIPģ���ƶ���MPS�豸�ϡ�
        """
        self._clip.to(torch.device("mps"))

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ���CUDA�豸�ϡ�

        ����:
            ��CLIPģ���ƶ���CUDA�豸�ϡ�
        """
        self._clip.to(torch.device("cuda"))

    @property
    def device(self) -> torch.device:
        """
        ��ȡģ�����ڵ��豸��

        ����:
            torch.device: ģ�͵�ǰ���ڵ��豸��
        """
        return next(self.parameters()).device

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        """
        ǰ�򴫲�������

        ����:
            images (torch.Tensor): �����ͼ��������
            prompts (list[str] | None, ��ѡ): ������ı���ʾ�б�Ĭ��ΪNone��

        ����:
            torch.Tensor: ͼ��������ı���logits��������״Ϊ [batch_size, n_classes]��

        ����:
            1. ����ṩ��prompts�������encode_text���������ı���ʾ�õ��ı�������
               �������Ԥ������ı���ʾ��������ʹ��Ԥ�����������
               �����׳�ValueError�쳣��
            2. ����encode_images�������������ͼ��õ�ͼ��������
            3. ����ͼ���������ı�����ת�õľ���˻���������logit_scale���õ�logits��
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
        �������ͼ����б��롣

        ����:
            images (torch.Tensor): �����ͼ��������

        ����:
            torch.Tensor: ������ͼ�����������������ѽ��й�һ������

        ����:
            1. �����ݶȼ�����������У���ͼ�������ƶ���ģ�������豸����ʹ��CLIPģ�͵�encode_image�������б��롣
            2. �Ա�����ͼ���������й�һ������
        """
        with torch.no_grad():
            image_features: torch.Tensor = self._clip.encode_image(images.to(self.device))
            image_features /= image_features.norm(dim=1, keepdim=True)

        return image_features

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """
        ��������ı���ʾ���б��롣

        ����:
            prompts (list[str]): ������ı���ʾ�б�

        ����:
            torch.Tensor: �������ı����������������ѽ��й�һ������

        ����:
            1. ��������ı���ʾ���зִʣ������ִʽ��ƴ�ӳ�һ��������Ȼ���ƶ���ģ�������豸��
            2. �����ݶȼ�����������У�ʹ��CLIPģ�͵�encode_text�������ı����б��롣
            3. �Ա������ı��������й�һ������
        """
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(self.device)

        with torch.no_grad():
            text_features: torch.Tensor = self._clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def precompute_prompt_features(self, prompts: list[str]) -> None:
        """
        Ԥ�����ı���ʾ��������

        ����:
            prompts (list[str]): ������ı���ʾ�б�

        ����:
            ����encode_text�������ı���ʾ���б��룬���������������洢��self._precomputed_prompt_features��
        """
        self._precomputed_prompt_features = self.encode_text(prompts)

    def _load_clip_to_cpu(self, backbone: str) -> tuple[nn.Module, Compose]:
        """
        ����CLIPģ�͵�CPU��

        ����:
            backbone (str): ģ��ʹ�õĹǸ��������ơ�

        ����:
            tuple[nn.Module, Compose]: ���غõ�CLIPģ�ͺ�ͼ��ת��������

        ����:
            1. ���ݹǸ��������ƻ�ȡģ�͵��������ӣ����������Ч���׳�KeyError�쳣��
            2. ����ģ���ļ���ָ���ĸ�Ŀ¼��
            3. ������JITģʽ����ģ�ͣ����ʧ�������ģ�͵�״̬�ֵ䡣
            4. ʹ��״̬�ֵ乹��CLIPģ�͡�
            5. ��ȡģ�͵�����ֱ��ʣ���������Ӧ��ͼ��ת��������
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