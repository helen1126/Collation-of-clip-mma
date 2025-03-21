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
        ��ʼ��ClipTransformer���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���ClipBase�Ĺ��캯�������ݹǸ�����͸�Ŀ¼������
            2. ��ʼ��һ��MaskedMultiheadAttentionģ�飬�洢��self.mmha�С�
        """
        # pass default arguments to the parent class
        super(ClipTransformer, self).__init__(backbone, root=root)

        self.mmha = MaskedMultiheadAttention()

    @property
    def learnable_param_names(self) -> set[str]:
        """
        ��ȡģ���п�ѧϰ���������Ƽ��ϡ�

        ����:
            set[str]: ������ѧϰ�������Ƶļ��ϣ���ǰ������ "mmha"��
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["mmha"])

    def eval(self) -> Self:
        """
        ��ģ������Ϊ����ģʽ��

        ����:
            Self: ���ص�ǰģ��ʵ����

        ����:
            1. �������е�_clipģ������Ϊ����ģʽ��
            2. ��self.mmhaģ������Ϊ����ģʽ��
        """
        self._clip.eval()
        self.mmha.eval()
        return self

    def train_(self) -> Self:
        """
        ��ģ������Ϊѵ��ģʽ��

        ����:
            Self: ���ص�ǰģ��ʵ����

        ����:
            1. �������е�_clipģ������Ϊѵ��ģʽ��
            2. ��self.mmhaģ������Ϊѵ��ģʽ��
        """
        self._clip.train()
        self.mmha.train()
        return self

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        ��ģ���ƶ���CPU�豸�ϡ�

        ����:
            1. �������е�_clipģ���ƶ���CPU�豸�ϡ�
            2. ��self.mmhaģ���ƶ���CPU�豸�ϡ�
            3. ��_clipģ�͵���������ת��Ϊfloat��
        """
        self._clip.to(torch.device("cpu"))
        self.mmha.to(torch.device("cpu"))
        self._clip.float()

    def to_mps(self) -> None:
        """
        ��ģ���ƶ���MPS��ƻ����Metal Performance Shaders���豸�ϡ�

        ����:
            1. �������е�_clipģ���ƶ���MPS�豸�ϡ�
            2. ��self.mmhaģ���ƶ���MPS�豸�ϡ�
        """
        self._clip.to(torch.device("mps"))
        self.mmha.to(torch.device("mps"))

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ���CUDA�豸�ϡ�

        ����:
            1. ��self.mmhaģ���ƶ���CUDA�豸�ϡ�
            2. �������е�_clipģ���ƶ���CUDA�豸�ϡ�
        """
        self.mmha.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        """
        ǰ�򴫲�������

        ����:
            images (torch.Tensor): �����ͼ��������
            prompts (list[str] | None, ��ѡ): ������ı���ʾ�б�Ĭ��ΪNone��

        ����:
            torch.Tensor: ͼ��������ı���logits������

        ����:
            1. ����ṩ��prompts�������encode_text���������ı���ʾ�õ��ı�������
               �������Ԥ������ı���ʾ��������ʹ��Ԥ�����������
               �����׳�ValueError�쳣��
            2. ����encode_images�������������ͼ��õ�ͼ����������������ά�Ⱥ��������͡�
            3. �����ı�������ά�Ⱥ��������͡�
            4. �������������
            5. ���ı�������ͼ�������ڵ�0άƴ�ӣ��õ��������С�
            6. ���������д���self.mmhaģ����д���
            7. ����ͼ���������ı�����������ͼ��������ı���logits��
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