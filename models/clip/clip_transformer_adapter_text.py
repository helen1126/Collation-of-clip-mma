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
        ��ʼ��CLIPTransformerAdapterText���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���ClipBase�Ĺ��캯�������ݹǸ�����͸�Ŀ¼������
            2. ��_clipģ���л�ȡͼ����������洢��self.image_encoder�С�
            3. ��_clipģ���л�ȡlogit_scale���洢��self.logit_scale�С�
            4. ��ʼ��һ��MaskedMultiheadAttentionDownsampledģ����Ϊ���������洢��self.adapter�С�
        """
        # pass default arguments to the parent class
        super(CLIPTransformerAdapterText, self).__init__(backbone, root=root)

        self.image_encoder = self._clip.visual
        self.logit_scale = self._clip.logit_scale
        self.adapter = MaskedMultiheadAttentionDownsampled()

    @property
    def learnable_param_names(self) -> set[str]:
        """
        ��ȡģ���п�ѧϰ���������Ƽ��ϡ�

        ����:
            set[str]: ������ѧϰ�������Ƶļ��ϣ���ǰ������ "adapter"��
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["adapter"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        ��ģ���ƶ���CPU�豸�ϡ�

        ����:
            1. �������е�_clipģ���ƶ���CPU�豸�ϡ�
            2. ��self.adapterģ���ƶ���CPU�豸�ϡ�
            3. ��_clipģ�͵���������ת��Ϊfloat��
        """
        self._clip.to(torch.device("cpu"))
        self.adapter.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ���CUDA�豸�ϡ�

        ����:
            1. ��self.adapterģ���ƶ���CUDA�豸�ϡ�
            2. �������е�_clipģ���ƶ���CUDA�豸�ϡ�
        """
        self.adapter.to(torch.device("cuda"))
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
            2. ����encode_images�������������ͼ��õ�ͼ����������������������ת��Ϊfloat32��
            3. �����ı�������ͼ��������ά�ȣ�ʹ���ܹ�����ƴ�ӡ�
            4. ���ı�������ͼ�������ڵ�0άƴ�ӣ��õ������������롣
            5. �����������봫��self.adapterģ����д����õ������������
            6. ������������з����ͼ���������ı�������
            7. �������������ͼ���������ı��������й�һ������
            8. ����ͼ���������ı��������ںϱ�����
            9. ��ԭʼ��ͼ���������ı��������й�һ������
            10. �����ںϱ�������ԭʼ��������������������������ںϡ�
            11. �ٴζ��ںϺ��ͼ���������ı��������й�һ������
            12. ����logit_scale��ָ��ֵ��
            13. ����ͼ���������ı������ĵ����������logit_scale���õ�logits��
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