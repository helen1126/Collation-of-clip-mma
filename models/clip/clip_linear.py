import torch
from torch import nn

from models.clip.clip_base import ClipBase



class ClipLinear(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��ClipLinear���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���ClipBase�Ĺ��캯�������Ǹ�����͸�Ŀ¼�������ݸ����ࡣ
            2. ��ʼ��һ�����Բ�self.image_linear����������ά�Ⱦ�ΪCLIPģ���Ӿ������������ά�ȡ�
        """
        super(ClipLinear, self).__init__(backbone, root=root)

        self.image_linear = nn.Sequential(
            nn.Linear(self._clip.visual.output_dim, self._clip.visual.output_dim)
        )

    @property
    def learnable_param_names(self) -> set[str]:
        """
        ��ȡģ���п�ѧϰ���������Ƽ��ϡ�

        ����:
            set[str]: ������ѧϰ�������Ƶļ��ϣ���ǰ������ "image_linear"��
        """
        return set(["image_linear"])

    def to_cpu(self) -> None:
        """
        ��ģ���ƶ���CPU�豸�ϡ�

        ����:
            1. ��CLIPģ���ƶ���CPU�豸�ϡ�
            2. ��self.image_linear���Բ��ƶ���CPU�豸�ϡ�
            3. ��CLIPģ�͵���������ת��Ϊfloat��
        """
        self._clip.to(torch.device("cpu"))
        self.image_linear.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ���CUDA�豸�ϡ�

        ����:
            1. ��self.image_linear���Բ��ƶ���CUDA�豸�ϡ�
            2. ��CLIPģ���ƶ���CUDA�豸�ϡ�
        """
        self.image_linear.to(torch.device("cuda"))
        self._clip.to(torch.device("cuda"))

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
            3. ��ͼ���������뵽self.image_linear���Բ���д���
            4. ����ͼ���������ı�����ת�õľ���˻���������logit_scale���õ�logits��
        """
        if prompts:
            text_features = self.encode_text(prompts)
        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features
        else:
            raise ValueError("At least one prompts or pre-computed promt features has to be present.")

        image_features = self.encode_images(images)

        image_features = self.image_linear(image_features)

        logits_per_image: torch.Tensor = self.logit_scale * image_features @ text_features.t()

        return logits_per_image