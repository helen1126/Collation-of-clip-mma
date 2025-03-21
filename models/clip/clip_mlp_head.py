import torch
from torch import nn

from models.clip.clip_base import ClipBase


class CLIPMLPHead(ClipBase):
    """Clip with a multimodal adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��CLIPMLPHead���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���ClipBase�Ĺ��캯�������ݹǸ�����͸�Ŀ¼������
            2. ��ȡCLIPģ���Ӿ������������ά�ȣ�����������������ά�ȣ�Ϊ���ά�ȵ���������
            3. �������ز��СΪ16��
            4. ����һ����ϲ㣬�����������Բ��һ��ReLU�������
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
        ��ȡģ���п�ѧϰ���������Ƽ��ϡ�

        ����:
            set[str]: ������ѧϰ�������Ƶļ��ϣ���ǰ������ "combination_layer"��
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(["combination_layer"])

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        ��ģ���ƶ���CPU�豸�ϡ�

        ����:
            1. ��CLIPģ���ƶ���CPU�豸�ϡ�
            2. ����ϲ��ƶ���CPU�豸�ϡ�
            3. ��CLIPģ�͵���������ת��Ϊfloat��
        """
        self._clip.to(torch.device("cpu"))
        self.combination_layer.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ���CUDA�豸�ϡ�

        ����:
            1. ����ϲ��ƶ���CUDA�豸�ϡ�
            2. ��CLIPģ���ƶ���CUDA�豸�ϡ�
        """
        self.combination_layer.to(torch.device("cuda"))
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
            3. ��ͼ���������ı���������������ת��Ϊfloat32��
            4. ��ͼ���������ı�����������չ��ʹ��ά��ƥ�䡣
            5. ����չ���ͼ���������ı����������һ��ά����ƴ�ӡ�
            6. ��ƴ�Ӻ���������뵽��ϲ���д�������ѹ���һ��ά�ȡ�
            7. ���ش�����logits������
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