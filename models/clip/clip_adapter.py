import torch
from torch import nn

from models.clip.clip_base import ClipBase

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        """
        ��ʼ��Adapterģ�顣

        ����:
            c_in (int): ����������ά�ȡ�
            reduction (int, ��ѡ): ��ά������Ĭ��Ϊ4��

        ����:
            1. ���ø���nn.Module�Ĺ��캯����
            2. ��ʼ��һ�������������Բ������ReLU�������˳��ģ��self.fc��
               ��һ�����Բ㽫����ά��c_in��ά��c_in // reduction��
               �ڶ������Բ㽫ά�Ȼָ���c_in��
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
        ǰ�򴫲�������

        ����:
            x (torch.Tensor): �����������

        ����:
            torch.Tensor: ����self.fcģ�鴦�������������
        """
        x = self.fc(x)
        return x

class CLIPAdapter(ClipBase):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��CLIPAdapter���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���ClipBase�Ĺ��캯�������Ǹ�����͸�Ŀ¼�������ݸ����ࡣ
            2. ��CLIPģ�͵��Ӿ���������ֵ��self.image_encoder��
            3. ��CLIPģ�͵�logit_scale��ֵ��self.logit_scale��
            4. ��ʼ��һ��Adapterģ�飬����ά��Ϊ512����ά����Ϊ4��������ת��Ϊtorch.float32���͡�
        """
        # pass default arguments to the parent class
        super(CLIPAdapter, self).__init__(backbone, root=root)

        self.image_encoder = self._clip.visual
        self.logit_scale = self._clip.logit_scale
        self.adapter = Adapter(512, 4).to(torch.float32)

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
            1. ��CLIPģ���ƶ���CPU�豸�ϡ�
            2. ��Adapterģ���ƶ���CPU�豸�ϡ�
            3. ��CLIPģ�͵���������ת��Ϊfloat��
        """
        self._clip.to(torch.device("cpu"))
        self.adapter.to(torch.device("cpu"))
        self._clip.float()

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ���CUDA�豸�ϡ�

        ����:
            1. ��Adapterģ���ƶ���CUDA�豸�ϡ�
            2. ��CLIPģ���ƶ���CUDA�豸�ϡ�
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
            torch.Tensor: ͼ��������ı���logits��������״Ϊ [batch_size, n_classes]��

        ����:
            1. ���������prompts��Ԥ������ı�������ȡ�ı�������
            2. �������ͼ����б��룬�õ�ͼ��������
            3. ��ͼ���������뵽Adapterģ���еõ�����������x��
            4. ����һ������������������x��ԭʼͼ�������ںϡ�
            5. ��ͼ���������ı��������й�һ������
            6. ����logit_scale��ָ����
            7. ����ͼ���������ı�������ת�õľ���˻���������logit_scale���õ�logits��
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