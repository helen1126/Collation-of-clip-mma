import torch
from torch import Tensor, nn

from models.clip.clip_base import ClipBase
from models.clip.modules.masked_multi_head_attention import (
    MaskedMultiheadAttention,
)


class HyperNet(nn.Module):
    def __init__(self, embed_dim: int = 512) -> None:
        """
        ��ʼ�� HyperNet ���ʵ����

        ����:
            embed_dim (int, ��ѡ): Ƕ��ά�ȣ�Ĭ��Ϊ 512��

        ����:
            1. ���ø��� nn.Module �Ĺ��캯����
            2. �洢Ƕ��ά�� embed_dim��
            3. �����ɳ�����������������Ȩ����״ weight_shapes��
            4. ���ó���������ز��С hidden_size��
            5. ��������Ȩ�ص����� num_weights��
            6. ��ʼ������ȫ���Ӳ� fc1 �� fc2��
        """
        super(HyperNet, self).__init__()

        self.embed_dim = embed_dim
        self.weight_shapes = [
            (embed_dim, 4),
            (4, embed_dim),
        ]  # Shape of the network parametrized by hypernetwork
        self.hidden_size = 16  # hypernetwork hidden size
        num_weights = sum(w[0] * w[1] for w in self.weight_shapes)

        self.fc1 = nn.Linear(embed_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        ǰ�򴫲�������

        ����:
            x (Tensor): ����������

        ����:
            Tensor: ���������紦�������������

        ����:
            1. ���� generate_weights ��������Ȩ�ء�
            2. ���� apply_main_network ���������ɵ�Ȩ��Ӧ�õ��������С�
        """
        weights = self.generate_weights(x)
        return self.apply_main_network(x, weights)

    def generate_weights(self, x: Tensor) -> Tensor:
        """
        ����Ȩ�صĺ�����

        ����:
            x (Tensor): ����������

        ����:
            Tensor: ���ɵ�Ȩ��������

        ����:
            1. ���������� x ����ȫ���Ӳ� fc1��
            2. �� fc1 �����Ӧ�� ReLU �������
            3. �������Ľ������ȫ���Ӳ� fc2���õ�Ȩ�ء�
        """
        x = self.fc1(x)
        x = torch.relu(x)
        weights = self.fc2(x)
        return weights

    def apply_main_network(self, x: Tensor, weights: Tensor) -> Tensor:
        """
        �����ɵ�Ȩ��Ӧ�õ�������ĺ�����

        ����:
            x (Tensor): ����������
            weights (Tensor): ���ɵ�Ȩ��������

        ����:
            Tensor: ���������紦�������������

        ����:
            1. ��ʼ��Ȩ����Ƭ����ʼ���� start Ϊ 0��
            2. ����Ȩ����״�б� weight_shapes��
            3. ���㵱ǰ��Ȩ�صĽ������� end��
            4. ��Ȩ����������ȡ��ǰ���Ȩ�أ���������״��
            5. ʹ�� einsum �������о���˷��������������͵�ǰ��Ȩ����ˡ�
            6. ����˽��Ӧ�� ReLU �������
            7. ������ʼ���� start Ϊ�������� end��
            8. �������յ����������
        """
        start = 0

        for shape in self.weight_shapes:
            layer_in_size, layer_out_size = shape[0], shape[1]
            end = start + layer_in_size * layer_out_size

            layer_weights = weights[:, :, start:end].view(
                weights.shape[0], weights.shape[1], layer_in_size, layer_out_size
            )

            x = torch.einsum("bce,bcej->bcj", x, layer_weights)
            x = torch.relu(x)

            start = end

        return x


class ClipTransformerWHypernet(ClipBase):
    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ�� ClipTransformerWHypernet ���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø��� ClipBase �Ĺ��캯�������ݹǸ�����͸�Ŀ¼������
            2. ���ý�ά���ά�� down_dim Ϊ 32��
            3. ��ʼ��һ�� MaskedMultiheadAttention ģ�飬�洢�� self.mmha �С�
            4. ��ʼ��ͼ����ı����²������Բ㣬�ֱ�洢�� self.image_downsample �� self.text_downsample �С�
            5. ��ʼ��ͼ����ı����ϲ������Բ㣬�ֱ�洢�� self.image_upsample �� self.text_upsample �С�
            6. ��ʼ��ͼ��ĳ����磬�洢�� self.image_hypernetwork �С�
        """
        # pass default arguments to the parent class
        super(ClipTransformerWHypernet, self).__init__(backbone, root=root)

        down_dim = 32

        self.mmha = MaskedMultiheadAttention(embed_dim=down_dim)

        # Downsampling from 512 to down_dim
        self.image_downsample = nn.Linear(512, down_dim)
        self.text_downsample = nn.Linear(512, down_dim)

        # Upsampling from down_dim to 512
        self.image_upsample = nn.Linear(down_dim, 512)
        self.text_upsample = nn.Linear(down_dim, 512)

        # Hypernetworks
        self.image_hypernetwork = HyperNet(embed_dim=down_dim)
        # self.text_hypernetwork = HyperNet(embed_dim=down_dim)

    @property
    def learnable_param_names(self) -> set[str]:
        """
        ��ȡģ���п�ѧϰ���������Ƽ��ϡ�

        ����:
            set[str]: ������ѧϰ�������Ƶļ��ϣ����� "mmha"��"image_downsample"��
                      "text_downsample"��"image_upsample"��"text_upsample" �� "image_hypernetwork"��
        """
        # IMPORTANT: Add the name of the learnable parameters in the model
        return set(
            [
                "mmha",
                "image_downsample",
                "text_downsample",
                "image_upsample",
                "text_upsample",
                "image_hypernetwork",
                # "text_hypernetwork",
            ]
        )

    # If needed you can override the to_cpu and to_cuda methods
    def to_cpu(self) -> None:
        """
        ��ģ���ƶ��� CPU �豸�ϡ�

        ����:
            1. �������е� _clip ģ���ƶ��� CPU �豸�ϣ���������������ת��Ϊ float��
            2. �� self.mmha ģ���ƶ��� CPU �豸�ϡ�
            3. ��ͼ����ı����²������ϲ������Բ��ƶ��� CPU �豸�ϡ�
            4. ��ͼ��ĳ�����ģ���ƶ��� CPU �豸�ϡ�
        """
        self._clip.to(torch.device("cpu"))
        self._clip.float()
        self.mmha.to(torch.device("cpu"))
        self.image_downsample.to(torch.device("cpu"))
        self.text_downsample.to(torch.device("cpu"))
        self.image_upsample.to(torch.device("cpu"))
        self.text_upsample.to(torch.device("cpu"))
        self.image_hypernetwork.to(torch.device("cpu"))
        # self.text_hypernetwork.to(torch.device("cpu"))

    def to_mps(self) -> None:
        """
        ��ģ���ƶ��� MPS��ƻ���� Metal Performance Shaders���豸�ϡ�

        ����:
            1. �������е� _clip ģ���ƶ��� MPS �豸�ϡ�
            2. �� self.mmha ģ���ƶ��� MPS �豸�ϡ�
            3. ��ͼ����ı����²������ϲ������Բ��ƶ��� MPS �豸�ϡ�
            4. ��ͼ��ĳ�����ģ���ƶ��� MPS �豸�ϡ�
        """
        self._clip.to(torch.device("mps"))
        self.mmha.to(torch.device("mps"))
        self.image_downsample.to(torch.device("mps"))
        self.text_downsample.to(torch.device("mps"))
        self.image_upsample.to(torch.device("mps"))
        self.text_upsample.to(torch.device("mps"))
        self.image_hypernetwork.to(torch.device("mps"))
        # self.text_hypernetwork.to(torch.device("mps"))

    def to_cuda(self) -> None:
        """
        ��ģ���ƶ��� CUDA �豸�ϡ�

        ����:
            1. �������е� _clip ģ���ƶ��� CUDA �豸�ϡ�
            2. �� self.mmha ģ���ƶ��� CUDA �豸�ϡ�
            3. ��ͼ����ı����²������ϲ������Բ��ƶ��� CUDA �豸�ϡ�
            4. ��ͼ��ĳ�����ģ���ƶ��� CUDA �豸�ϡ�
        """
        self._clip.to(torch.device("cuda"))
        self.mmha.to(torch.device("cuda"))
        self.image_downsample.to(torch.device("cuda"))
        self.text_downsample.to(torch.device("cuda"))
        self.image_upsample.to(torch.device("cuda"))
        self.text_upsample.to(torch.device("cuda"))
        self.image_hypernetwork.to(torch.device("cuda"))
        # self.text_hypernetwork.to(torch.device("cuda"))

    def forward(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        """
        ǰ�򴫲�������

        ����:
            images (torch.Tensor): �����ͼ��������
            prompts (list[str] | None, ��ѡ): ������ı���ʾ�б�Ĭ��Ϊ None��

        ����:
            torch.Tensor: ͼ��������ı��� logits ������

        ����:
            1. ����ṩ�� prompts������� encode_text ���������ı���ʾ�õ��ı�������
               �������Ԥ������ı���ʾ��������ʹ��Ԥ�����������
               �����׳� ValueError �쳣��
            2. ��ȡͼ���������С�����������
            3. ���� encode_images �������������ͼ��õ�ͼ����������������ά�Ⱥ��������͡�
            4. �����ı�������ά�Ⱥ��������͡�
            5. ��ͼ���������ı����������²�����
            6. ���²�������ı�������ͼ�������ڵ� 0 άƴ�ӣ��õ��������С�
            7. ���������д��� self.mmha ģ����д���
            8. �Ӵ������з����ͼ���������ı���������������ά�ȡ�
            9. ��ͼ����������ͼ��������д���
            10. �Դ�����ͼ���������ı����������ϲ�����
            11. ����ͼ��������ά�ȡ�
            12. ���ϲ������������ԭʼ������ӡ�
            13. ����ͼ��������ı��� logits��
        """
        if prompts:
            text_features = self.encode_text(prompts)

        elif self._precomputed_prompt_features is not None:
            text_features = self._precomputed_prompt_features

        else:
            raise ValueError("At least one prompts or pre-computed prompt features has to be present.")

        batch_size = images.shape[0]
        num_classes = text_features.shape[0]

        image_features = self.encode_images(images)

        image_features = image_features.to(torch.float32).unsqueeze(0)

        text_features = text_features.to(torch.float32).unsqueeze(1).expand(-1, batch_size, -1)

        _image_features = self.image_downsample(image_features)
        _text_features = self.text_downsample(text_features)

        input_seq = torch.cat([_text_features, _image_features], dim=0)

        tr_outputs = self.mmha.forward(input_seq)

        #input_seq = input_seq + tr_outputs

        _image_features = tr_outputs[num_classes:]  # [1, batch_size, embed_dim]
        _text_features = tr_outputs[:num_classes]  # [n_classes, batch_size, embed_dim]

        _image_features = _image_features.permute(1, 0, 2)  # [batch_size, 1, embed_dim]
        _text_features = _text_features.permute(1, 0, 2)  # [batch_size, n_classes, embed_dim]

        _image_features = self.image_hypernetwork(_image_features)
        # _text_features = self.text_hypernetwork(_text_features)

        _image_features = self.image_upsample(_image_features)
        _text_features = self.text_upsample(_text_features)

        _image_features = _image_features.transpose(1, 2)

        text_features = text_features.permute(1, 0, 2) + _text_features
        image_features = image_features.permute(1, 2, 0) + _image_features

        logits_per_image: torch.Tensor = torch.bmm(text_features, image_features).squeeze(2)

        return logits_per_image