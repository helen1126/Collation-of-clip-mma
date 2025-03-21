from models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPMhaAdapterMlpBottleneck(CLIPTransformerAdapterText):
    """
    Reproduction of CLIP-Adapter.

    This class inherits from CLIPTransformerAdapterText and adds a MaskedMultiheadAttentionDownsampled
    adapter with specific downsampling and upsampling types.
    """

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��CLIPMhaAdapterMlpBottleneck���ʵ����

        ����:
            backbone (str): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���CLIPTransformerAdapterText�Ĺ��캯�������ݹǸ�����͸�Ŀ¼������
            2. ��ʼ��һ��MaskedMultiheadAttentionDownsampled���͵���������
               �������Ϊ "mha"���²�������Ϊ "mlp"���ϲ�������Ϊ "mlp"��
        """
        # pass default arguments to the parent class
        super(CLIPMhaAdapterMlpBottleneck, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(
            block_type="mha", downsampling_type="mlp", upsampling_type="mlp"
        )