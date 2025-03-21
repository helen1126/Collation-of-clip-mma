from models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPTransformerAdapterMlpBottleneck(CLIPTransformerAdapterText):
    """
    ����CLIP-Adapterģ�͡�����̳���CLIPTransformerAdapterText��ͨ�����һ�������ض����͵�
    MaskedMultiheadAttentionDownsampled����������ǿģ��������
    """

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��CLIPTransformerAdapterMlpBottleneck���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���CLIPTransformerAdapterText�Ĺ��캯�������Ǹ�����͸�Ŀ¼�������ݸ����ࡣ
            2. ��ʼ��һ��MaskedMultiheadAttentionDownsampled���͵��������������������Ϊ "transformer"��
               �²�������Ϊ "mlp"���ϲ�������Ϊ "mlp"��
        """
        # ��Ĭ�ϲ������ݸ�����
        super(CLIPTransformerAdapterMlpBottleneck, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(
            block_type="transformer", downsampling_type="mlp", upsampling_type="mlp"
        )