from models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPMhaAdapter(CLIPTransformerAdapterText):
    """
    ����CLIP-Adapterģ��

    ����̳���CLIPTransformerAdapterText����������������һ�������ض������͵������ͷע�����²�����������
    """

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        ��ʼ��CLIPMhaAdapter���ʵ����

        ����:
            backbone (str, ��ѡ): ģ��ʹ�õĹǸ��������ƣ�Ĭ��Ϊ "ViT-B/16"��
            root (str, ��ѡ): ���ݴ洢�ĸ�Ŀ¼��Ĭ��Ϊ "./data"��

        ����:
            1. ���ø���CLIPTransformerAdapterText�Ĺ��캯�������Ǹ�����͸�Ŀ¼�������ݸ����ࡣ
            2. ��ʼ��һ��MaskedMultiheadAttentionDownsampled���͵��������������������Ϊ "mha"��
        """
        # ��Ĭ�ϲ������ݸ�����
        super(CLIPMhaAdapter, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(block_type="mha")