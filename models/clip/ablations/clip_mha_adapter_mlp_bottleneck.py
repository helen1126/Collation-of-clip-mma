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
        初始化CLIPMhaAdapterMlpBottleneck类的实例。

        参数:
            backbone (str): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类CLIPTransformerAdapterText的构造函数，传递骨干网络和根目录参数。
            2. 初始化一个MaskedMultiheadAttentionDownsampled类型的适配器，
               其块类型为 "mha"，下采样类型为 "mlp"，上采样类型为 "mlp"。
        """
        # pass default arguments to the parent class
        super(CLIPMhaAdapterMlpBottleneck, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(
            block_type="mha", downsampling_type="mlp", upsampling_type="mlp"
        )