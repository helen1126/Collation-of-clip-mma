from models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPTransformerAdapterMlpBottleneck(CLIPTransformerAdapterText):
    """
    复现CLIP-Adapter模型。该类继承自CLIPTransformerAdapterText，通过添加一个具有特定类型的
    MaskedMultiheadAttentionDownsampled适配器来增强模型能力。
    """

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        """
        初始化CLIPTransformerAdapterMlpBottleneck类的实例。

        参数:
            backbone (str, 可选): 模型使用的骨干网络名称，默认为 "ViT-B/16"。
            root (str, 可选): 数据存储的根目录，默认为 "./data"。

        功能:
            1. 调用父类CLIPTransformerAdapterText的构造函数，将骨干网络和根目录参数传递给父类。
            2. 初始化一个MaskedMultiheadAttentionDownsampled类型的适配器，设置其块类型为 "transformer"，
               下采样类型为 "mlp"，上采样类型为 "mlp"。
        """
        # 将默认参数传递给父类
        super(CLIPTransformerAdapterMlpBottleneck, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(
            block_type="transformer", downsampling_type="mlp", upsampling_type="mlp"
        )