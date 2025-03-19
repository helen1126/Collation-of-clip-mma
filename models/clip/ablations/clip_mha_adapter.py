from models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from models.clip.modules.masked_multi_head_attention_downsampled import (
    MaskedMultiheadAttentionDownsampled,
)


class CLIPMhaAdapter(CLIPTransformerAdapterText):
    """Reproduction of CLIP-Adapter"""

    def __init__(self, backbone: str = "ViT-B/16", root: str = "./data") -> None:
        # pass default arguments to the parent class
        super(CLIPMhaAdapter, self).__init__(backbone, root=root)
        self.adapter = MaskedMultiheadAttentionDownsampled(block_type="mha")
