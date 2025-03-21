import torch
from torch import nn


class MaskedMultiheadAttentionDownsampled(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        downsamling_dim=128,
        num_heads: int = 4,
        downsampling_type="linear",
        upsampling_type="linear",
        block_type="mha",
    ) -> None:
        """
        初始化MaskedMultiheadAttentionDownsampled模块。

        参数:
            embed_dim (int, 可选): 输入嵌入的维度，默认为512。
            downsamling_dim (int, 可选): 下采样后的维度，默认为128。
            num_heads (int, 可选): 多头注意力机制中的头数，默认为4。
            downsampling_type (str, 可选): 下采样的类型，可选值为 "linear" 或 "mlp"，默认为 "linear"。
            upsampling_type (str, 可选): 上采样的类型，可选值为 "linear" 或 "mlp"，默认为 "linear"。
            block_type (str, 可选): 块的类型，可选值为 "mha" 或 "transformer"，默认为 "mha"。

        功能:
            1. 调用父类nn.Module的构造函数。
            2. 初始化模块的各种参数，如下采样类型、上采样类型、块类型等。
            3. 根据不同的下采样类型和上采样类型初始化下采样器和上采样器。
            4. 根据块类型初始化多头注意力模块或Transformer编码器层。
            5. 初始化注意力掩码。
        """
        super(MaskedMultiheadAttentionDownsampled, self).__init__()

        self._num_heads = num_heads
        self.downsampling_type = downsampling_type
        self.upsampling_type = upsampling_type
        self.block_type = block_type

        hidden_dim = downsamling_dim

        if downsampling_type == "linear":
            self.downsampler = nn.Linear(embed_dim, downsamling_dim)
        elif downsampling_type == "mlp":
            self.downsampler = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, downsamling_dim),
            )

        if upsampling_type == "linear":
            self.upsampler = nn.Linear(downsamling_dim, embed_dim)
        elif upsampling_type == "mlp":
            self.upsampler = nn.Sequential(
                nn.Linear(downsamling_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        if block_type == "mha":
            self.mha = nn.MultiheadAttention(downsamling_dim, num_heads=self._num_heads)
        elif block_type == "transformer":
            self.transformer = nn.TransformerEncoderLayer(downsamling_dim, nhead=self._num_heads)

        self._attn_mask: torch.Tensor = self._init_attn_mask(1, 1)

    @property
    def device(self) -> torch.device:
        """
        获取当前模块所在的设备。

        返回:
            torch.device: 当前模块所在的设备。
        """
        return next(self.parameters()).device

    @staticmethod
    def _init_attn_mask(num_prompts: int, num_images: int) -> torch.Tensor:
        """
        初始化注意力掩码。

        参数:
            num_prompts (int): 提示的数量。
            num_images (int): 图像的数量。

        返回:
            torch.Tensor: 初始化后的注意力掩码。
        """
        num_total = num_prompts + num_images
        mask = torch.zeros((num_total, num_total))
        mask = mask - float("inf")

        for i in range(num_prompts):
            for j in range(num_prompts, num_total):
                mask[i, j] = 0

        for i in range(num_prompts, num_total):
            for j in range(num_prompts):
                mask[i, j] = 0

        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            inputs (torch.Tensor): 输入的张量。

        返回:
            torch.Tensor: 经过处理后的输出张量。

        功能:
            1. 获取输入张量的批次大小和序列长度。
            2. 如果注意力掩码的形状与序列长度不匹配，则重新初始化注意力掩码。
            3. 克隆并扩展注意力掩码，使其适应批次大小和头数。
            4. 对输入进行下采样。
            5. 根据块类型，将下采样后的输入通过多头注意力模块或Transformer编码器层进行处理。
            6. 对处理后的输出进行上采样。
            7. 返回上采样后的输出。
        """
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        if self._attn_mask.shape[0] != seq_len:
            self._attn_mask = self._init_attn_mask(seq_len - 1, 1)

        mask = self._attn_mask.clone().unsqueeze(0).repeat(batch_size * self._num_heads, 1, 1).to(self.device)

        output = self.downsampler(inputs)
        if self.block_type == "mha":
            output, _ = self.mha.forward(output, output, output, attn_mask=mask)
        elif self.block_type == "transformer":
            output = self.transformer.forward(output, src_mask=mask)
        output = self.upsampler(output)
        return output