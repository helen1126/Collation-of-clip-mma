import torch
from torch import nn


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads: int = 4) -> None:
        """
        初始化MaskedMultiheadAttention模块。

        参数:
            embed_dim (int, 可选): 输入嵌入的维度，默认为512。
            num_heads (int, 可选): 多头注意力机制中的头数，默认为4。

        功能:
            1. 调用父类nn.Module的构造函数。
            2. 存储多头注意力的头数。
            3. 初始化一个PyTorch的多头注意力层。
            4. 初始化注意力掩码。
            5. 初始化一个线性层序列，包含两个线性层和一个GELU激活函数。
        """
        super(MaskedMultiheadAttention, self).__init__()

        self._num_heads = num_heads

        self.mha = nn.MultiheadAttention(embed_dim, num_heads=self._num_heads)
        self._attn_mask: torch.Tensor = self._init_attn_mask(1, 1)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
        )

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
            torch.Tensor: 初始化后的注意力掩码，形状为 (num_total, num_total)，
                          其中 num_total = num_prompts + num_images。掩码中特定位置的值为1，其余为0。
        """
        num_total = num_prompts + num_images
        mask = torch.zeros((num_total, num_total))

        for i in range(num_prompts):
            for j in range(num_prompts, num_total):
                mask[i, j] = 1

        for i in range(num_prompts, num_total):
            for j in range(num_prompts):
                mask[i, j] = 1

        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            inputs (torch.Tensor): 输入的张量，形状为 (seq_len, batch_size, embed_dim)。

        返回:
            torch.Tensor: 经过处理后的输出张量，形状与输入相同，为 (seq_len, batch_size, embed_dim)。

        功能:
            1. 获取输入张量的批次大小和序列长度。
            2. 如果注意力掩码的形状与序列长度不匹配，则重新初始化注意力掩码。
            3. 克隆并扩展注意力掩码，使其适应批次大小和头数，并移动到当前设备。
            4. 将输入张量通过多头注意力层进行处理，同时应用注意力掩码。
            5. 将多头注意力层的输出通过线性层序列进行进一步处理。
            6. 返回最终的输出张量。
        """
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        if self._attn_mask.shape[0] != seq_len:
            self._attn_mask = self._init_attn_mask(seq_len - 1, 1)

        mask = self._attn_mask.clone().unsqueeze(0).repeat(batch_size * self._num_heads, 1, 1).to(self.device)

        output, _ = self.mha.forward(inputs, inputs, inputs, attn_mask=mask)
        output = self.linear(output)
        return output