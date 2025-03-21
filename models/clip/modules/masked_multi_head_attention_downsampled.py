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
        ��ʼ��MaskedMultiheadAttentionDownsampledģ�顣

        ����:
            embed_dim (int, ��ѡ): ����Ƕ���ά�ȣ�Ĭ��Ϊ512��
            downsamling_dim (int, ��ѡ): �²������ά�ȣ�Ĭ��Ϊ128��
            num_heads (int, ��ѡ): ��ͷע���������е�ͷ����Ĭ��Ϊ4��
            downsampling_type (str, ��ѡ): �²��������ͣ���ѡֵΪ "linear" �� "mlp"��Ĭ��Ϊ "linear"��
            upsampling_type (str, ��ѡ): �ϲ��������ͣ���ѡֵΪ "linear" �� "mlp"��Ĭ��Ϊ "linear"��
            block_type (str, ��ѡ): ������ͣ���ѡֵΪ "mha" �� "transformer"��Ĭ��Ϊ "mha"��

        ����:
            1. ���ø���nn.Module�Ĺ��캯����
            2. ��ʼ��ģ��ĸ��ֲ��������²������͡��ϲ������͡������͵ȡ�
            3. ���ݲ�ͬ���²������ͺ��ϲ������ͳ�ʼ���²��������ϲ�������
            4. ���ݿ����ͳ�ʼ����ͷע����ģ���Transformer�������㡣
            5. ��ʼ��ע�������롣
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
        ��ȡ��ǰģ�����ڵ��豸��

        ����:
            torch.device: ��ǰģ�����ڵ��豸��
        """
        return next(self.parameters()).device

    @staticmethod
    def _init_attn_mask(num_prompts: int, num_images: int) -> torch.Tensor:
        """
        ��ʼ��ע�������롣

        ����:
            num_prompts (int): ��ʾ��������
            num_images (int): ͼ���������

        ����:
            torch.Tensor: ��ʼ�����ע�������롣
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
        ǰ�򴫲�������

        ����:
            inputs (torch.Tensor): �����������

        ����:
            torch.Tensor: �������������������

        ����:
            1. ��ȡ�������������δ�С�����г��ȡ�
            2. ���ע�����������״�����г��Ȳ�ƥ�䣬�����³�ʼ��ע�������롣
            3. ��¡����չע�������룬ʹ����Ӧ���δ�С��ͷ����
            4. ����������²�����
            5. ���ݿ����ͣ����²����������ͨ����ͷע����ģ���Transformer����������д���
            6. �Դ�������������ϲ�����
            7. �����ϲ�����������
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