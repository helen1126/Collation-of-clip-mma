import torch
from torch import nn


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads: int = 4) -> None:
        """
        ��ʼ��MaskedMultiheadAttentionģ�顣

        ����:
            embed_dim (int, ��ѡ): ����Ƕ���ά�ȣ�Ĭ��Ϊ512��
            num_heads (int, ��ѡ): ��ͷע���������е�ͷ����Ĭ��Ϊ4��

        ����:
            1. ���ø���nn.Module�Ĺ��캯����
            2. �洢��ͷע������ͷ����
            3. ��ʼ��һ��PyTorch�Ķ�ͷע�����㡣
            4. ��ʼ��ע�������롣
            5. ��ʼ��һ�����Բ����У������������Բ��һ��GELU�������
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
            torch.Tensor: ��ʼ�����ע�������룬��״Ϊ (num_total, num_total)��
                          ���� num_total = num_prompts + num_images���������ض�λ�õ�ֵΪ1������Ϊ0��
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
        ǰ�򴫲�������

        ����:
            inputs (torch.Tensor): �������������״Ϊ (seq_len, batch_size, embed_dim)��

        ����:
            torch.Tensor: ���������������������״��������ͬ��Ϊ (seq_len, batch_size, embed_dim)��

        ����:
            1. ��ȡ�������������δ�С�����г��ȡ�
            2. ���ע�����������״�����г��Ȳ�ƥ�䣬�����³�ʼ��ע�������롣
            3. ��¡����չע�������룬ʹ����Ӧ���δ�С��ͷ�������ƶ�����ǰ�豸��
            4. ����������ͨ����ͷע��������д���ͬʱӦ��ע�������롣
            5. ����ͷע����������ͨ�����Բ����н��н�һ������
            6. �������յ����������
        """
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        if self._attn_mask.shape[0] != seq_len:
            self._attn_mask = self._init_attn_mask(seq_len - 1, 1)

        mask = self._attn_mask.clone().unsqueeze(0).repeat(batch_size * self._num_heads, 1, 1).to(self.device)

        output, _ = self.mha.forward(inputs, inputs, inputs, attn_mask=mask)
        output = self.linear(output)
        return output