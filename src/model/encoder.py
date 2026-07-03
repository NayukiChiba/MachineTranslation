"""
Transformer Encoder 模块

功能:
1. TransformerEncoderLayer — 单层 Encoder(Self-Attention + FeedForward)
2. TransformerEncoder — 堆叠 N 层 EncoderLayer

结构(每层):
    x → MultiHeadSelfAttention(x, x, x, padding_mask) → Add & Norm
      → FeedForward → Add & Norm

说明:
    - Self-Attention 中 Q、K、V 都来自同一个输入 x
    - padding mask 用于屏蔽 <pad> token
    - FeedForward 是两层线性变换 + ReLU/GeLU
"""

import torch.nn as nn
from torch import Tensor

from configs.defaults import ModelConfig


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer Encoder

    结构:
        x → MultiHeadSelfAttention → Dropout → Add(x) → LayerNorm
          → FeedForward → Dropout → Add → LayerNorm

    Args:
        d_model (int): 模型隐藏维度,通常 512
        num_heads (int): 多头注意力头数,通常 8
        d_feedforward (int): 前馈网络隐藏层维度,通常 2048
        dropout (float): dropout 概率,通常 0.1

    输入:
        x (Tensor): shape = (batch, source_length, d_model)
        source_padding_mask (Tensor | None): shape = (batch, source_length),True 表示 <pad>

    输出:
        Tensor: shape = (batch, source_length, d_model)

    使用示例:
        >>> layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_feedforward=2048)
        >>> x = torch.randn(2, 50, 512)  # batch=2, source_length=50
        >>> mask = torch.zeros(2, 50, dtype=torch.bool)  # 无 pad
        >>> out = layer(x, source_padding_mask=mask)
        >>> out.shape
        torch.Size([2, 50, 512])
    """

    def __init__(
        self,
        d_model: int = ModelConfig.d_model,
        num_heads: int = ModelConfig.num_heads,
        d_feedforward: int = ModelConfig.d_feedforward,
        dropout: float = ModelConfig.dropout,
        norm_first: str = ModelConfig.norm_first,
    ) -> None:
        super().__init__()
        # 初始化各个子层
        # 需要创建:
        #   1. self.multi_head_attention — 多头自注意力
        #      提示: nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
        #              dropout=dropout, batch_first=True)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        #   2. self.feedforward — 前馈网络
        #      结构: Linear(d_model, d_feedforward) → ReLU → Dropout → Linear(d_feedforward, d_model)
        #      提示: 用 nn.Sequential 组合
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )
        #   3. self.norm1, self.norm2 — 两个 LayerNorm
        #      提示: nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #   4. self.dropout — 公用 Dropout
        #      提示: nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: Tensor, source_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: shape = (batch, source_length, d_model)
            source_padding_mask: shape = (batch, source_length),True = <pad> 位置

        Returns:
            Tensor: shape = (batch, source_length, d_model)
        """
        # Post-LN 是原始论文中的
        # temp = LN(input + Attention(input))
        # y = LN(temp + FFN(temp))
        if self.norm_first == "post":
            attention_output, _ = self.multi_head_attention(
                query=x, key=x, value=x, key_padding_mask=source_padding_mask
            )
            x = self.norm1(x + self.dropout(attention_output))
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
            return x

        # Pre-LN 更加稳定
        # temp = input + Attention(LN(input))
        # output = temp + FFN(LN(temp))
        if self.norm_first == "pre":
            x_normed = self.norm1(x)
            attention_output, _ = self.multi_head_attention(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=source_padding_mask,
            )
            x = x + self.dropout(attention_output)

            # feed forward
            x = x + self.dropout(self.feed_forward(self.norm2(x)))

            return x


class TransformerEncoder(nn.Module):
    """
    堆叠 N 层的 Transformer Encoder

    Args:
        d_model (int): 模型隐藏维度
        num_heads (int): 注意力头数
        d_feedforward (int): 前馈网络隐藏层维度
        num_layers (int): Encoder 层数,通常 6
        dropout (float): dropout 概率

    输入:
        x (Tensor): shape = (batch, source_length, d_model)
        source_padding_mask (Tensor | None): shape = (batch, source_length)

    输出:
        Tensor: shape = (batch, source_length, d_model)

    使用示例:
        >>> encoder = TransformerEncoder(d_model=512, num_heads=8, d_feedforward=2048, num_layers=6)
        >>> x = torch.randn(2, 50, 512)
        >>> mask = torch.zeros(2, 50, dtype=torch.bool)
        >>> out = encoder(x, source_padding_mask=mask)
        >>> out.shape
        torch.Size([2, 50, 512])
    """

    def __init__(
        self,
        d_model: int = ModelConfig.d_model,
        num_heads: int = ModelConfig.num_heads,
        d_feedforward: int = ModelConfig.d_feedforward,
        num_layers: int = ModelConfig.encoder_num_layers,
        dropout: float = ModelConfig.dropout,
        norm_first: str = ModelConfig.norm_first,
    ) -> None:
        super().__init__()
        # 步骤:
        #   1. 用 nn.ModuleList 堆叠 num_layers 个 TransformerEncoderLayer
        #      提示: nn.ModuleList([TransformerEncoderLayer(...) for _ in range(num_layers)])
        #   2. 可选: 加一个最终的 LayerNorm(如果使用 Pre-LN,通常会在最后加一层 LayerNorm)
        self.layers = nn.ModuleList(
            modules=[
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )  # 替换为 nn.ModuleList(...)
        self.final_norm = nn.LayerNorm(d_model) if norm_first == "pre" else None

    def forward(self, x: Tensor, source_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: shape = (batch, source_length, d_model)
            source_padding_mask: shape = (batch, source_length)

        Returns:
            Tensor: shape = (batch, source_length, d_model)
        """
        # TODO: 实现 forward
        # 步骤:
        #   1. 逐层调用: for layer in self.layers: x = layer(x, source_padding_mask)
        #   2. 返回最终的 x
        raise NotImplementedError("TODO: 实现 TransformerEncoder.forward")
