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

# =============================================================================
# TODO: 如果你想把 MultiHeadAttention 抽出来复用(encoder/decoder 都用到),
#       建议在 model/ 下新建一个 attention.py.
#       这里先用占位注释标出需要的接口,你在 encoder.py 中可以直接
#       使用 nn.MultiheadAttention(PyTorch 内置)或手写一个.
#
#   PyTorch 内置用法提示:
#       self.self_attention = nn.MultiheadAttention(
#           embed_dim=d_model, num_heads=num_heads,
#           dropout=dropout, batch_first=True
#       )
#       调用: attention_output, attention_weights = self.self_attention(x, x, x, key_padding_mask=...)
#
#   手写提示(如果你想从零实现):
#       - Q = x @ W_q, K = x @ W_k, V = x @ W_v
#       - scores = Q @ K^T / sqrt(d_k)
#       - scores.masked_fill(mask, -1e9)
#       - attention = softmax(scores) @ V
#       - 最后过一个线性层 W_o
# =============================================================================


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
        self, d_model: int, num_heads: int, d_feedforward: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        # 初始化各个子层
        # 需要创建:
        #   1. self.self_attention — 多头自注意力
        #      提示: nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
        #              dropout=dropout, batch_first=True)
        self.nulti_head_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        #   2. self.feedforward — 前馈网络
        #      结构: Linear(d_model, d_feedforward) → ReLU → Dropout → Linear(d_feedforward, d_model)
        #      提示: 用 nn.Sequential 组合
        self.feed_forward = nn.Sequential(
            [
                nn.Linear(d_model, d_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_feedforward, d_model),
            ]
        )
        #   3. self.norm1, self.norm2 — 两个 LayerNorm
        #      提示: nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #   4. self.dropout — 公用 Dropout
        #      提示: nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, source_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: shape = (batch, source_length, d_model)
            source_padding_mask: shape = (batch, source_length),True = <pad> 位置

        Returns:
            Tensor: shape = (batch, source_length, d_model)
        """
        # TODO: 实现 EncoderLayer 的 forward
        # 步骤(严格按照 Pre-LN 或 Post-LN 选一种,保持一致):
        #
        # [Post-LN 风格(Transformer 原论文)]:
        #   1. attention_output, _ = self.self_attention(x, x, x, key_padding_mask=source_padding_mask)
        #   2. x = self.norm1(x + self.dropout(attention_output))
        #   3. feedforward_output = self.feedforward(x)
        #   4. x = self.norm2(x + self.dropout(feedforward_output))
        #   5. return x
        #
        # [Pre-LN 风格(更稳定,推荐)]:
        #   1. residual = x
        #   2. x = self.norm1(x)
        #   3. attention_output, _ = self.self_attention(x, x, x, key_padding_mask=source_padding_mask)
        #   4. x = residual + self.dropout(attention_output)
        #   5. residual = x
        #   6. x = self.norm2(x)
        #   7. feedforward_output = self.feedforward(x)
        #   8. x = residual + self.dropout(feedforward_output)
        #   9. return x
        raise NotImplementedError("TODO: 实现 TransformerEncoderLayer.forward")


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
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # TODO: 初始化
        # 步骤:
        #   1. 用 nn.ModuleList 堆叠 num_layers 个 TransformerEncoderLayer
        #      提示: nn.ModuleList([TransformerEncoderLayer(...) for _ in range(num_layers)])
        #   2. 可选: 加一个最终的 LayerNorm(如果使用 Pre-LN,通常会在最后加一层 LayerNorm)
        self.layers = None  # 替换为 nn.ModuleList(...)

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
