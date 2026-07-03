"""
Transformer Decoder 模块

功能:
1. TransformerDecoderLayer — 单层 Decoder
   (Masked Self-Attention + Cross-Attention + FeedForward)
2. TransformerDecoder — 堆叠 N 层 DecoderLayer

结构(每层):
    x → MaskedMultiHeadSelfAttention(x, x, x, causal_mask + padding_mask) → Add & Norm
      → MultiHeadCrossAttention(x, encoder_output, encoder_output, source_padding_mask) → Add & Norm
      → FeedForward → Add & Norm

说明:
    - Self-Attention 中 Q、K、V 都来自 Decoder 自身输入，但需要用 causal mask
      防止看到未来 token
    - Cross-Attention 中 Q 来自 Decoder，K、V 来自 Encoder 输出
    - padding mask 用于屏蔽 <pad> token
    - FeedForward 与 Encoder 中的结构相同
"""

import torch.nn as nn
from torch import Tensor

from configs.defaults import ModelConfig


class TransformerDecoderLayer(nn.Module):
    """
    单层 Transformer Decoder

    结构:
        x → Masked MultiHeadSelfAttention → Dropout → Add(x) → LayerNorm
          → MultiHeadCrossAttention → Dropout → Add → LayerNorm
          → FeedForward → Dropout → Add → LayerNorm

    Args:
        d_model (int): 模型隐藏维度, 通常 512
        num_heads (int): 多头注意力头数, 通常 8
        d_feedforward (int): 前馈网络隐藏层维度, 通常 2048
        dropout (float): dropout 概率, 通常 0.1
        norm_first (str): LayerNorm 放置策略, "pre" 或 "post"

    Input:
        x (Tensor): Decoder 输入, shape = (batch, target_length, d_model)
        encoder_output (Tensor): Encoder 输出, shape = (batch, source_length, d_model)
        target_padding_mask (Tensor | None): target 序列 padding mask,
            shape = (batch, target_length), True 表示 <pad>
        target_causal_mask (Tensor | None): target 序列 causal mask,
            shape = (target_length, target_length) 或 (batch, ...), True 表示需要屏蔽
        source_padding_mask (Tensor | None): source 序列 padding mask,
            shape = (batch, source_length), True 表示 <pad>

    Returns:
        Tensor: shape = (batch, target_length, d_model)

    使用示例:
        >>> layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_feedforward=2048)
        >>> x = torch.randn(2, 50, 512)  # batch=2, target_length=50
        >>> encoder_out = torch.randn(2, 60, 512)  # batch=2, source_length=60
        >>> tgt_pad_mask = torch.zeros(2, 50, dtype=torch.bool)
        >>> tgt_causal_mask = create_causal_mask(50)  # shape (1, 1, 50, 50)
        >>> src_pad_mask = torch.zeros(2, 60, dtype=torch.bool)
        >>> out = layer(x, encoder_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
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
        # 需要创建:
        #   1. self.self_attention — Masked 多头自注意力
        #      提示: nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
        #              dropout=dropout, batch_first=True)
        #      注意: causal mask 在 forward 中通过 attn_mask 参数传入,
        #            不能写在 __init__ 里, 因为每个 batch 的 target 长度可能不同
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        #   2. self.cross_attention — 交叉注意力(Decoder → Encoder)
        #      提示: nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
        #              dropout=dropout, batch_first=True)
        #      Q 来自 Decoder, K/V 来自 Encoder 输出
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        #   3. self.feed_forward — 前馈网络
        #      结构: Linear(d_model, d_feedforward) → ReLU → Dropout
        #           → Linear(d_feedforward, d_model)
        #      提示: 用 nn.Sequential 组合, 与 Encoder 中完全一致
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )
        #   4. self.norm1, self.norm2, self.norm3 — 三个 LayerNorm
        #      (比 Encoder 多一个, 因为多了 Cross-Attention 子层)
        #      提示: nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        #   5. self.dropout — 公用 Dropout
        #      提示: nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        #   6. self.norm_first — 记录 LayerNorm 放置策略
        self.norm_first = norm_first

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        target_padding_mask: Tensor | None = None,
        target_causal_mask: Tensor | None = None,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: Decoder 输入, shape = (batch, target_length, d_model)
            encoder_output: Encoder 输出, shape = (batch, source_length, d_model)
            target_padding_mask: target 序列的 padding mask,
                shape = (batch, target_length), True = <pad> 位置
            target_causal_mask: target 序列的 causal mask,
                通常是 create_causal_mask(target_length) 的结果,
                shape = (target_length, target_length) 或已扩展为 (batch, ...)
            source_padding_mask: source 序列的 padding mask,
                shape = (batch, source_length), True = <pad> 位置

        Returns:
            Tensor: shape = (batch, target_length, d_model)
        """
        # =========================================================================
        # 整体结构(三层子层, 比 Encoder 多一个 Cross-Attention):
        if self.norm_first == "post":
            # [Post-LN 风格]:
            #   1. Self-Attention(带 causal mask):
            #      attention_output, _ = self.self_attention(
            #          query=x, key=x, value=x,
            #          attn_mask=target_causal_mask,      ← 防止看到未来 token
            #          key_padding_mask=target_padding_mask ← 忽略 <pad>
            #      )
            #      x = self.norm1(x + self.dropout(attention_output))
            attention_output, _ = self.multi_head_attention(
                query=x,
                key=x,
                value=x,
                attn_mask=target_causal_mask,  # 防止看到未来token
                key_padding_mask=target_padding_mask,  # 忽略 <pad>
            )
            #   2. Cross-Attention(Q=Decoder, K/V=Encoder):
            #      cross_output, _ = self.cross_attention(
            #          query=x,                            ← Decoder 当前状态
            #          key=encoder_output,                  ← Encoder 输出
            #          value=encoder_output,                ← Encoder 输出
            #          key_padding_mask=source_padding_mask  ← 忽略 source <pad>
            #      )
            #      x = self.norm2(x + self.dropout(cross_output))
            cross_output, _ = self.cross_attention(
                query=x,
                key=encoder_output,  # Encoder输出
                value=encoder_output,  # Encoder输出
                key_padding_mask=source_padding_mask,
            )
            x = self.norm2(x + self.dropout(cross_output))
            #   3. FeedForward:
            #      ff_output = self.feed_forward(x)
            #      x = self.norm3(x + self.dropout(ff_output))
            feedforward_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout(feedforward_output))
            #   4. return x
            return x
        #
        # [Pre-LN 风格(推荐, 更稳定)]:
        #   1. Self-Attention:
        #      normed = self.norm1(x)
        #      attention_output, _ = self.self_attention(
        #          query=normed, key=normed, value=normed,
        #          attn_mask=target_causal_mask,
        #          key_padding_mask=target_padding_mask
        #      )
        #      x = x + self.dropout(attention_output)
        #
        #   2. Cross-Attention:
        #      normed = self.norm2(x)
        #      cross_output, _ = self.cross_attention(
        #          query=normed,
        #          key=encoder_output,
        #          value=encoder_output,
        #          key_padding_mask=source_padding_mask
        #      )
        #      x = x + self.dropout(cross_output)
        #
        #   3. FeedForward:
        #      x = x + self.dropout(self.feed_forward(self.norm3(x)))
        #
        #   4. return x
        if self.norm_first == "pre":
            x_normed = self.norm1(x)
            attention_output, _ = self.multi_head_attention(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                attn_mask=target_causal_mask,
                key_padding_mask=target_padding_mask,
            )
            x = x + self.dropout(attention_output)

            x_normed = self.norm2(x)
            cross_output, _ = self.cross_attention(
                query=x_normed,
                key=encoder_output,
                value=encoder_output,
                key_padding_mask=source_padding_mask,
            )
            x = x + self.dropout(cross_output)

            x = x + self.dropout(self.feed_forward(self.norm3(x)))
            return x
        # 关键区别:
        #   - Self-Attention 多了 attn_mask=target_causal_mask, 防止 decoder 看到未来 token
        #   - Cross-Attention 的 K/V 来自 encoder_output, 不是 x
        #   - 比 Encoder 多一层 LayerNorm(norm3), 因为多了一个子层


class TransformerDecoder(nn.Module):
    """
    堆叠 N 层的 Transformer Decoder

    Args:
        d_model (int): 模型隐藏维度
        num_heads (int): 注意力头数
        d_feedforward (int): 前馈网络隐藏层维度
        num_layers (int): Decoder 层数, 通常 6
        dropout (float): dropout 概率
        norm_first (str): LayerNorm 放置策略, "pre" 或 "post"

    输入:
        x (Tensor): Decoder 输入, shape = (batch, target_length, d_model)
        encoder_output (Tensor): Encoder 输出, shape = (batch, source_length, d_model)
        target_padding_mask (Tensor | None): shape = (batch, target_length)
        target_causal_mask (Tensor | None): causal mask, 通常由外部生成后传入
        source_padding_mask (Tensor | None): shape = (batch, source_length)

    输出:
        Tensor: shape = (batch, target_length, d_model)

    使用示例:
        >>> decoder = TransformerDecoder(d_model=512, num_heads=8,
        ...                              d_feedforward=2048, num_layers=6)
        >>> x = torch.randn(2, 50, 512)
        >>> encoder_out = torch.randn(2, 60, 512)
        >>> tgt_pad_mask = torch.zeros(2, 50, dtype=torch.bool)
        >>> tgt_causal_mask = create_causal_mask(50)
        >>> src_pad_mask = torch.zeros(2, 60, dtype=torch.bool)
        >>> out = decoder(x, encoder_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
        >>> out.shape
        torch.Size([2, 50, 512])
    """

    def __init__(
        self,
        d_model: int = ModelConfig.d_model,
        num_heads: int = ModelConfig.num_heads,
        d_feedforward: int = ModelConfig.d_feedforward,
        num_layers: int = ModelConfig.decoder_num_layers,
        dropout: float = ModelConfig.dropout,
        norm_first: str = ModelConfig.norm_first,
    ) -> None:
        super().__init__()
        # 步骤:
        #   1. 用 nn.ModuleList 堆叠 num_layers 个 TransformerDecoderLayer
        #      注意: 必须用 ModuleList 而非 Sequential,
        #            因为每层的 forward 需要传入多个 mask 参数,
        #            Sequential 只能传递单个 x
        #      提示: nn.ModuleList([
        #               TransformerDecoderLayer(
        #                   d_model=d_model,
        #                   num_heads=num_heads,
        #                   d_feedforward=d_feedforward,
        #                   dropout=dropout,
        #                   norm_first=norm_first,
        #               )
        #               for _ in range(num_layers)
        #            ])
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                    norm_first=norm_first,
                )
            ]
        )
        #   2. 如果使用 Pre-LN, 在所有层之后加一个最终的 LayerNorm:
        #      self.final_norm = nn.LayerNorm(d_model) if norm_first == "pre" else None
        #      这是 Pre-LN 的标准做法, 保证最终输出的数值稳定性
        self.final_norm = nn.LayerNorm(d_model) if norm_first == "pre" else None

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        target_padding_mask: Tensor | None = None,
        target_causal_mask: Tensor | None = None,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: Decoder 输入, shape = (batch, target_length, d_model)
            encoder_output: Encoder 输出, shape = (batch, source_length, d_model)
            target_padding_mask: shape = (batch, target_length)
            target_causal_mask: causal mask, shape 需与 MultiheadAttention
                的 attn_mask 要求兼容
            source_padding_mask: shape = (batch, source_length)

        Returns:
            Tensor: shape = (batch, target_length, d_model)
        """
        # 步骤:
        #   1. 逐层调用:
        #      for layer in self.layers:
        #          x = layer(
        #              x=x,
        #              encoder_output=encoder_output,
        #              target_padding_mask=target_padding_mask,
        #              target_causal_mask=target_causal_mask,
        #              source_padding_mask=source_padding_mask,
        #          )
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                target_padding_mask=target_padding_mask,
                target_causal_mask=target_causal_mask,
                source_padding_mask=source_padding_mask,
            )
        if self.final_norm:
            x = self.final_norm(x)

        #   2. 如果 Pre-LN, 过最终 LayerNorm:
        #      if self.final_norm is not None:
        #          x = self.final_norm(x)
        #   3. 返回最终的 x
        return x
