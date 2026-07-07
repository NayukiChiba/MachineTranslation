"""
完整 Encoder-Decoder Transformer 模型

功能:
1. 将 TokenEmbedding + PositionalEncoding + Encoder + Decoder + 输出投影
   组装为完整的 Transformer 翻译模型
2. 提供 encode() / decode() / forward() 三组接口,
   分别用于仅编码、仅解码(推理时自回归)和训练时一次前向

结构:
    source_ids → SourceEmbedding → PositionalEncoding → Encoder → encoder_output
    target_ids → TargetEmbedding → PositionalEncoding → Decoder(+ encoder_output) → Linear → logits

说明:
    - 源语言和目标语言共享同一个 SentencePiece 词表, 因此 SourceEmbedding 与
      TargetEmbedding 可以共用同一个 TokenEmbedding 实例(权重共享)
    - 最终 Linear 投影层的权重可选择与 TargetEmbedding 绑定(weight tying),
      减少参数量并提升泛化能力
    - 所有 mask 在 forward 内部通过调用 src/model/mask.py 中的函数生成,
      外部只需传入 source_ids 和 target_ids
"""

import torch.nn as nn
from decoder import TransformerDecoder
from embedding import PositionalEncoding, TokenEmbedding
from encoder import TransformerEncoder
from mask import create_padding_mask
from torch import Tensor

from configs.defaults import ModelConfig, TokenizerConfig


class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer 翻译模型

    Args:
        vocab_size (int): 共享词表大小, 默认取 TokenizerConfig.vocab_size
        d_model (int): 模型隐藏维度, 默认 512
        num_heads (int): 注意力头数, 默认 8
        d_feedforward (int): 前馈网络隐藏层维度, 默认 2048
        encoder_num_layers (int): Encoder 层数, 默认 6
        decoder_num_layers (int): Decoder 层数, 默认 6
        dropout (float): dropout 概率, 默认 0.1
        max_sequence_length (int): 位置编码最大长度, 默认 5000
        pad_id (int): <pad> token id, 默认 0
        norm_first (str): LayerNorm 放置策略, 默认 "pre"

    输入(forward):
        source_ids (Tensor): 源语言 token id, shape = (batch, source_length)
        target_ids (Tensor): 目标语言 token id, shape = (batch, target_length)

    输出(forward):
        Tensor: logits, shape = (batch, target_length, vocab_size),
                未经过 softmax, 直接接 CrossEntropyLoss

    使用示例:
        >>> model = Transformer(vocab_size=16000, d_model=512, num_heads=8,
        ...                     d_feedforward=2048, encoder_num_layers=6,
        ...                     decoder_num_layers=6)
        >>> src = torch.randint(0, 16000, (2, 50))  # batch=2, source_length=50
        >>> tgt = torch.randint(0, 16000, (2, 60))  # batch=2, target_length=60
        >>> logits = model(src, tgt)
        >>> logits.shape
        torch.Size([2, 60, 16000])
    """

    def __init__(
        self,
        vocab_size: int = TokenizerConfig.vocab_size,
        d_model: int = ModelConfig.d_model,
        num_heads: int = ModelConfig.num_heads,
        d_feedforward: int = ModelConfig.d_feedforward,
        encoder_num_layers: int = ModelConfig.encoder_num_layers,
        decoder_num_layers: int = ModelConfig.decoder_num_layers,
        dropout: float = ModelConfig.dropout,
        max_seq_length: int = ModelConfig.max_seq_length,
        pad_id: int = TokenizerConfig.pad_id,
        norm_first: str = ModelConfig.norm_first,
    ) -> None:
        super().__init__()
        # 1. TokenEmbedding — 源语言和目标语言共享
        #    因为使用 SentencePiece 联合训练的共享词表, 源和目标共用同一个
        #    embedding 矩阵可以:
        #      - 减少参数量(省掉一套 vocab_size × d_model 的参数)
        #      - 让两种语言的子词在同一个语义空间中表示
        #    创建方式:
        #      self.source_embedding = TokenEmbedding(
        #          vocab_size=vocab_size, hidden_dim=d_model, pad_id=pad_id
        #      )
        #      self.target_embedding = self.source_embedding  # 共享权重
        self.source_embedding = TokenEmbedding(
            vocab_size=vocab_size, d_model=d_model, pad_id=pad_id
        )
        # 2. PositionalEncoding — 源和目标共用
        #    位置编码不涉及可训练参数, 一份实例即可
        #    创建方式:
        #      self.positional_encoding = PositionalEncoding(
        #          hidden_dim=d_model, dropout=dropout,
        #          max_length=max_sequence_length
        #      )
        self.position_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_seq_length=max_seq_length
        )
        # 3. TransformerEncoder
        #    创建方式:
        #      self.encoder = TransformerEncoder(
        #          d_model=d_model, num_heads=num_heads,
        #          d_feedforward=d_feedforward, num_layers=encoder_num_layers,
        #          dropout=dropout, norm_first=norm_first
        #      )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            num_layers=encoder_num_layers,
            dropout=dropout,
            norm_first=norm_first,
        )
        # 4. TransformerDecoder
        #    创建方式:
        #      self.decoder = TransformerDecoder(
        #          d_model=d_model, num_heads=num_heads,
        #          d_feedforward=d_feedforward, num_layers=decoder_num_layers,
        #          dropout=dropout, norm_first=norm_first
        #      )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            num_layers=decoder_num_layers,
            dropout=dropout,
            norm_first=norm_first,
        )
        # 5. 输出投影层 — 将 d_model 映射回 vocab_size, 得到每个位置的词表概率分布
        #    创建方式:
        #      self.projection = nn.Linear(d_model, vocab_size)
        #
        #    可选优化(weight tying):
        #      将 projection 的权重与 target_embedding 的权重绑定,
        #      减少参数量: vocab_size × d_model
        #      做法: self.projection.weight = self.target_embedding.embedding.weight
        #      注意: 如果 vocab_size 较大(>50k), weight tying 能显著降低过拟合风险
        self.projection = nn.Linear(d_model, vocab_size)

        self.pad_id = pad_id
        self.d_model = d_model
        # 6. 保存常用配置
        #    self.pad_id = pad_id
        #    self.d_model = d_model

    def encode(
        self, source_ids: Tensor, source_padding_mask: Tensor | None = None
    ) -> Tensor:
        """
        仅执行 Encoder 部分, 用于:
          - 推理时预先计算 encoder_output, 避免每步解码都重跑 Encoder
          - 获取源语言的上下文表示用于分析

        Args:
            source_ids (Tensor): 源语言 token id, shape = (batch, source_length)
            source_padding_mask (Tensor | None): 如果已提前生成可传入,
                否则内部根据 pad_id 自动生成

        Returns:
            Tensor: encoder_output, shape = (batch, source_length, d_model)
        """
        # 步骤:
        #   1. 如果 source_padding_mask 为 None, 根据 source_ids 和 pad_id 生成:
        #      source_padding_mask = create_padding_mask(source_ids, pad_id)
        #      注意: 这里生成的 mask shape 是 (batch, 1, 1, source_length),
        #            但 Encoder 的 MultiheadAttention 期望的 key_padding_mask
        #            是 (batch, source_length), 需要 squeeze 掉中间两个维度
        #            (或者在 mask.py 中新增一个返回 (batch, seq_len) 版本的函数)
        if not source_padding_mask:
            source_padding_mask = create_padding_mask(x=source_ids, pad_id=self.pad_id)
            source_padding_mask = source_padding_mask.squeeze(1).squeeze(1)
        #   2. x = self.source_embedding(source_ids)
        #      → shape (batch, source_length, d_model)
        x = self.source_embedding(source_ids)
        #   3. x = self.positional_encoding(x)
        #      → shape (batch, source_length, d_model)
        x = self.position_encoding(x)
        #   4. encoder_output = self.encoder(x, source_padding_mask=source_padding_mask)
        #      → shape (batch, source_length, d_model)
        encoder_output = self.encoder(x, source_padding_mask=source_padding_mask)
        #   5. return encoder_output
        return encoder_output

    def decode(
        self,
        target_ids: Tensor,
        encoder_output: Tensor,
        target_padding_mask: Tensor | None = None,
        target_causal_mask: Tensor | None = None,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        仅执行 Decoder 部分, 用于推理时的自回归解码

        Args:
            target_ids (Tensor): 目标语言 token id, shape = (batch, target_length)
                推理时每次传入逐步增长的目标序列
            encoder_output (Tensor): Encoder 输出, shape = (batch, source_length, d_model)
                通常已提前通过 encode() 计算好
            target_padding_mask (Tensor | None): target 序列的 padding mask
            target_causal_mask (Tensor | None): causal mask, 推理时每次 target_length
                变化都需要重新生成
            source_padding_mask (Tensor | None): source 序列的 padding mask

        Returns:
            Tensor: logits, shape = (batch, target_length, vocab_size)
        """
        # =========================================================================
        # TODO: 实现 decode
        # 步骤:
        #   1. x = self.target_embedding(target_ids)
        #      → shape (batch, target_length, d_model)
        #
        #   2. x = self.positional_encoding(x)
        #      → shape (batch, target_length, d_model)
        #
        #   3. x = self.decoder(
        #          x=x,
        #          encoder_output=encoder_output,
        #          target_padding_mask=target_padding_mask,
        #          target_causal_mask=target_causal_mask,
        #          source_padding_mask=source_padding_mask,
        #      )
        #      → shape (batch, target_length, d_model)
        #
        #   4. logits = self.projection(x)
        #      → shape (batch, target_length, vocab_size)
        #
        #   5. return logits
        #
        # 注意:
        #   推理时 decode() 不应用 softmax, 保持 logits 输出,
        #   让外部(translator.py)自行决定使用 greedy / beam search / temperature 等策略
        raise NotImplementedError("TODO: 实现 Transformer.decode")

    # =========================================================================
    # 主前向传播

    def forward(self, source_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        训练时的完整前向传播: Encoder → Decoder → logits

        训练时调用此方法, 一次获得所有位置的 logits,
        与 target_ids 的偏移版本计算 CrossEntropyLoss

        Args:
            source_ids (Tensor): 源语言 token id, shape = (batch, source_length)
            target_ids (Tensor): 目标语言 token id, shape = (batch, target_length)

        Returns:
            Tensor: logits, shape = (batch, target_length, vocab_size)

        训练时的 loss 计算(在外部 trainer 中进行):
            logits = model(source_ids, target_ids)
            # logits[:, :-1, :] 预测 target_ids[:, 1:]
            # 即: 用 target 的前 n-1 个 token 预测后 n-1 个 token
            loss = CrossEntropyLoss(logits[:, :-1, :].transpose(1, 2),
                                     target_ids[:, 1:], ignore_index=pad_id)
        """
        # =========================================================================
        # TODO: 实现 forward (训练时完整流程)
        # 步骤:
        #
        # === Encoder 阶段 ===
        #   1. 生成 source_padding_mask:
        #      source_padding_mask = create_padding_mask(source_ids, self.pad_id)
        #      → shape (batch, 1, 1, source_length)
        #      注意: 传给 Encoder 的 key_padding_mask 需要 (batch, source_length) 形状
        #            所以可能需要 squeeze(1).squeeze(1) 或将 mask 转为合适的形状
        #
        #   2. source_embed = self.source_embedding(source_ids)
        #      → shape (batch, source_length, d_model)
        #
        #   3. source_embed = self.positional_encoding(source_embed)
        #      → shape (batch, source_length, d_model)
        #
        #   4. encoder_output = self.encoder(
        #          x=source_embed,
        #          source_padding_mask=source_padding_mask  # 需要调整为 (batch, source_length)
        #      )
        #      → shape (batch, source_length, d_model)
        #
        # === Decoder 阶段 ===
        #   5. 生成 target 的 mask:
        #      target_padding_mask = create_padding_mask(target_ids, self.pad_id)
        #      target_causal_mask = create_causal_mask(target_ids.size(1))
        #      → padding: (batch, 1, 1, target_length)
        #      → causal: (1, 1, target_length, target_length)
        #
        #      同样, key_padding_mask 传给 MultiheadAttention 时需要
        #      (batch, target_length) 形状
        #
        #   6. target_embed = self.target_embedding(target_ids)
        #      → shape (batch, target_length, d_model)
        #
        #   7. target_embed = self.positional_encoding(target_embed)
        #      → shape (batch, target_length, d_model)
        #
        #   8. decoder_output = self.decoder(
        #          x=target_embed,
        #          encoder_output=encoder_output,
        #          target_padding_mask=target_padding_mask,  # (batch, target_length)
        #          target_causal_mask=target_causal_mask,
        #          source_padding_mask=source_padding_mask,  # (batch, source_length)
        #      )
        #      → shape (batch, target_length, d_model)
        #
        # === 输出投影 ===
        #   9. logits = self.projection(decoder_output)
        #      → shape (batch, target_length, vocab_size)
        #
        #  10. return logits
        #
        # 关于 mask 形状的说明:
        #   当前 mask.py 返回的所有 mask 都是为 attention score 广播设计的
        #   (batch, 1, 1, seq_len) 形状, 适合直接做 scores.masked_fill(mask, -inf)
        #
        #   但 PyTorch nn.MultiheadAttention 的 key_padding_mask 参数期望
        #   (batch, seq_len) 的 bool Tensor, True 表示忽略
        #
        #   处理方式有两种:
        #     a) 在 mask.py 中新增一个返回 (batch, seq_len) 形状的辅助函数
        #     b) 在调用处 squeeze 掉中间维度: mask.squeeze(1).squeeze(1)
        #
        #   推荐方案 a, 保持 mask.py 的接口完整性和类型安全
        raise NotImplementedError("TODO: 实现 Transformer.forward")
