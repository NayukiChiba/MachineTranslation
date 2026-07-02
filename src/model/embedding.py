"""
Token Embedding 与 Positional Encoding 模块

功能:
1. TokenEmbedding — 将 token id 映射为稠密向量,并做缩放
2. PositionalEncoding — 正弦/余弦位置编码,与 token embedding 相加

说明:
    - TokenEmbedding 在 lookup 后乘以 sqrt(hidden_dim),这是 Transformer 论文的标准做法
    - PositionalEncoding 使用 sin/cos 固定编码,不需要训练参数
    - 两个模块的输出直接相加即可得到 encoder/decoder 的输入
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Token 嵌入层

    将 token id 映射为 hidden_dim 维向量,并乘以 sqrt(hidden_dim)

    Args:
        vocab_size (int): 词表大小,通常为 TokenizerConfig.VOCAB_SIZE
        hidden_dim (int): 模型隐藏维度,通常为 512
        pad_id (int): <pad> token id,其 embedding 向量始终为 0

    输入:
        x: token id 序列,shape = (batch, seq_length),dtype = long

    输出:
        Tensor,shape = (batch, seq_length, hidden_dim),dtype = float32

    使用示例:
        >>> embed = TokenEmbedding(vocab_size=16000, hidden_dim=512, pad_id=0)
        >>> x = torch.tensor([[5, 3, 0], [2, 7, 9]])  # batch=2, seq_length=3
        >>> out = embed(x)
        >>> out.shape
        torch.Size([2, 3, 512])
    """

    def __init__(self, vocab_size: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        # 初始化 nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        # 提示: padding_idx 参数会让 <pad> 的 embedding 始终为 0,且不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): token id 序列,shape = (batch, seq_length)

        Returns:
            Tensor: shape = (batch, seq_length, hidden_dim)
        """
        # TODO: 实现 forward
        # 步骤:
        #   1. self.embedding(x) → shape (batch, seq_length, hidden_dim)
        x = self.embedding(x)
        #   2. 乘以 math.sqrt(self.hidden_dim)(缩放,稳定训练)
        x = x * math.sqrt(self.hidden_dim)
        #   3. 返回结果
        return x


class PositionalEncoding(nn.Module):
    """
    正弦/余弦固定位置编码

    PE(pos, 2i)   = sin(pos / 10000^(2i/hidden_dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/hidden_dim))

    Args:
        hidden_dim (int): 模型隐藏维度
        dropout (float): dropout 概率
        max_length (int): 支持的最大序列长度,默认 5000

    输入:
        x: token embedding 输出,shape = (batch, seq_length, hidden_dim)

    输出:
        Tensor,shape = (batch, seq_length, hidden_dim),已加上位置编码并过 dropout

    使用示例:
        >>> pe = PositionalEncoding(hidden_dim=512, dropout=0.1)
        >>> x = torch.randn(2, 50, 512)  # batch=2, seq_length=50
        >>> out = pe(x)
        >>> out.shape
        torch.Size([2, 50, 512])
    """

    def __init__(
        self, hidden_dim: int, dropout: float = 0.1, max_length: int = 5000
    ) -> None:
        super().__init__()
        # 步骤:
        #   1. 创建 nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        #   2. 用 register_buffer 注册位置编码矩阵 pe: shape = (1, max_length, hidden_dim)
        #      提示: 位置编码在 __init__ 中一次性算好,forward 中直接切片相加
        position_encoding_matrix = torch.zeros(max_length, hidden_dim)
        #   计算 pe 的关键步骤:
        #   a. position = torch.arange(max_length).unsqueeze(1)  → shape (max_length, 1)
        #   - postion 指的是每一个 token 的位置编号, 所以从0开始编号
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        #   b. div_term = exp( log(10000) * (-2i / hidden_dim) ) → shape (hidden_dim/2,)
        #      其中 i = arange(0, hidden_dim, 2) / hidden_dim
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )
        #   c. pe[:, 0::2] = sin(position * div_term)
        position_encoding_matrix[:, 0::2] = torch.sin(position * div_term)
        #   d. pe[:, 1::2] = cos(position * div_term)
        position_encoding_matrix[:, 1::2] = torch.cos(position * div_term)
        #   e. pe = pe.unsqueeze(0) → shape (1, max_length, hidden_dim)
        position_encoding_matrix = position_encoding_matrix.unsqueeze(0)
        #   为什么要用 register_buffer？
        #   - pe 不是可训练参数,但需要随模型一起移动到 GPU
        #   - register_buffer 会在 model.to(device) 时自动迁移
        # self.register_buffer("pe", ...)
        self.register_buffer("position_encoding_matrix", position_encoding_matrix)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape = (batch, seq_length, hidden_dim)

        Returns:
            Tensor: shape = (batch, seq_length, hidden_dim)
        """
        # 步骤:
        #   1. seq_length = x.size(1)
        seq_length = x.size(1)
        #   2. 从 self.pe 中切片 self.pe[:, :seq_length, :] → shape (1, seq_length, hidden_dim)
        #   3. x + pe_slice(广播相加)
        x = x + self.position_encoding_matrix[:, :seq_length, :]
        #   4. self.dropout(result)
        x = self.dropout(x)
        return x
