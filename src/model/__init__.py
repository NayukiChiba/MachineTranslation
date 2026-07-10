"""
机器翻译模型结构模块

功能:
1. 定义 Transformer 模型的网络结构
2. 实现词嵌入与位置编码
3. 提供编码器-解码器架构
4. 生成 attention mask(填充掩码与因果掩码)

包含的子模块:
- transformer: Transformer 主体模型
- embedding: 词嵌入层与位置编码
- encoder: Transformer 编码器
- decoder: Transformer 解码器
- mask: attention mask 生成函数

使用方法:
    from src.model import Transformer
    from src.model import TokenEmbedding, PositionalEncoding
    from src.model import create_padding_mask, create_causal_mask
"""

# 完整 Transformer 模型
# 解码器
from src.model.decoder import TransformerDecoder, TransformerDecoderLayer

# 嵌入层
from src.model.embedding import PositionalEncoding, TokenEmbedding

# 编码器
from src.model.encoder import TransformerEncoder, TransformerEncoderLayer

# Mask 生成函数
from src.model.mask import create_causal_mask, create_combined_mask, create_padding_mask
from src.model.transformer import Transformer

__all__ = [
    "Transformer",
    "TokenEmbedding",
    "PositionalEncoding",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "create_padding_mask",
    "create_causal_mask",
    "create_combined_mask",
]
