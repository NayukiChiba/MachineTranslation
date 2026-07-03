"""
Mask 生成模块

功能:
1. 生成 padding mask — 标记 source/target 序列中哪些位置是 <pad>
2. 生成 causal mask — 防止 decoder 自注意力看到未来 token

说明:
    所有 mask 返回形状为 (batch, 1, seq_len, seq_len) 或 (batch, 1, 1, seq_len),
    可以直接与 attention score 做 broadcast 加法(masked_fill 用 -inf).
"""

import torch
from torch import Tensor

from configs.defaults import TokenizerConfig


def create_padding_mask(x: Tensor, pad_id: int = TokenizerConfig.pad_id) -> Tensor:
    """
    为输入序列生成 padding mask

    Args:
        x (Tensor): 输入 token id 序列,shape = (batch, seq_length)
        pad_id (int): <pad> 对应的 token id,通常为 0

    Returns:
        Tensor: padding mask,shape = (batch, 1, 1, seq_length)
                True 表示该位置需要被 mask 掉(是 <pad>)
                False 表示该位置是有效 token

    使用示例:
        >>> source = torch.tensor([[5, 3, 0, 0], [2, 7, 9, 0]])  # batch=2, seq_length=4
        >>> mask = create_padding_mask(source, pad_id=0)
        >>> mask.shape
        torch.Size([2, 1, 1, 4])
        >>> mask[0, 0, 0]  # 第 0 句的前 4 个位置的 mask 状态
        tensor([False, False,  True,  True])

    提示:
        1. 先找到 x == pad_id 的位置
        2. 用 unsqueeze 把维度从 (batch, seq_length) 扩充到 (batch, 1, 1, seq_length)
        3. 返回的 mask 在后续 attention 中用作: scores.masked_fill(mask, -1e9)
    """
    # 步骤:
    #   1. (x == pad_id) → shape (batch, seq_length)
    padding_mask = x == pad_id
    #   2. unsqueeze(1).unsqueeze(2) → shape (batch, 1, 1, seq_length)
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    return padding_mask


def create_causal_mask(seq_length: int) -> Tensor:
    """
    生成 causal (上三角) mask,用于 decoder 自注意力

    确保位置 i 只能 attend 到位置 j <= i

    Args:
        seq_length (int): 目标序列长度
        device: 张量所在设备,默认 None 即 CPU

    Returns:
        Tensor: causal mask,shape = (1, 1, seq_length, seq_length)
                True 表示该位置需要被 mask 掉(未来 token)
                False 表示可以 attend

    使用示例:
        >>> mask = create_causal_mask(4)
        >>> mask[0, 0]  # shape (4, 4)
        tensor([[False,  True,  True,  True],
                [False, False,  True,  True],
                [False, False, False,  True],
                [False, False, False, False]])

    提示:
        1. 用 torch.triu 生成上三角全 1 矩阵
        2. 转为 bool 类型(True = 需要 mask)
        3. 用 unsqueeze(0).unsqueeze(0) 扩充 batch 和 head 维度
    """
    # 实现 causal mask 生成逻辑
    # 步骤:
    #   1. torch.ones(seq_length, seq_length) 或用 torch.triu
    causal_mask = torch.ones(seq_length, seq_length)
    #   2. triu(diagonal=1) → 上三角为 1(mask 掉未来位置)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    #   3. 转为 bool
    causal_mask = causal_mask.bool()
    #   4. unsqueeze(0).unsqueeze(0) → shape (1, 1, seq_length, seq_length)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return causal_mask


def create_combined_mask(
    source: Tensor,
    target: Tensor,
    pad_id: int = TokenizerConfig.pad_id,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    同时生成训练时所需的全部 mask

    Args:
        source (Tensor): source token ids,shape = (batch, source_length)
        target (Tensor): target token ids,shape = (batch, target_length)
        pad_id (int): <pad> token id

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - source_padding_mask: (batch, 1, 1, source_length) — encoder 自注意力用
            - target_padding_mask: (batch, 1, 1, target_length) — decoder 自注意力用
            - target_causal_mask:  (1, 1, target_length, target_length) — decoder causal mask
            - cross_mask:          (batch, 1, 1, source_length) — cross-attention 用(复用 source_padding_mask)

    提示:
        直接调用上面的 create_padding_mask 和 create_causal_mask 组合即可.
    """
    # 组合调用 create_padding_mask + create_causal_mask
    source_padding_mask = create_padding_mask(x=source, pad_id=pad_id)
    target_padding_mask = create_padding_mask(x=target, pad_id=pad_id)
    target_causal_mask = create_causal_mask(seq_length=target.size(1))
    cross_mask = source_padding_mask
    return source_padding_mask, target_padding_mask, target_causal_mask, cross_mask
