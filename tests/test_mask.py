"""
Attention Mask 测试模块

测试功能:
1. Padding Mask:验证 padding mask 正确标记填充位置
2. Causal Mask:验证因果遮罩的下三角结构
3. Combined Mask:验证组合遮罩的形状和设备一致性

使用方法:
    pytest tests/test_mask.py -v
"""

import torch

from src.model.mask import create_causal_mask, create_combined_mask, create_padding_mask


def test_create_padding_mask_marks_only_pad_tokens() -> None:
    """
    测试 padding mask 是否正确标记填充 token

    验证点:
    - mask 的形状为 (batch_size, 1, 1, seq_len)
    - mask 的数据类型为 bool
    - 实际 token 位置为 False(不屏蔽),pad 位置为 True(屏蔽)
    """
    # 构造测试数据:batch_size=2,每行末尾是 pad_id=0
    token_ids = torch.tensor([[4, 0, 0], [5, 6, 0]])
    mask = create_padding_mask(token_ids, pad_id=0)

    # 验证形状:多头的 head 维度已提前广播
    assert mask.shape == (2, 1, 1, 3)
    # 验证数据类型为布尔型遮罩
    assert mask.dtype == torch.bool
    # 第一行:位置0是真实token(False),位置1、2是pad(True)
    assert mask[0, 0, 0].tolist() == [False, True, True]


def test_create_causal_and_combined_masks() -> None:
    """
    测试因果遮罩和组合遮罩的生成逻辑

    验证点:
    - 因果遮罩为下三角结构,未来位置被屏蔽
    - 组合遮罩返回三个分量:源、目标、因果
    - 组合遮罩中 causal_mask 的设备与 target 一致
    """
    # 构造源序列和目标序列,均含一个 pad token
    source = torch.tensor([[4, 3, 0]])
    target = torch.tensor([[2, 5, 0]])

    # 生成长度为3的因果遮罩(下三角矩阵)
    causal_mask = create_causal_mask(3, device="cpu")
    # 同时生成 padding、causal 组合遮罩
    masks = create_combined_mask(source, target, pad_id=0)

    # 验证因果遮罩形状
    assert causal_mask.shape == (1, 1, 3, 3)
    # 下三角结构:位置0只能看到自己,位置1看到0-1,位置2看到0-2
    assert causal_mask[0, 0].tolist() == [
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ]

    # 验证源序列 padding mask 形状
    assert masks[0].shape == (1, 1, 1, 3)
    # 验证因果遮罩与 target 在同一设备上
    assert masks[2].device == target.device
