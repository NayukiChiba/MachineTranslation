"""
评估指标测试

测试 src/evaluate/metrics.py 中的 BLEU 和 token 准确率计算函数.
"""

import pytest
import torch

from src.evaluate.metrics import calculate_bleu, calculate_token_accuracy


def test_token_accuracy_ignores_padding() -> None:
    """
    测试 token 准确率计算是否正确排除填充 token

    设置 pad_id=0,第三个位置的目标 token 为 0(填充),因此在计算准确率时将忽略该位置.
    logits 中前两个位置 argmax 为 1,与 targets 匹配,所以有效位置的准确率为 2/2 = 1.0 吗？
    不——此处 targets[0, 0] = 1, targets[0, 1] = 1,
    logits[0, 0] argmax = 1 → 正确,logits[0, 1] argmax = 2 → 错误,
    第三个位置 pad 被忽略,因此正确率 = 1/2 = 0.5.
    """
    # 构造单样本 batch:3 个时间步,词表大小 3
    logits = torch.tensor([[[0.0, 3.0, 1.0], [0.0, 1.0, 3.0], [3.0, 0.0, 0.0]]])
    # 前两个为真实 token,第三个为填充
    targets = torch.tensor([[1, 1, 0]])

    assert calculate_token_accuracy(logits, targets, pad_id=0) == pytest.approx(0.5)


def test_bleu_rewards_identical_sequences() -> None:
    """
    测试 BLEU 分数的两个边界情况

    - 候选序列与参考序列完全相同时,BLEU 应为 100.0
    - 空序列的 BLEU 应为 0.0
    """
    references = [[1, 2, 3, 4], [5, 6, 7]]

    # 候选与参考完全一致 → BLEU = 100.0
    assert calculate_bleu(references, references) == pytest.approx(100.0)
    # 空序列 → BLEU = 0.0
    assert calculate_bleu([], []) == 0.0
