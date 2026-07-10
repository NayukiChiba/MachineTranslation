"""
评估指标模块

提供机器翻译质量评估的核心指标计算:
1. Token-level Accuracy —— 逐 token 准确率,忽略 padding 位置
2. Corpus BLEU —— 轻量级语料库级别 BLEU 分数(0-100)

设计原则:
- 纯函数接口,无副作用,便于独立测试和复用
- 不依赖框架特定结构,仅使用 Python 标准库和 torch.Tensor
- BLEU 实现与原论文算法保持一致,包含平滑和长度惩罚
"""

import math
from collections import Counter

from torch import Tensor


def calculate_token_accuracy(logits: Tensor, targets: Tensor, pad_id: int) -> float:
    """
    计算忽略 padding 后的 token 准确率

    Args:
        logits: 模型输出,形状 (batch_size, seq_len, vocab_size)
        targets: 目标 token id,形状 (batch_size, seq_len)
        pad_id: padding token 的 id,对应位置不计入准确率

    Returns:
        float: token 准确率,范围 [0.0, 1.0]
    """
    # 沿词表维度取 argmax 得到预测的 token id
    predictions = logits.argmax(dim=-1)
    # 构建有效位置掩码,排除 padding token
    valid_mask = targets != pad_id
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return 0.0
    # 统计预测正确且非 padding 的位置数量
    correct_count = int(((predictions == targets) & valid_mask).sum().item())
    return correct_count / valid_count


def _count_ngrams(tokens: list[int], order: int) -> Counter[tuple[int, ...]]:
    """
    统计指定阶数的 n-gram 频次

    Args:
        tokens: token id 列表
        order: n-gram 阶数(1 为 unigram,2 为 bigram,以此类推)

    Returns:
        Counter: n-gram 到出现次数的映射
    """
    # 滑动窗口遍历 tokens,提取所有 order 阶连续子序列
    return Counter(
        tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)
    )


def calculate_bleu(
    references: list[list[int]],
    hypotheses: list[list[int]],
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """
    计算 0 到 100 范围的轻量 corpus BLEU

    实现 BLEU(Bilingual Evaluation Understudy)算法,在语料库级别聚合统计.
    默认使用 BLEU-4 并开启加一平滑,避免零 n-gram 匹配导致几何平均归零.

    Args:
        references: 参考译文列表,每个元素为一条参考译文的 token id 列表
        hypotheses: 候选译文列表,每个元素为一条候选译文的 token id 列表
        max_order: 最大 n-gram 阶数,默认 4(即 BLEU-4)
        smooth: 是否启用加一平滑,默认 True

    Returns:
        float: BLEU 分数,范围 [0.0, 100.0]

    Raises:
        ValueError: references 与 hypotheses 数量不一致或 max_order 非正数
    """
    if len(references) != len(hypotheses):
        raise ValueError("references 与 hypotheses 数量必须一致")
    if not references:
        return 0.0
    if max_order <= 0:
        raise ValueError("max_order 必须大于 0")

    # 各阶 n-gram 的匹配数与可能数
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    # 语料库级别的总参考长度与总候选长度
    reference_length = 0
    hypothesis_length = 0

    # 逐句统计各阶 n-gram 匹配情况
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, max_order + 1):
            reference_ngrams = _count_ngrams(reference, order)
            hypothesis_ngrams = _count_ngrams(hypothesis, order)
            # Counter 交集:取各 n-gram 在参考和候选中的最小出现次数
            overlap = reference_ngrams & hypothesis_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(hypothesis) - order + 1, 0)

    # 计算各阶修正后的 n-gram 精度
    precisions: list[float] = []
    for matches, possible in zip(matches_by_order, possible_by_order, strict=True):
        if possible == 0:
            continue
        if smooth:
            # 加一平滑:分子分母各加 1,避免零匹配时精度为 0
            precisions.append((matches + 1.0) / (possible + 1.0))
        else:
            precisions.append(matches / possible)

    # 如果所有阶精度为空或存在零精度,BLEU 直接归零
    if not precisions or min(precisions) <= 0 or hypothesis_length == 0:
        return 0.0

    # 几何平均:对数空间求和后取 exp,避免浮点下溢
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / len(precisions))
    # 简短惩罚:候选译文短于参考时施加指数惩罚
    length_ratio = hypothesis_length / max(reference_length, 1)
    brevity_penalty = 1.0 if length_ratio > 1.0 else math.exp(1.0 - 1.0 / length_ratio)
    return 100.0 * geo_mean * brevity_penalty
