"""
训练工具函数模块

功能:
1. set_seed — 固定随机种子, 保证实验可复现
2. get_device — 获取可用训练设备
3. count_parameters — 统计模型可训练参数数量
4. clip_gradients — 梯度裁剪, 防止梯度爆炸
5. move_batch_to_device — 将 batch 数据迁移到指定设备

说明:
    - 本模块不依赖 train/ 下其他模块, 可独立使用
    - 所有超参数从 configs.defaults.TrainConfig 读取
    - 梯度裁剪阈值等参数由 TrainConfig 统一管理
"""

import torch.nn as nn
from torch import Tensor

from configs.defaults import TrainConfig

# def set_seed(seed: int = TrainConfig.random_seed) -> None:
#     """
#     固定 Python、NumPy、PyTorch 的随机种子

#     同时开启 CUDA 确定性后端, 牺牲少量性能换取实验可复现

#     Args:
#         seed (int): 随机种子, 默认取 TrainConfig.random_seed (42)

#     使用示例:
#         >>> set_seed(42)

#     提示:
#         1. 需要分别在 random / numpy / torch / torch.cuda 四个层面调用 seed 函数
#         2. torch.backends.cudnn.deterministic = True 会强制 cuDNN 使用确定性算法,
#            对 Transformer(全连接 + attention)影响较小
#         3. torch.backends.cudnn.benchmark = False 关闭自动算法搜索,
#            避免不同运行选择不同卷积算法导致结果波动
#         4. 训练脚本在最开始调用一次即可
#     """
#     # 步骤:
#     #   1. random.seed(seed) — 固定 Python 内置 random
#     #
#     #   2. np.random.seed(seed) — 固定 NumPy 随机数
#     #
#     #   3. torch.manual_seed(seed) — 固定 PyTorch CPU 随机数
#     #
#     #   4. torch.cuda.manual_seed_all(seed) — 固定所有 GPU 随机数
#     #
#     #   5. torch.backends.cudnn.deterministic = True — cuDNN 确定性模式
#     #
#     #   6. torch.backends.cudnn.benchmark = False — 关闭自动算法搜索
#     raise NotImplementedError("TODO: 实现 set_seed")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    统计模型参数量

    Args:
        model (nn.Module): PyTorch 模型

    Returns:
        tuple[int, int]: (可训练参数量, 总参数量)

    使用示例:
        >>> trainable, total = count_parameters(model)
        >>> print(f"可训练参数: {trainable:,}")

    提示:
        1. model.parameters() 返回所有参数张量迭代器
        2. param.numel() 返回张量元素总数
        3. param.requires_grad 判断是否参与梯度更新
        4. 冻结部分参数(如 embedding)后, 可训练数 < 总数
    """
    # 步骤:
    #   1. trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    #   2. total = sum(p.numel() for p in model.parameters())
    total = sum(param.numel() for param in model.parameters())
    #   3. return trainable, total
    return trainable, total


def clip_gradients(
    model: nn.Module, max_norm: float = TrainConfig.gradient_clip_norm
) -> float:
    """
    梯度裁剪, 防止训练初期梯度爆炸

    应在 backward() 之后、optimizer.step() 之前调用

    Args:
        model (nn.Module): 已执行 backward 的模型
        max_norm (float): 最大梯度范数, 默认 1.0

    Returns:
        float: 裁剪前的梯度总范数

    使用示例:
        >>> loss.backward()
        >>> grad_norm = clip_gradients(model)
        >>> optimizer.step()

    提示:
        1. 使用 nn.utils.clip_grad_norm_(model.parameters(), max_norm)
           函数名末尾的下划线表示 in-place 操作
        2. 返回值是裁剪前的范数, 用 .item() 转为 Python float
        3. AMP 场景: 需先 scaler.unscale_(optimizer) 再裁剪,
           否则裁剪的是缩放后的梯度
        4. 裁剪后再调用 optimizer.step()
    """
    # 步骤:
    #   1. grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    normed_grad = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    #   2. return grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
    return normed_grad.item() if isinstance(normed_grad, Tensor) else normed_grad
