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

import random

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from configs.defaults import TrainConfig


def set_seed(seed: int = TrainConfig.random_seed) -> None:
    """
    固定 Python、NumPy、PyTorch 的随机种子

    同时开启 CUDA 确定性后端, 牺牲少量性能换取实验可复现

    Args:
        seed (int): 随机种子, 默认取 TrainConfig.random_seed (42)

    使用示例:
        >>> set_seed(42)

    提示:
        1. 需要分别在 random / numpy / torch / torch.cuda 四个层面调用 seed 函数
        2. torch.backends.cudnn.deterministic = True 会强制 cuDNN 使用确定性算法,
           对 Transformer(全连接 + attention)影响较小
        3. torch.backends.cudnn.benchmark = False 关闭自动算法搜索,
           避免不同运行选择不同卷积算法导致结果波动
        4. 训练脚本在最开始调用一次即可
    """
    # 固定 Python 内置 random 模块种子
    random.seed(seed)
    # 固定 NumPy 随机数生成器种子
    np.random.seed(seed)
    # 固定 PyTorch CPU 随机数生成器种子
    torch.manual_seed(seed)

    # 如果 CUDA 可用,固定所有 GPU 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 强制 cuDNN 使用确定性算法(可能略微降低性能)
    torch.backends.cudnn.deterministic = True
    # 关闭 cuDNN 自动算法搜索,避免不同运行选择不同算法
    torch.backends.cudnn.benchmark = False


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    获取训练设备

    优先使用调用方传入的设备,未指定时回退到 TrainConfig.device.
    若显式要求 CUDA 但当前环境不可用,则抛出 RuntimeError.

    Args:
        device (str | torch.device | None): 目标设备,默认为 None(使用配置值)

    Returns:
        torch.device: 可用的训练设备

    使用示例:
        >>> device = get_device()                # 使用默认配置
        >>> device = get_device("cuda:0")        # 指定 GPU
        >>> device = get_device(torch.device("cpu"))  # 指定 CPU

    提示:
        1. device or TrainConfig.device 利用短路求值选择有效设备
        2. torch.device() 可接受 str 或 torch.device,统一转为标准类型
    """
    # 解析目标设备:优先使用传入参数,未指定则取配置默认值
    selected_device = torch.device(device or TrainConfig.device)
    # 检查 CUDA 可用性:指定了 GPU 但环境不支持时直接报错
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定 CUDA 设备,但当前环境无法使用 CUDA")
    return selected_device


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
    # 统计可训练参数:requires_grad=True 的参数量
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    # 统计所有参数总量(含冻结参数如预训练 embedding)
    total = sum(param.numel() for param in model.parameters())
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
    # 执行梯度裁剪:将所有参数的梯度缩放到 max_norm 以内
    # clip_grad_norm_ 末尾下划线表示 in-place 操作
    normed_grad = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # 返回裁剪前的梯度总范数(转为 Python float,方便日志记录)
    return normed_grad.item() if isinstance(normed_grad, Tensor) else normed_grad


def move_batch_to_device(
    batch: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    """
    将 batch 字典中的所有张量迁移到指定设备

    训练循环中每个 step 需要将数据从 CPU 搬到 GPU 时调用.

    Args:
        batch (dict[str, Tensor]): 数据加载器产出的字典,键为字段名,值为张量
        device (torch.device): 目标设备(如 cuda:0 或 cpu)

    Returns:
        dict[str, Tensor]: 所有张量已迁移到目标设备的新字典

    使用示例:
        >>> batch = {"src": srcTensor, "tgt": tgtTensor}
        >>> batch = move_batch_to_device(batch, device)

    提示:
        1. 字典推导式一次性处理所有键值对,简洁高效
        2. 张量的 .to() 方法在目标设备与当前设备相同时为无操作,不会额外开销
    """
    # 遍历字典,将每个张量移动到目标设备
    return {name: value.to(device) for name, value in batch.items()}
