"""
优化器工厂模块

功能:
1. create_optimizer — 根据配置创建优化器实例

支持的优化器:
    - adam  : Adam, 标准自适应学习率
    - adamw : AdamW, 权重衰减与学习率解耦 (Transformer 推荐)
    - sgd   : SGD + Momentum

说明:
    - 不依赖 train/ 下其他模块
    - 所有超参数从 configs.defaults.TrainConfig 读取
    - trainer.py 是唯一调用方

为什么 Transformer 推荐 AdamW:
    Adam 的 weight decay 实现等价于 L2 正则化, 与自适应学习率耦合.
    AdamW 将 weight decay 作为独立衰减项, 与梯度更新解耦.
    论文: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
"""

import torch.nn as nn
import torch.optim as optim

from configs.defaults import TrainConfig


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = TrainConfig.optimizer_type,
    learning_rate: float = TrainConfig.learning_rate,
    weight_decay: float = TrainConfig.weight_decay,
    betas: tuple[float, float] = (
        TrainConfig.optimizer_beta1,
        TrainConfig.optimizer_beta2,
    ),
    eps: float = TrainConfig.optimizer_eps,
    momentum: float = TrainConfig.sgd_momentum,
) -> optim.Optimizer:
    """
    根据配置创建优化器

    Args:
        model (nn.Module): 待优化模型, 内部调用 model.parameters()
        optimizer_type (str): 优化器类型:
            - "adam"  : Adam
            - "adamw" : AdamW (推荐)
            - "sgd"   : SGD + Momentum
        learning_rate (float): 初始学习率, 默认 1e-4
        weight_decay (float): 权重衰减系数, 默认 0.0
        betas (tuple): Adam/AdamW 动量参数 (beta1, beta2), 默认 (0.9, 0.999)
        eps (float): Adam/AdamW 数值稳定项, 默认 1e-8
        momentum (float): SGD 动量系数, 默认 0.9

    Returns:
        optim.Optimizer: PyTorch 优化器实例

    Raises:
        ValueError: optimizer_type 不在 ["adam", "adamw", "sgd"] 中

    使用示例:
        >>> optimizer = create_optimizer(model, optimizer_type="adamw")

    提示:
        1. 获取参数: params = model.parameters()

        2. Adam / AdamW 的参数含义:
           - lr: 初始学习率
           - betas: (beta1, beta2), 一阶和二阶动量衰减系数
           - eps: 防止除零的小常数
           - weight_decay: 权重衰减强度

        3. SGD 使用 momentum 参数而非 betas:
           - optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)

        4. 如果 optimizer_type 不在支持列表中, raise ValueError 并列出可选值

        5. 可选扩展: 对不同参数组使用不同学习率(如 embedding 层更小 lr):
           传入 param_groups 列表而非 model.parameters(), 当前版本暂不实现
    """
    # 步骤:
    #   1. params = model.parameters()
    params = model.parameters()
    #   2. if optimizer_type == "adam":
    #        return optim.Adam(params, lr=learning_rate, betas=betas,
    #                          eps=eps, weight_decay=weight_decay)
    if optimizer_type == "adam":
        return optim.Adam(
            params, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay
        )
    #   3. elif optimizer_type == "adamw":
    #        return optim.AdamW(params, lr=learning_rate, betas=betas,
    #                           eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(
            params=params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    #   4. elif optimizer_type == "sgd":
    #        return optim.SGD(params, lr=learning_rate, momentum=momentum,
    #                         weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(
            params=params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    #
    #   5. else:
    #        raise ValueError(
    #            f"不支持的优化器类型: {optimizer_type}, 可选: adam / adamw / sgd"
    #        )
    else:
        raise ValueError(
            f"不支持的优化器类型: {optimizer_type}, 可选: adam / adamw / sgd"
        )
