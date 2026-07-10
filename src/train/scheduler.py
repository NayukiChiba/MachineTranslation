"""
学习率调度器工厂模块

功能:
1. create_scheduler — 根据配置创建学习率调度器

支持的调度策略:
    - constant        : 恒定学习率, 不做调整 (baseline / fine-tune)
    - cosine          : 余弦退火, 从初始值平滑衰减
    - step            : StepLR, 每隔固定步数乘 gamma 衰减
    - exponential     : ExponentialLR, 每步指数衰减
    - cosine_warmup   : 线性 warmup + 余弦退火 (Transformer 标配)

说明:
    - 不依赖 train/ 下其他模块
    - 所有超参数从 configs.defaults.TrainConfig 读取
    - 调度器在 optimizer.step() 之后调用 scheduler.step()
    - constant 类型可以返回 None, trainer 中做判断跳过

调度策略详解:

    cosine_warmup (推荐):
        前 warmup_steps 步学习率从 0 线性增长到 initial_lr,
        之后按余弦曲线退火到 min_lr.
        公式:
          warmup:  lr = initial_lr * step / warmup_steps
          cosine:  lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * progress))
        这是 "Attention Is All You Need" 论文使用的策略.

    cosine:
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * step / total_steps))
        无 warmup, 直接从初始值开始衰减.

    step:
        lr = initial_lr * gamma^(step // step_size)
        阶梯式衰减, 每 step_size 步乘一次 gamma.

    exponential:
        lr = initial_lr * gamma^step
        连续指数衰减, 后期学习率极低.

    实现选择:
        cosine_warmup 推荐用 LambdaLR + 自定义 lambda 函数:
          - 比 SequentialLR 更灵活, 不依赖 PyTorch 版本
          - lambda 函数接收当前步数, 返回相对于 initial_lr 的缩放因子
"""

import math
from typing import Literal

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    StepLR,
)

from configs.defaults import TrainConfig


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: Literal[
        "constant", "cosine", "step", "exponential", "cosine_warmup"
    ] = TrainConfig.scheduler_type,
    total_steps: int = TrainConfig.total_training_steps,
    warmup_steps: int = TrainConfig.scheduler_warmup_steps,
    min_lr_ratio: float = TrainConfig.scheduler_min_learning_rate_ratio,
    step_size: int = TrainConfig.scheduler_step_size,
    gamma: float = TrainConfig.scheduler_gamma,
) -> LambdaLR | CosineAnnealingLR | StepLR | ExponentialLR | None:
    """
    根据配置创建学习率调度器

    Args:
        optimizer (optim.Optimizer): 已创建的优化器
        scheduler_type (Literal): 调度器类型:
            "constant" / "cosine" / "step" / "exponential" / "cosine_warmup"
        total_steps (int): 总训练步数, cosine / cosine_warmup 需要
        warmup_steps (int): warmup 步数, cosine_warmup 需要, 默认 500
        min_lr_ratio (float): 最小学习率比例 (min_lr / initial_lr), 默认 0.01
        step_size (int): StepLR 衰减步长, 默认 1000
        gamma (float): StepLR / ExponentialLR 衰减因子, 默认 0.8

    Returns:
        LRScheduler | None: 调度器实例, constant 时返回 None

    使用示例:
        >>> scheduler = create_scheduler(optimizer, scheduler_type="cosine_warmup")
        >>> for batch in dataloader:
        ...     loss.backward()
        ...     optimizer.step()
        ...     if scheduler:
        ...         scheduler.step()

    提示:
        1. constant:
           直接 return None, trainer 中判断 if scheduler: scheduler.step()

        2. cosine:
           CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
           其中 min_lr = optimizer.param_groups[0]["lr"] * min_lr_ratio

        3. step:
           StepLR(optimizer, step_size=step_size, gamma=gamma)

        4. exponential:
           ExponentialLR(optimizer, gamma=gamma)
           可选: gamma = min_lr_ratio ** (1.0 / total_steps),
           确保结束时学习率降到 min_lr

        5. cosine_warmup (推荐实现):
           用 LambdaLR + 自定义 lr_lambda 函数:

           def lr_lambda(current_step):
               # warmup 阶段: 线性增长
               if current_step < warmup_steps:
                   return current_step / max(warmup_steps, 1)
               # cosine 退火阶段
               progress = (current_step - warmup_steps) / max(
                   total_steps - warmup_steps, 1
               )
               cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
               return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

           return LambdaLR(optimizer, lr_lambda=lr_lambda)

           注意:
           - lr_lambda 返回的是缩放因子, 实际 lr = initial_lr * factor
           - warmup 阶段 factor 从 0 到 1
           - cosine 阶段 factor 从 1 到 min_lr_ratio
           - max(denominator, 1) 防止除零

        6. 其他类型应 raise ValueError
    """
    # 读取优化器当前学习率,作为所有调度器的初始基准.
    # 当前项目默认所有参数组使用同一学习率,因此取第一个参数组即可.
    initial_lr = optimizer.param_groups[0]["lr"]

    # 将最小学习率配置为初始学习率的比例,便于不同 lr 下复用同一配置.
    min_lr = initial_lr * min_lr_ratio

    # min_lr_ratio 超出范围会导致学习率反向增长或变成负数,直接显式报错.
    if min_lr_ratio < 0 or min_lr_ratio > 1:
        raise ValueError(f"最小学习率比例必须在 [0, 1] 范围内: {min_lr_ratio}")

    # constant 表示不启用调度器,由 trainer 判断 None 后跳过 scheduler.step().
    if scheduler_type == "constant":
        return None

    # 余弦退火:从 initial_lr 平滑衰减到 min_lr,适合已知总步数的训练.
    if scheduler_type == "cosine":
        if total_steps <= 0:
            raise ValueError(f"cosine 调度器要求 total_steps > 0: {total_steps}")

        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )

    # 阶梯衰减:每隔 step_size 个 step 将学习率乘以 gamma.
    if scheduler_type == "step":
        if step_size <= 0:
            raise ValueError(f"step 调度器要求 step_size > 0: {step_size}")

        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    # 指数衰减:每次 scheduler.step() 都将学习率乘以 gamma.
    if scheduler_type == "exponential":
        return ExponentialLR(optimizer, gamma=gamma)

    # Transformer 常用策略:先线性 warmup,再进行余弦退火.
    if scheduler_type == "cosine_warmup":
        if total_steps <= 0:
            raise ValueError(f"cosine_warmup 调度器要求 total_steps > 0: {total_steps}")
        if warmup_steps < 0:
            raise ValueError(
                f"cosine_warmup 调度器要求 warmup_steps >= 0: {warmup_steps}"
            )

        def calculate_lr_factor(current_step: int) -> float:
            """计算当前 step 的学习率缩放比例"""
            # warmup 阶段从 0 线性增长到 1,避免训练初期更新过猛.
            if current_step < warmup_steps:
                return current_step / max(warmup_steps, 1)

            # 退火阶段 progress 从 0 增长到 1,对应从峰值学习率降到最小比例.
            progress = (current_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )

            # 将进度限制在 [0, 1],防止超过 total_steps 后学习率再次上升.
            progress = min(max(progress, 0.0), 1.0)

            # cosine_decay 从 1 平滑衰减到 0,再映射到 [min_lr_ratio, 1].
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda=calculate_lr_factor)

    # 类型注解可以拦住大多数误用;运行时仍保留检查,方便配置文件错误定位.
    raise ValueError(
        f"不支持的调度器类型: {scheduler_type}, "
        "可选: constant / cosine / step / exponential / cosine_warmup"
    )
