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
        之后按余弦曲线退火到 min_lr。
        公式:
          warmup:  lr = initial_lr * step / warmup_steps
          cosine:  lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * progress))
        这是 "Attention Is All You Need" 论文使用的策略。

    cosine:
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * step / total_steps))
        无 warmup, 直接从初始值开始衰减。

    step:
        lr = initial_lr * gamma^(step // step_size)
        阶梯式衰减, 每 step_size 步乘一次 gamma。

    exponential:
        lr = initial_lr * gamma^step
        连续指数衰减, 后期学习率极低。

    实现选择:
        cosine_warmup 推荐用 LambdaLR + 自定义 lambda 函数:
          - 比 SequentialLR 更灵活, 不依赖 PyTorch 版本
          - lambda 函数接收当前步数, 返回相对于 initial_lr 的缩放因子
"""

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
    scheduler_type: str = TrainConfig.scheduler_type,
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
        scheduler_type (str): 调度器类型:
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
    # 步骤:
    #   1. 获取初始学习率:
    #      initial_lr = optimizer.param_groups[0]["lr"]
    #      min_lr = initial_lr * min_lr_ratio
    #
    #   2. if scheduler_type == "constant":
    #        return None
    #
    #   3. elif scheduler_type == "cosine":
    #        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    #
    #   4. elif scheduler_type == "step":
    #        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    #
    #   5. elif scheduler_type == "exponential":
    #        return ExponentialLR(optimizer, gamma=gamma)
    #
    #   6. elif scheduler_type == "cosine_warmup":
    #        def lr_lambda(current_step): ...  # 见上方提示
    #        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    #
    #   7. else:
    #        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    raise NotImplementedError("TODO: 实现 create_scheduler")
