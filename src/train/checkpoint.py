"""
模型检查点模块

功能:
1. save_checkpoint — 保存完整训练状态到磁盘
2. load_checkpoint — 从磁盘恢复完整训练状态

检查点内容:
    - model_state_dict      : 模型参数
    - optimizer_state_dict  : 优化器动量/方差
    - scheduler_state_dict  : 调度器步数/学习率(可选)
    - early_stopping_state  : 早停计数/最佳指标(可选)
    - epoch                 : 当前 epoch
    - global_step           : 全局训练步数
    - best_metric           : 历史最佳验证指标
    - scaler_state_dict     : AMP scaler 状态(可选)

说明:
    - 不依赖 train/ 下其他模块
    - 保存路径由 configs.paths 管理
    - 两个文件: best_model.pth (验证最佳) + last_model.pth (断点续训)
    - best: 仅当 is_best=True 时更新
    - last: 每次 save 都更新
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn
import torch.optim as optim

from configs import paths


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    best_metric: float,
    is_best: bool,
    scheduler: Any = None,
    early_stopping: Any = None,
    scaler: Any = None,
    config: Optional[Dict] = None,
    checkpoint_dir: Path = paths.CHECKPOINTS_DIR,
) -> None:
    """
    保存完整训练状态到检查点

    始终写入 last_model.pth, is_best 为 True 时同时写入 best_model.pth

    Args:
        model (nn.Module): 模型
        optimizer (optim.Optimizer): 优化器
        epoch (int): 当前 epoch (0-indexed)
        global_step (int): 全局训练步数
        best_metric (float): 当前最佳验证指标
        is_best (bool): 是否是最佳 checkpoint
        scheduler: 学习率调度器, 有 state_dict() 方法, 可为 None
        early_stopping: 早停控制器, 有 state_dict() 方法, 可为 None
        scaler: AMP GradScaler, 有 state_dict() 方法, 可为 None
        config (dict | None): 训练配置快照
        checkpoint_dir (Path): 检查点保存目录

    使用示例:
        >>> save_checkpoint(model, optimizer, epoch=5, global_step=5000,
        ...                 best_metric=2.1, is_best=True)

    提示:
        1. 构建 checkpoint 字典:
           checkpoint = {
               "epoch": epoch,
               "global_step": global_step,
               "best_metric": best_metric,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
           }

        2. 可选字段(判断不为 None 再写入):
           - scheduler_state_dict
           - early_stopping_state   (调用 early_stopping.state_dict())
           - scaler_state_dict
           - config

        3. 确保目录存在:
           checkpoint_dir.mkdir(parents=True, exist_ok=True)

        4. 保存 last:
           torch.save(checkpoint, checkpoint_dir / "last_model.pth")

        5. 如果 is_best:
           torch.save(checkpoint, checkpoint_dir / "best_model.pth")

        6. 打印保存信息, 包含 epoch / step / metric
    """
    # 步骤详见上方提示
    raise NotImplementedError("TODO: 实现 save_checkpoint")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Any = None,
    early_stopping: Any = None,
    scaler: Any = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    从检查点文件恢复训练状态

    Args:
        checkpoint_path (Path): 检查点文件路径
        model (nn.Module): 待恢复参数的模型
        optimizer (Optimizer | None): 待恢复的优化器, 不为 None 时加载
        scheduler: 待恢复的调度器, 不为 None 时加载
        early_stopping: 待恢复的早停控制器, 需要有 load_state_dict() 方法
        scaler: 待恢复的 AMP scaler, 不为 None 时加载
        device (str): 加载到的目标设备, 默认 "cpu"

    Returns:
        dict: 检查点元信息, 包含 epoch / global_step / best_metric 等

    使用示例:
        >>> info = load_checkpoint(paths.LAST_MODEL_PATH, model, optimizer)
        >>> start_epoch = info["epoch"] + 1

    提示:
        1. 检查文件是否存在:
           if not checkpoint_path.exists():
               返回初始值 {"epoch": 0, "global_step": 0, "best_metric": float("inf")}

        2. torch.load(checkpoint_path, map_location=device) 加载检查点

        3. 恢复模型:
           model.load_state_dict(checkpoint["model_state_dict"])

        4. 恢复优化器(需传入 optimizer):
           optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        5. 恢复调度器(需传入 scheduler):
           scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        6. 恢复早停(需传入 early_stopping):
           early_stopping.load_state_dict(checkpoint["early_stopping_state"])

        7. 恢复 scaler(需传入 scaler):
           scaler.load_state_dict(checkpoint["scaler_state_dict"])

        8. 返回元信息字典

        9. 恢复后需手动将 optimizer 状态迁移到正确设备:
           如果模型在 GPU 上但 checkpoint 在 CPU 加载, optimizer 的动量缓存
           可能仍在 CPU, 需要额外处理或直接用 map_location=device 匹配
    """
    # 步骤详见上方提示
    raise NotImplementedError("TODO: 实现 load_checkpoint")
