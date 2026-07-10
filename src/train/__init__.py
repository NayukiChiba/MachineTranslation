"""
训练模块

子模块:
    - utils.py            : set_seed / get_device / count_parameters /
                            clip_gradients / move_batch_to_device
    - optimizer.py        : create_optimizer
    - scheduler.py        : create_scheduler
    - early_stopping.py   : EarlyStopping
    - checkpoint.py       : save_checkpoint / load_checkpoint
    - logger.py           : Logger
    - trainer.py          : Trainer (唯一调度者)

依赖规则:
    trainer.py 可以 import 所有 train/ 下模块。
    optimizer / scheduler / early_stopping / checkpoint / logger / utils
    彼此之间不互相 import。
"""

# 训练调度器
# 检查点管理
from src.train.checkpoint import load_checkpoint, save_checkpoint

# 早停控制器
from src.train.early_stopping import EarlyStopping

# 日志记录器
from src.train.logger import Logger

# 优化器工厂
from src.train.optimizer import create_optimizer

# 学习率调度器工厂
from src.train.scheduler import create_scheduler
from src.train.trainer import Trainer

# 工具函数
from src.train.utils import (
    clip_gradients,
    count_parameters,
    get_device,
    move_batch_to_device,
    set_seed,
)

__all__ = [
    "Trainer",
    "set_seed",
    "get_device",
    "count_parameters",
    "clip_gradients",
    "move_batch_to_device",
    "create_optimizer",
    "create_scheduler",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    "Logger",
]
