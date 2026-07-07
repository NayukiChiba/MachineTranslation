"""
训练日志记录模块

功能:
1. Logger — 统一日志记录器, 同时输出到终端 + TensorBoard + 文件

记录内容:
    - 训练/验证 loss + 困惑度(ppl)
    - 学习率变化曲线
    - 梯度范数
    - token 处理速度(tokens/s)
    - 配置快照

说明:
    - 不依赖 train/ 下其他模块
    - 默认启用 TensorBoard, 未安装时自动降级
    - 路径由 configs.paths 管理
    - Logger 只负责记录, 调用时机由 trainer.py 决定

输出目标:
    1. 终端(stdout)      — 进度条 + epoch 摘要
    2. TensorBoard        — 标量曲线(scalar)
    3. 日志文件(.log)     — 完整训练记录, 事后分析

使用方式:
    logger = Logger(name="training", log_dir=paths.LOGS_DIR,
                    tensorboard_dir=paths.TENSORBOARD_DIR)
    logger.start()
    logger.info("训练开始")
    logger.log_metrics(step=100, metrics={"loss": 1.23}, prefix="train")
    logger.log_epoch(epoch=1, train_metrics={...}, val_metrics={...})
    logger.close()
"""

from pathlib import Path
from typing import Optional

import torch.nn as nn

from configs import paths
from configs.defaults import TrainConfig

# 尝试导入 TensorBoard
# from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    训练日志记录器

    同时将日志写入:
        - Python logging (终端 + 文件)
        - TensorBoard (标量曲线)

    Args:
        name (str): logger 名称, 默认取 TrainConfig.logger_name
        log_dir (Path): 日志文件目录, 默认 paths.LOGS_DIR
        tensorboard_dir (Path): TensorBoard 事件目录, 默认 paths.TENSORBOARD_DIR

    使用示例:
        >>> logger = Logger(name="training")
        >>> logger.start()
        >>> logger.log_config({"epochs": 10, "lr": 1e-4})
    """

    def __init__(
        self,
        name: str = TrainConfig.logger_name,
        log_dir: Path = paths.LOGS_DIR,
        tensorboard_dir: Path = paths.TENSORBOARD_DIR,
    ) -> None:
        # 需要初始化的属性:
        #   self.name = name
        #   self.log_dir = Path(log_dir)
        #   self.tensorboard_dir = Path(tensorboard_dir)
        #   self.logger = None          — Python logging.Logger 实例
        #   self.writer = None          — TensorBoard SummaryWriter 实例
        #   self.timestamp = None       — 启动时间戳字符串
        #   self.log_file = None        — 日志文件路径
        #
        # 注意:
        #   - __init__ 只保存配置, 不创建 handler
        #   - 实际初始化在 start() 中完成
        #   - 这样设计是为了让 trainer 可以在调用 start() 前做其他初始化
        raise NotImplementedError("TODO: 实现 Logger.__init__")

    def start(self) -> "Logger":
        """
        启动 logger, 创建 logging handler 和 TensorBoard writer

        返回 self 以支持链式调用

        Returns:
            Logger: self

        提示:
            1. 创建日志目录: self.log_dir.mkdir(parents=True, exist_ok=True)

            2. 生成时间戳:
               self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
               self.log_file = self.log_dir / f"{self.name}_{self.timestamp}.log"

            3. 创建 Python logger:
               self.logger = logging.getLogger(f"{self.name}_{self.timestamp}")
               self.logger.setLevel(logging.DEBUG)
               清除已有 handlers (避免重复添加)

            4. 添加 FileHandler (DEBUG 级别, 记录所有):
               - 格式: "%(asctime)s | %(levelname)-8s | %(message)s"
               - 日期格式: "%Y-%m-%d %H:%M:%S"
               - 编码: utf-8

            5. 添加 StreamHandler(sys.stdout) (INFO 级别, 终端):
               - 格式: "%(levelname)-8s | %(message)s"

            6. 禁止日志传播到根 logger:
               self.logger.propagate = False

            7. 创建 TensorBoard writer:
               如果 TENSORBOARD_AVAILABLE:
                   tensorboard_log_dir = self.tensorboard_dir / self.timestamp
                   self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
               否则:
                   打印警告 "TensorBoard 不可用"

            8. 打印启动信息(日志文件路径 + TensorBoard 目录)
            9. return self
        """
        raise NotImplementedError("TODO: 实现 Logger.start")

    def close(self) -> None:
        """关闭 logger, 清理 TensorBoard writer 和 logging handler"""
        # 步骤:
        #   1. 关闭 TensorBoard: if self.writer: self.writer.close()
        #   2. 关闭 logging handlers: 遍历 self.logger.handlers, close + remove
        raise NotImplementedError("TODO: 实现 Logger.close")

    def info(self, message: str) -> None:
        """INFO 级别日志(终端可见)"""
        # 步骤: if self.logger: self.logger.info(message)
        raise NotImplementedError("TODO: 实现 Logger.info")

    def debug(self, message: str) -> None:
        """DEBUG 级别日志(仅文件可见)"""
        # 步骤: if self.logger: self.logger.debug(message)
        raise NotImplementedError("TODO: 实现 Logger.debug")

    def warning(self, message: str) -> None:
        """WARNING 级别日志"""
        # 步骤: if self.logger: self.logger.warning(message)
        raise NotImplementedError("TODO: 实现 Logger.warning")

    def error(self, message: str) -> None:
        """ERROR 级别日志"""
        # 步骤: if self.logger: self.logger.error(message)
        raise NotImplementedError("TODO: 实现 Logger.error")

    def log_config(self, config: dict) -> None:
        """
        打印训练配置摘要

        Args:
            config (dict): 配置键值对
        """
        # 步骤:
        #   1. self.info("=" * 60)
        #   2. self.info("训练配置")
        #   3. self.info("-" * 60)
        #   4. 遍历 config.items(), 逐行打印: self.info(f"  {k}: {v}")
        #   5. self.info("=" * 60)
        raise NotImplementedError("TODO: 实现 Logger.log_config")

    def log_model_info(self, model: nn.Module) -> None:
        """
        打印模型参数量统计

        Args:
            model (nn.Module): 模型
        """
        # 步骤:
        #   1. 统计参数(内联, 不依赖 utils):
        #      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #      total = sum(p.numel() for p in model.parameters())
        #   2. self.info(f"可训练参数: {trainable:,}")
        #   3. self.info(f"总参数: {total:,}")
        #   4. 打印设备信息: next(model.parameters()).device
        raise NotImplementedError("TODO: 实现 Logger.log_model_info")

    def log_metrics(self, step: int, metrics: dict, prefix: str = "") -> None:
        """
        记录步级指标 (写入日志文件 + TensorBoard)

        Args:
            step (int): 当前全局步数
            metrics (dict): 指标字典, 如 {"loss": 1.23, "lr": 1e-4}
            prefix (str): 前缀, 如 "train" / "val"

        提示:
            1. 格式化指标字符串:
               metrics_str = " | ".join(
                   f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                   for k, v in metrics.items()
               )

            2. 以 DEBUG 级别写日志(步级指标不刷屏):
               如果有 prefix: self.debug(f"[{prefix}] Step {step:6d} | {metrics_str}")
               否则: self.debug(f"Step {step:6d} | {metrics_str}")

            3. 写入 TensorBoard:
               if self.writer:
                   for key, value in metrics.items():
                       if isinstance(value, (int, float)):
                           tag = f"{prefix}/{key}" if prefix else key
                           self.writer.add_scalar(tag, value, step)
        """
        raise NotImplementedError("TODO: 实现 Logger.log_metrics")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: Optional[dict] = None,
    ) -> None:
        """
        记录 epoch 级摘要 (终端可见 + TensorBoard)

        Args:
            epoch (int): 当前 epoch
            train_metrics (dict): 训练指标
            val_metrics (dict | None): 验证指标

        提示:
            1. 打印分隔线: self.info("=" * 60)
            2. self.info(f"Epoch {epoch}")
            3. 格式化并打印 train_metrics (INFO 级别)
            4. 如果有 val_metrics, 格式化并打印 (INFO 级别)
            5. 打印结束分隔线
            6. 写入 TensorBoard (按 epoch):
               - train/{key}: epoch
               - val/{key}: epoch
        """
        raise NotImplementedError("TODO: 实现 Logger.log_epoch")
