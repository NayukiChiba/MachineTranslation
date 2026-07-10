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

import logging
import sys
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn

from configs import paths
from configs.defaults import TrainConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


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
        enable_tensorboard: bool = TrainConfig.enable_tensorboard,
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
        self.name = name
        self.log_dir = Path(log_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.enable_tensorboard = enable_tensorboard
        self.logger: logging.Logger | None = None
        self.writer: Any = None
        self.timestamp: str | None = None
        self.log_file: Path | None = None

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
        if self.logger is not None:
            return self

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_file = self.log_dir / f"{self.name}_{self.timestamp}.log"

        logger = logging.getLogger(f"{self.name}_{self.timestamp}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        self.logger = logger

        if self.enable_tensorboard and SummaryWriter is not None:
            tensorboard_log_dir = self.tensorboard_dir / self.timestamp
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        elif self.enable_tensorboard:
            logger.warning("TensorBoard 不可用,已仅启用控制台和文件日志")

        logger.info("日志文件: %s", self.log_file)
        return self

    def close(self) -> None:
        """关闭 logger, 清理 TensorBoard writer 和 logging handler"""
        # 步骤:
        #   1. 关闭 TensorBoard: if self.writer: self.writer.close()
        #   2. 关闭 logging handlers: 遍历 self.logger.handlers, close + remove
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.logger is not None:
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)
            self.logger = None

    def info(self, message: str) -> None:
        """INFO 级别日志(终端可见)"""
        # 步骤: if self.logger: self.logger.info(message)
        self._require_logger().info(message)

    def debug(self, message: str) -> None:
        """DEBUG 级别日志(仅文件可见)"""
        # 步骤: if self.logger: self.logger.debug(message)
        self._require_logger().debug(message)

    def warning(self, message: str) -> None:
        """WARNING 级别日志"""
        # 步骤: if self.logger: self.logger.warning(message)
        self._require_logger().warning(message)

    def error(self, message: str) -> None:
        """ERROR 级别日志"""
        # 步骤: if self.logger: self.logger.error(message)
        self._require_logger().error(message)

    def _require_logger(self) -> logging.Logger:
        """返回已启动的 logger,未启动时明确报错"""
        if self.logger is None:
            raise RuntimeError("Logger 尚未启动,请先调用 start()")
        return self.logger

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
        self.info("=" * 60)
        self.info("训练配置")
        self.info("-" * 60)
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 60)

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
        # 统计可训练参数数量(requires_grad=True)
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        # 统计全部参数数量
        total_count = sum(parameter.numel() for parameter in model.parameters())
        # 获取模型所在设备
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else "无参数"
        self.info(f"可训练参数: {trainable_count:,}")
        self.info(f"总参数: {total_count:,}")
        self.info(f"模型设备: {device}")

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
        # 格式化指标为字符串,浮点数保留4位小数
        metrics_text = " | ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in metrics.items()
        )
        # 构建带前缀的日志消息,DEBUG级别避免刷屏
        message_prefix = f"[{prefix}] " if prefix else ""
        self.debug(f"{message_prefix}Step {step:6d} | {metrics_text}")

        # 写入 TensorBoard 标量曲线
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, Real):
                    tag = f"{prefix}/{key}" if prefix else key
                    self.writer.add_scalar(tag, float(value), step)

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

        def format_metrics(metrics: dict) -> str:
            """格式化指标字典为字符串,浮点数保留4位小数"""
            return " | ".join(
                f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in metrics.items()
            )

        # 打印 epoch 摘要到头(INFO级别,终端可见)
        self.info("=" * 60)
        self.info(f"Epoch {epoch + 1}")
        self.info(f"train | {format_metrics(train_metrics)}")
        if val_metrics is not None:
            self.info(f"val   | {format_metrics(val_metrics)}")
        self.info("=" * 60)

        # 写入 TensorBoard(以epoch为横轴)
        if self.writer is not None:
            for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
                if metrics is None:
                    continue
                for key, value in metrics.items():
                    if isinstance(value, Real):
                        self.writer.add_scalar(f"{prefix}/{key}", float(value), epoch)
