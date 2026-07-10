"""
训练主循环模块

功能:
1. Trainer — 训练流程唯一调度者, 编排所有训练组件

调度关系:
    Trainer (唯一调度者)
    ├── model              (Transformer)
    ├── optimizer          (create_optimizer)
    ├── scheduler          (create_scheduler)
    ├── criterion          (nn.CrossEntropyLoss)
    ├── early_stopping     (EarlyStopping)
    ├── logger             (Logger)
    ├── checkpoint         (save_checkpoint / load_checkpoint)
    ├── utils              (set_seed / get_device / count_parameters /
    │                        clip_gradients / move_batch_to_device)
    └── AMP scaler         (torch.amp.GradScaler, 可选)

训练流程(每个 epoch):
    1. train_epoch()    — 遍历训练集, 逐 batch 前向/反向/更新
    2. validate_epoch() — 遍历验证集, 计算 loss + accuracy (无梯度)
    3. 记录日志, 早停判断, 保存检查点

模块依赖规则:
    - trainer.py 可以 import train/ 下所有模块
    - optimizer / scheduler / early_stopping / checkpoint / logger / utils
      彼此之间不互相 import

入口命令:
    python main.py train

Teacher Forcing:
    训练时使用 ground truth 作为 decoder 输入(target_input_ids),
    推理时使用模型自己生成的 token.
    这是 seq2seq 模型训练的标准做法.

loss 计算:
    logits shape: (batch, target_length, vocab_size)
    labels shape: (batch, target_length)
    需要 transpose logits → (batch, vocab_size, target_length)
    设置 ignore_index=pad_id 忽略 <pad> 位置的 loss
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs import paths
from configs.defaults import TokenizerConfig, TrainConfig
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.logger import Logger
from src.train.optimizer import create_optimizer
from src.train.scheduler import create_scheduler
from src.train.utils import (
    clip_gradients,
    get_device,
    move_batch_to_device,
    set_seed,
)


class Trainer:
    """
    训练调度器

    负责编排完整的训练流程:
        初始化组件 → 训练循环 → 验证 → 日志 → 早停 → 保存

    Args:
        model (Transformer): Transformer 翻译模型
        train_loader (DataLoader): 训练集 DataLoader
        val_loader (DataLoader): 验证集 DataLoader
        device (torch.device | None): 训练设备, None 则自动选择
        resume_from (Path | None): 恢复训练的检查点路径

    使用示例:
        >>> model = Transformer(vocab_size=16000)
        >>> train_dl, val_dl, _ = create_dataloaders()
        >>> trainer = Trainer(model, train_dl, val_dl)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | str | None = None,
        resume_from: Path | None = None,
        config: TrainConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        criterion: nn.Module | None = None,
        early_stopping: EarlyStopping | None = None,
        logger: Logger | None = None,
        checkpoint_dir: Path = paths.CHECKPOINTS_DIR,
    ) -> None:
        # ============================================================
        # 1. 基础设置
        # ============================================================
        #   - set_seed(TrainConfig.random_seed)
        #   - self.device = device or get_device()
        #   - self.model = model.to(self.device)
        #   - self.train_loader = train_loader
        #   - self.val_loader = val_loader
        #
        # ============================================================
        # 2. 日志
        # ============================================================
        #   - self.logger = Logger()
        #   - self.logger.start()
        #   - self.logger.log_model_info(self.model)
        #
        # ============================================================
        # 3. 训练组件
        # ============================================================
        #   - self.optimizer = create_optimizer(self.model)
        #
        #   - self.scheduler = create_scheduler(
        #         self.optimizer,
        #         scheduler_type=TrainConfig.scheduler_type,
        #         total_steps=TrainConfig.total_training_steps,
        #     )
        #
        #   - self.criterion = nn.CrossEntropyLoss(
        #         ignore_index=TokenizerConfig.pad_id
        #     )
        #     注意: ignore_index 跳过 <pad> 位置的 loss
        #
        #   - self.early_stopping = EarlyStopping(
        #         patience=TrainConfig.early_stopping_patience,
        #         mode=TrainConfig.early_stopping_mode,
        #     )
        #
        # ============================================================
        # 4. AMP 混合精度 (仅 CUDA)
        # ============================================================
        #   self.use_amp = (
        #       TrainConfig.enable_mixed_precision
        #       and self.device.type == "cuda"
        #   )
        #   self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        #   注意: PyTorch >= 2.0 用 torch.amp.GradScaler("cuda")
        #        旧版本用 torch.cuda.amp.GradScaler()
        #
        # ============================================================
        # 5. 训练状态
        # ============================================================
        #   self.current_epoch = 0
        #   self.global_step = 0
        #   self.best_metric = float("inf")
        #
        #   # 训练历史记录
        #   self.history = {
        #       "train_loss": [],
        #       "val_loss": [],
        #       "val_accuracy": [],
        #   }
        #
        # ============================================================
        # 6. 恢复训练 (如果 resume_from 不为 None)
        # ============================================================
        #   if resume_from is not None:
        #       info = load_checkpoint(
        #           resume_from, self.model,
        #           optimizer=self.optimizer,
        #           scheduler=self.scheduler,
        #           early_stopping=self.early_stopping,
        #           scaler=self.scaler,
        #           device=str(self.device),
        #       )
        #       self.current_epoch = info["epoch"] + 1
        #       self.global_step = info["global_step"]
        #       self.best_metric = info.get("best_metric", float("inf"))
        #       self.logger.info(f"从检查点恢复: epoch={info['epoch']}, "
        #                        f"step={info['global_step']}")
        #
        # 注意:
        #   - 恢复训练时 scheduler 步数可能不对齐,
        #     简单方案: 循环 scheduler.step() global_step 次
        #   - optimizer 状态需要和 model 在同一设备,
        #     如果 checkpoint 在 CPU 加载但 model 在 GPU,
        #     optimizer 的动量缓存可能还在 CPU, 需注意处理
        # 未注入配置时创建独立默认实例,避免不同 Trainer 共享可变配置.
        self.config = config or TrainConfig()
        # 在模型和 DataLoader 工作前固定 Python、NumPy、CPU/CUDA 随机源.
        set_seed(self.config.random_seed)

        # 显式 device 参数优先于配置;get_device 会检查 CUDA 是否真实可用.
        self.device = get_device(device or self.config.device)
        # 先迁移模型再创建优化器,确保优化器引用的是目标设备上的参数对象.
        self.model = model.to(self.device)
        # train_loader 每个 batch 提供 source、target_input、target_output.
        self.train_loader = train_loader
        # val_loader 只用于无梯度验证,不参与参数更新.
        self.val_loader = val_loader
        # checkpoint_dir 统一转成 Path,保存函数内部会负责创建目录.
        self.checkpoint_dir = Path(checkpoint_dir)

        # 外部可注入测试 logger;生产环境默认创建文件、终端和 TensorBoard logger.
        self.logger = logger or Logger(
            name=self.config.logger_name,
            enable_tensorboard=self.config.enable_tensorboard,
        )
        # start 创建 handler;后续任何 info/debug 调用都要求 logger 已启动.
        self.logger.start()
        # asdict 把 dataclass 转成普通字典,便于日志逐项记录最终配置.
        self.logger.log_config(asdict(self.config))
        # 输出参数量和模型设备,便于训练开始前核对模型规模.
        self.logger.log_model_info(self.model)

        # 外部注入 optimizer 时保持原实例,否则根据 TrainConfig 创建.
        self.optimizer = optimizer or create_optimizer(
            self.model,
            optimizer_type=self.config.optimizer_type,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.optimizer_beta1, self.config.optimizer_beta2),
            eps=self.config.optimizer_eps,
            momentum=self.config.sgd_momentum,
        )
        # scheduler 可以独立注入;None 表示按配置创建默认调度器.
        self.scheduler = scheduler
        if self.scheduler is None:
            # 调度器持有 optimizer 引用,每次 step 修改各参数组学习率.
            self.scheduler = create_scheduler(
                self.optimizer,
                scheduler_type=self.config.scheduler_type,
                total_steps=self.config.total_training_steps,
                warmup_steps=self.config.scheduler_warmup_steps,
                min_lr_ratio=self.config.scheduler_min_learning_rate_ratio,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        # ignore_index 让 target_output_ids 中的 padding 不产生 loss 和梯度.
        self.criterion = criterion or nn.CrossEntropyLoss(
            ignore_index=TokenizerConfig.pad_id
        )
        # EarlyStopping 只维护指标状态,不导入 Trainer 或 checkpoint 模块.
        self.early_stopping = early_stopping or EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode=self.config.early_stopping_mode,
            overfitting_threshold=(self.config.early_stopping_overfitting_threshold),
            convergence_window=self.config.early_stopping_convergence_window,
            convergence_threshold=(self.config.early_stopping_convergence_threshold),
        )

        # PyTorch CUDA AMP 只在 CUDA 设备启用;CPU 自动保持 FP32.
        self.use_amp = self.config.enable_mixed_precision and self.device.type == "cuda"
        # GradScaler 防止 float16 梯度下溢;FP32 路径不需要该对象.
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        # current_epoch 使用零起始索引,日志展示时再加 1.
        self.current_epoch = 0
        # global_step 统计 optimizer 更新次数,调度器和验证间隔都依赖它.
        self.global_step = 0
        # loss 越小越好时从正无穷开始,max 模式则从负无穷开始.
        self.best_metric = (
            float("inf") if self.config.early_stopping_mode == "min" else -float("inf")
        )
        # history 每个 epoch 追加一个值,用于训练结束后的曲线绘制.
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        # 记录最近一次验证发生的 global_step,避免 epoch 末重复验证同一步.
        self._last_validation_step = -1
        # 最近一次验证 loss/accuracy 供 epoch 历史和摘要复用.
        self._last_val_loss = float("nan")
        self._last_val_accuracy = 0.0
        # 最近一次裁剪前梯度范数显示在 tqdm 和 TensorBoard 中.
        self._last_grad_norm = 0.0

        if resume_from is not None:
            # 一次恢复模型、优化器、调度器、早停和 AMP scaler 的完整状态.
            checkpoint_info = load_checkpoint(
                resume_from,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                scaler=self.scaler,
                device=str(self.device),
            )
            # checkpoint 保存的是已经完成的 epoch,恢复后从下一轮开始.
            self.current_epoch = checkpoint_info["epoch"] + 1
            # 继续原 global_step,保持学习率和验证间隔连续.
            self.global_step = checkpoint_info["global_step"]
            # 恢复历史最优指标,后续 best checkpoint 比较不会从头开始.
            self.best_metric = checkpoint_info["best_metric"]
            self.logger.info(
                f"从检查点恢复: epoch={checkpoint_info['epoch'] + 1}, "
                f"step={self.global_step}"
            )

    # =========================================================================
    # 主训练入口
    # =========================================================================

    def train(self) -> dict[str, list[float]]:
        """
        启动完整训练流程

        Returns:
            dict: 训练历史 {"train_loss": [...], "val_loss": [...], ...}

        提示:
            1. self.logger.info("开始训练")

            2. for epoch in range(self.current_epoch, TrainConfig.epoch_count):
                   self.current_epoch = epoch

                   # 训练
                   train_loss = self.train_epoch()

                   # 验证
                   val_loss, val_accuracy = self.validate_epoch()

                   # 记录 epoch 摘要
                   self.logger.log_epoch(
                       epoch, {"loss": train_loss},
                       {"loss": val_loss, "accuracy": val_accuracy}
                   )

                   # 早停更新
                   is_improved = self.early_stopping(
                       val_loss, train_loss=train_loss, epoch=epoch
                   )

                   # 保存检查点
                   save_checkpoint(
                       self.model, self.optimizer, epoch, self.global_step,
                       val_loss, is_best=is_improved,
                       scheduler=self.scheduler,
                       early_stopping=self.early_stopping,
                       scaler=self.scaler,
                   )

                   # 记录历史
                   self.history["train_loss"].append(train_loss)
                   self.history["val_loss"].append(val_loss)
                   self.history["val_accuracy"].append(val_accuracy)

                   # 早停判断
                   if self.early_stopping.should_stop:
                       self.logger.info(f"早停: {self.early_stopping.stop_reason}")
                       break

            3. 异常处理 (KeyboardInterrupt):
               try:
                   ...训练循环...
               except KeyboardInterrupt:
                   self.logger.warning("训练被手动中断")
                   save_checkpoint(...)  # 保存 last 检查点

            4. self.logger.info("训练完成")
            5. self.logger.close()
            6. return self.history
        """
        self.logger.info("开始训练")
        try:
            for epoch in range(self.current_epoch, self.config.epoch_count):
                if self.global_step >= self.config.total_training_steps:
                    self.logger.info("已达到最大训练步数")
                    break

                self.current_epoch = epoch
                train_loss = self.train_epoch()

                # 如果当前 step 未在 train_epoch 内触发间隔验证,则补一次 epoch 验证.
                if self._last_validation_step != self.global_step:
                    val_loss, val_accuracy = self._validate_and_checkpoint(train_loss)
                else:
                    # validation_interval 恰好落在 epoch 末时,直接复用已有结果.
                    val_loss = self._last_val_loss
                    val_accuracy = self._last_val_accuracy

                # history 始终每个 epoch 追加一次,便于 loss 曲线横轴保持一致.
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)
                self.logger.log_epoch(
                    epoch,
                    {"loss": train_loss},
                    {"loss": val_loss, "accuracy": val_accuracy},
                )

                if self.early_stopping.should_stop:
                    self.logger.info(f"早停: {self.early_stopping.stop_reason}")
                    break
        except KeyboardInterrupt:
            self.logger.warning("训练被手动中断,正在保存最后检查点")
            self._save_checkpoint(is_best=False)
        finally:
            self.logger.info("训练结束")
            self.logger.close()

        return self.history

    def _validate_and_checkpoint(self, train_loss: float) -> tuple[float, float]:
        """
        执行验证、更新早停状态并保存当前训练状态

        在以下时机调用:
            - epoch 结束时 (如果本 epoch 还未在该 global_step 做过验证)
            - train_epoch 内每隔 validation_interval 步

        Args:
            train_loss (float): 当前 epoch 的运行平均训练 loss

        Returns:
            tuple[float, float]: (验证 loss, token 级 accuracy)
        """
        # validate_epoch 返回样本平均 loss 和忽略 padding 的 token accuracy.
        val_loss, val_accuracy = self.validate_epoch()
        # step 级指标写入日志和 TensorBoard,不额外打印 epoch 摘要.
        self.logger.log_metrics(
            self.global_step,
            {"loss": val_loss, "accuracy": val_accuracy},
            prefix="val",
        )
        # EarlyStopping 以 val_loss 为主指标,并用 train_loss 检查过拟合差距.
        is_improved = self.early_stopping(
            val_loss,
            train_loss=train_loss,
            epoch=self.current_epoch,
        )
        # best_score 可能在第一次验证前为 None,验证后一定会被初始化.
        if self.early_stopping.best_score is not None:
            self.best_metric = self.early_stopping.best_score
        # 每次验证更新 last,只有指标改善时同时更新 best.
        self._save_checkpoint(is_best=is_improved)
        # 保存最近验证状态,供 epoch 末去重和训练历史记录使用.
        self._last_validation_step = self.global_step
        self._last_val_loss = val_loss
        self._last_val_accuracy = val_accuracy
        return val_loss, val_accuracy

    def _save_checkpoint(self, is_best: bool) -> None:
        """
        保存当前完整训练状态

        同时保存 last.pt (每次更新) 和 best.pt (指标改善时):
            - last.pt 用于恢复训练
            - best.pt 用于最终推理部署

        Args:
            is_best (bool): 当前指标是否为历史最优
        """
        model_config = getattr(self.model, "config", {})
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_metric=self.best_metric,
            is_best=is_best,
            scheduler=self.scheduler,
            early_stopping=self.early_stopping,
            scaler=self.scaler,
            config={"model": model_config, "train": asdict(self.config)},
            checkpoint_dir=self.checkpoint_dir,
        )

    # =========================================================================
    # 训练一个 epoch
    # =========================================================================

    def train_epoch(self) -> float:
        """
        遍历训练集一个 epoch

        Returns:
            float: 本 epoch 平均训练 loss

        提示:
            1. self.model.train()

            2. total_loss = 0.0, total_samples = 0

            3. for batch in self.train_loader:
                   batch = move_batch_to_device(batch, self.device)
                   source_ids = batch["source_ids"]
                   target_input_ids = batch["target_input_ids"]
                   target_output_ids = batch["target_output_ids"]

                   loss = self._train_step(
                       source_ids, target_input_ids, target_output_ids
                   )

                   累积 loss 和样本数

                   日志 (每 log_interval 步):
                   self.logger.log_metrics(
                       self.global_step,
                       {"loss": loss, "lr": current_lr},
                       prefix="train"
                   )

                   # 达到 total_steps 提前结束
                   if self.global_step >= TrainConfig.total_training_steps:
                       break

            4. return total_loss / max(total_samples, 1)

            5. DataLoader batch 格式:
               {"source_ids": (batch, src_len),
                "target_input_ids": (batch, tgt_len),
                "target_output_ids": (batch, tgt_len)}
        """
        # 启用 Dropout 等训练行为;此状态会影响模型前向传播的随机性.
        self.model.train()
        # loss 是 batch 均值,因此需要乘 batch_size 后再累计.
        total_loss = 0.0
        # 记录真实样本数,避免最后一个不完整 batch 影响 epoch 均值.
        total_samples = 0

        # 进度单位是 batch;disable 开关用于测试和日志重定向场景.
        progress_bar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"train epoch {self.current_epoch + 1}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not self.config.show_progress,
        )
        for batch in progress_bar:
            # 将字典中的三个二维 long Tensor 一次性迁移到训练设备.
            batch = move_batch_to_device(batch, self.device)
            # shape=(batch_size, source_length),作为 Encoder 输入.
            source_ids = batch["source_ids"]
            # shape=(batch_size, target_length),以 <bos> 开头.
            target_input_ids = batch["target_input_ids"]
            # shape=(batch_size, target_length),以 <eos> 结尾.
            target_output_ids = batch["target_output_ids"]
            # 最后一个 batch 可能小于配置值,所以读取张量真实第一维.
            batch_size = source_ids.size(0)

            # 单步完成前向、反向、梯度裁剪、参数更新和学习率调度.
            loss = self._train_step(
                source_ids,
                target_input_ids,
                target_output_ids,
            )
            # 按样本加权累积 loss.
            total_loss += loss * batch_size
            # 累加本轮实际处理的平行句对数量.
            total_samples += batch_size

            # postfix 在同一行展示关键动态指标,不会持续刷出新日志行.
            progress_bar.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                grad=f"{self._last_grad_norm:.3f}",
            )

            if self.global_step % self.config.log_interval == 0:
                self.logger.log_metrics(
                    self.global_step,
                    {
                        "loss": loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "gradient_norm": self._last_grad_norm,
                    },
                    prefix="train",
                )

            # 按 global_step 执行快速验证,间隔参数来自 CLI 或 menu 配置.
            if self.global_step % self.config.validation_interval == 0:
                running_train_loss = total_loss / max(total_samples, 1)
                self._validate_and_checkpoint(running_train_loss)
                # 早停可以在 epoch 中途触发,此时立即结束当前训练迭代.
                if self.early_stopping.should_stop:
                    break

            if self.global_step >= self.config.total_training_steps:
                break

        if total_samples == 0:
            raise ValueError("训练 DataLoader 没有可用样本")
        return total_loss / total_samples

    def _train_step(
        self,
        source_ids: Tensor,
        target_input_ids: Tensor,
        target_output_ids: Tensor,
    ) -> float:
        """
        单步训练: forward → loss → backward → clip → step

        这是训练循环的最小单元, 从 train_epoch 中抽离便于测试

        Args:
            source_ids (Tensor): (batch, source_length)
            target_input_ids (Tensor): (batch, target_length)
            target_output_ids (Tensor): (batch, target_length)

        Returns:
            float: 当前 batch 的标量 loss

        提示:
            === AMP 路径 (self.use_amp == True) ===
            1. with torch.amp.autocast("cuda"):
                   logits = self.model(source_ids, target_input_ids)
                   loss = self.criterion(
                       logits.transpose(1, 2), target_output_ids
                   )
               注意: logits 需要 transpose 为 (batch, vocab, seq_len)

            2. self.scaler.scale(loss).backward()
            3. self.scaler.unscale_(self.optimizer)
            4. grad_norm = clip_gradients(self.model)
            5. self.scaler.step(self.optimizer)
            6. self.scaler.update()

            === FP32 路径 (self.use_amp == False) ===
            1. self.optimizer.zero_grad()
            2. logits = self.model(source_ids, target_input_ids)
            3. loss = self.criterion(logits.transpose(1, 2), target_output_ids)
            4. loss.backward()
            5. grad_norm = clip_gradients(self.model)
            6. self.optimizer.step()

            === 公共步骤 ===
            7. if self.scheduler: self.scheduler.step()
            8. self.global_step += 1

            9. return loss.item()

        注意:
            - AMP 和 FP32 都必须在每一步开始时清空旧梯度
            - 梯度裁剪在 AMP 时必须放在 unscale_ 之后、step 之前
            - CrossEntropyLoss 需要 logits 的 class 维度在 dim=1
        """
        # set_to_none=True 减少显存写入,并确保本步不会累积上一步梯度.
        self.optimizer.zero_grad(set_to_none=True)
        # AMP 仅在 CUDA 且配置启用时生效;CPU 会自动沿用 FP32.
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # logits shape=(batch_size, target_length, vocab_size).
            logits = self.model(source_ids, target_input_ids)
            # CrossEntropyLoss 要求 class 位于 dim=1,因此转换成
            # (batch_size, vocab_size, target_length).
            loss = self.criterion(logits.transpose(1, 2), target_output_ids)

        # AMP 路径用 GradScaler 防止 float16 小梯度下溢.
        if self.scaler is not None:
            # 缩放 loss 后执行反向传播,暂时得到缩放后的参数梯度.
            self.scaler.scale(loss).backward()
            # 梯度裁剪前先恢复真实梯度尺度.
            self.scaler.unscale_(self.optimizer)
            # 返回裁剪前总范数,供日志和 tqdm 观察训练稳定性.
            self._last_grad_norm = clip_gradients(
                self.model, self.config.gradient_clip_norm
            )
            # 出现 inf/nan 时 scaler.step 会安全跳过本次更新.
            self.scaler.step(self.optimizer)
            # 根据当前数值状态调整下一步缩放倍数.
            self.scaler.update()
        else:
            # FP32 路径不需要缩放,直接构建参数梯度.
            loss.backward()
            # optimizer.step 前限制梯度总范数,减少梯度爆炸风险.
            self._last_grad_norm = clip_gradients(
                self.model, self.config.gradient_clip_norm
            )
            # 根据梯度更新所有 requires_grad=True 的参数.
            self.optimizer.step()

        # step 级调度器必须在 optimizer 更新之后推进.
        if self.scheduler is not None:
            self.scheduler.step()
        # global_step 表示已经完成的参数更新次数,会写入 checkpoint.
        self.global_step += 1
        # detach 后转成 Python float,防止训练历史持有计算图和显存.
        return float(loss.detach().item())

    # =========================================================================
    # 验证评估
    # =========================================================================

    def validate_epoch(self) -> tuple[float, float]:
        """
        在验证集上评估, 不更新参数

        Returns:
            tuple[float, float]: (平均验证 loss, token 级 accuracy)

        提示:
            1. self.model.eval()

            2. total_loss = 0.0, total_samples = 0
               total_correct = 0, total_tokens = 0

            3. with torch.no_grad():
                   for batch_idx, batch in enumerate(self.val_loader):
                       # 限制验证 batch 数, 避免验证时间过长
                       if batch_idx >= TrainConfig.validation_batch_count:
                           break

                       batch = move_batch_to_device(batch, self.device)
                       source_ids = batch["source_ids"]
                       target_input_ids = batch["target_input_ids"]
                       target_output_ids = batch["target_output_ids"]

                       # 前向传播
                       logits = self.model(source_ids, target_input_ids)

                       # loss
                       loss = self.criterion(
                           logits.transpose(1, 2), target_output_ids
                       )

                       # 累积 loss
                       total_loss += loss.item() * len(source_ids)
                       total_samples += len(source_ids)

                       # token 级 accuracy (可选)
                       predictions = logits.argmax(dim=-1)
                       mask = target_output_ids != TokenizerConfig.pad_id
                       total_correct += (
                           (predictions == target_output_ids) & mask
                       ).sum().item()
                       total_tokens += mask.sum().item()

            4. avg_loss = total_loss / max(total_samples, 1)
               accuracy = total_correct / max(total_tokens, 1)

            5. self.model.train()  # 恢复训练模式

            6. return avg_loss, accuracy

        注意:
            - torch.no_grad() 禁用梯度计算, 节省显存
            - 不调用 backward() 或 optimizer.step()
            - 验证后需切回 train() 模式 (影响 dropout / batch_norm 行为)
        """
        # 保存进入验证前的模式,finally 中按原状态恢复.
        was_training = self.model.training
        # eval 会关闭 Dropout,确保验证结果稳定.
        self.model.eval()
        # loss 按样本数量加权,accuracy 按有效 token 数量加权.
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_tokens = 0

        try:
            with torch.no_grad():
                # 进度条总数与实际验证上限保持一致.
                validation_total = min(
                    len(self.val_loader), self.config.validation_batch_count
                )
                progress_bar = tqdm(
                    self.val_loader,
                    total=validation_total,
                    desc=f"val epoch {self.current_epoch + 1}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                    disable=not self.config.show_progress,
                )
                for batch_index, batch in enumerate(progress_bar):
                    if batch_index >= self.config.validation_batch_count:
                        break

                    batch = move_batch_to_device(batch, self.device)
                    source_ids = batch["source_ids"]
                    target_input_ids = batch["target_input_ids"]
                    target_output_ids = batch["target_output_ids"]
                    batch_size = source_ids.size(0)

                    with torch.amp.autocast(
                        device_type=self.device.type, enabled=self.use_amp
                    ):
                        logits = self.model(source_ids, target_input_ids)
                        loss = self.criterion(logits.transpose(1, 2), target_output_ids)

                    total_loss += float(loss.item()) * batch_size
                    total_samples += batch_size
                    predictions = logits.argmax(dim=-1)
                    valid_mask = target_output_ids != TokenizerConfig.pad_id
                    total_correct += int(
                        ((predictions == target_output_ids) & valid_mask).sum().item()
                    )
                    total_tokens += int(valid_mask.sum().item())
                    # 展示截至当前 batch 的累计指标,减少单 batch 波动.
                    progress_bar.set_postfix(
                        loss=f"{total_loss / max(total_samples, 1):.4f}",
                        accuracy=f"{total_correct / max(total_tokens, 1):.4f}",
                    )
        finally:
            if was_training:
                self.model.train()

        if total_samples == 0:
            raise ValueError("验证 DataLoader 没有可用样本")
        average_loss = total_loss / total_samples
        accuracy = total_correct / max(total_tokens, 1)
        return average_loss, accuracy
