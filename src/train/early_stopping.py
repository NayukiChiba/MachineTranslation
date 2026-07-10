"""
早停控制器模块

功能:
1. EarlyStopping — 监控验证集指标, 无提升时提前终止训练

早停逻辑:
    - 每次验证后调用实例(通过 __call__), 传入当前验证指标
    - 如果指标提升超过 min_delta, 重置计数器
    - 连续 patience 次无提升则触发早停
    - 支持 "min"(loss) / "max"(accuracy) 两种优化方向

附加检测:
    - 过拟合检测: train/val 差距超过阈值时触发
    - 收敛检测: 验证指标波动过小时触发

说明:
    - 不依赖 train/ 下其他模块
    - 所有默认值从 configs.defaults.TrainConfig 读取
    - trainer.py 在每次验证后调用, 根据 should_stop 决定是否退出

使用方式:
    early_stopper = EarlyStopping(patience=10, mode="min", min_delta=1e-4)
    for epoch in range(epochs):
        train_loss = train_epoch()
        val_loss = validate_epoch()
        is_improved = early_stopper(val_loss, train_loss=train_loss)
        if early_stopper.should_stop:
            print(f"早停: {early_stopper.stop_reason}")
            break
"""

from typing import Optional

from configs.defaults import TrainConfig


class EarlyStopping:
    """
    早停控制器

    支持三种停止条件(按检查优先级):
        1. 过拟合检测 — train/val 差距过大
        2. 收敛检测  — 验证指标波动极小
        3. 无改善计数 — patience 次验证无提升

    Args:
        patience (int): 容忍无提升次数
        min_delta (float): 最小改善阈值, 避免噪声误判
        mode (str): "min"(越小越好) 或 "max"(越大越好)
        overfitting_threshold (float | None): 过拟合阈值, None 关闭
        convergence_window (int): 收敛检测窗口大小
        convergence_threshold (float): 收敛标准差阈值

    使用示例:
        >>> stopper = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     if stopper(val_loss):
        ...         pass  # 触发早停
    """

    def __init__(
        self,
        patience: int = TrainConfig.early_stopping_patience,
        min_delta: float = TrainConfig.early_stopping_min_delta,
        mode: str = TrainConfig.early_stopping_mode,
        overfitting_threshold: Optional[float] = (
            TrainConfig.early_stopping_overfitting_threshold
        ),
        convergence_window: int = TrainConfig.early_stopping_convergence_window,
        convergence_threshold: float = (
            TrainConfig.early_stopping_convergence_threshold
        ),
    ) -> None:
        # 需要初始化的属性:
        #
        # 基本配置:
        #   self.patience = patience
        #   self.min_delta = min_delta
        #   self.mode = mode
        #   self.overfitting_threshold = overfitting_threshold
        #   self.convergence_window = convergence_window
        #   self.convergence_threshold = convergence_threshold
        #
        # 状态变量:
        #   self.counter = 0          — 无改善计数器
        #   self.best_score = None    — 历史最佳指标
        #   self.best_epoch = 0       — 最佳 epoch
        #   self.should_stop = False  — 早停信号
        #   self.stop_reason = ""     — 停止原因描述
        #   self.val_history = []     — 验证指标历史(收敛检测用)
        #
        # 注意:
        #   - best_score 初始化为 None, 第一次调用 __call__ 时自动赋值
        #   - mode 影响比较方向, 见 _is_improved 方法的提示
        if patience <= 0:
            raise ValueError("patience 必须大于 0")
        if min_delta < 0:
            raise ValueError("min_delta 不能小于 0")
        if mode not in {"min", "max"}:
            raise ValueError("mode 必须是 'min' 或 'max'")
        if convergence_window <= 1:
            raise ValueError("convergence_window 必须大于 1")
        if convergence_threshold < 0:
            raise ValueError("convergence_threshold 不能小于 0")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.overfitting_threshold = overfitting_threshold
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.reset()

    def __call__(
        self,
        val_loss: float,
        train_loss: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        更新早停状态并判断是否需要停止

        Args:
            val_loss (float): 当前验证集指标
            train_loss (float | None): 训练集指标, 过拟合检测用, 可为 None
            epoch (int | None): 当前 epoch, 记录最佳 epoch 用

        Returns:
            bool: True 表示本轮有改善, False 表示无改善

        提示:
            1. 记录到 val_history
            2. 首次调用: best_score = val_loss, return True
            3. 判断改善: _is_improved(val_loss)
               - 改善 → 更新 best_score, counter=0, return True
               - 未改善 → counter += 1, return False
            4. 调用 _check_early_stop 更新 should_stop 和 stop_reason
            5. 检测优先级: 过拟合 > 收敛 > patience
        """
        # 步骤:
        #   1. self.val_history.append(val_loss)
        #
        #   2. if self.best_score is None:
        #        self.best_score = val_loss
        #        if epoch is not None: self.best_epoch = epoch
        #        return True
        #
        #   3. is_improved = self._is_improved(val_loss)
        #
        #   4. if is_improved:
        #        self.best_score = val_loss
        #        self.counter = 0
        #        if epoch is not None: self.best_epoch = epoch
        #      else:
        #        self.counter += 1
        #
        #   5. self._check_early_stop(val_loss, train_loss)
        #
        #   6. return is_improved
        self.val_history.append(val_loss)

        if self.best_score is None:
            self.best_score = val_loss
            if epoch is not None:
                self.best_epoch = epoch
            return True

        is_improved = self._is_improved(val_loss)
        if is_improved:
            self.best_score = val_loss
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
        else:
            self.counter += 1

        self._check_early_stop(val_loss, train_loss)
        return is_improved

    def _is_improved(self, score: float) -> bool:
        """
        判断是否有实质改善

        Args:
            score (float): 当前指标

        Returns:
            bool: 是否改善

        提示:
            - mode == "min": score < best_score - min_delta
            - mode == "max": score > best_score + min_delta
            - min_delta 避免数值噪声被误判为提升
        """
        # 步骤:
        #   1. if self.mode == "min":
        #        return score < self.best_score - self.min_delta
        #      else:
        #        return score > self.best_score + self.min_delta
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def _check_early_stop(self, val_loss: float, train_loss: Optional[float]) -> None:
        """
        按优先级检查是否触发早停

        检查顺序: 过拟合 → 收敛 → 无改善计数
        一旦触发则设置 self.should_stop = True 和 self.stop_reason

        Args:
            val_loss (float): 验证集指标
            train_loss (float | None): 训练集指标
        """
        # 步骤:
        #   1. 过拟合检测:
        #      if self.overfitting_threshold is not None and train_loss is not None:
        #          gap = val_loss - train_loss  (mode=="min"时)
        #          if gap > self.overfitting_threshold:
        #              self.should_stop = True
        #              self.stop_reason = f"过拟合: gap={gap:.4f}"
        #              return
        #
        #   2. 收敛检测:
        #      if len(self.val_history) >= self.convergence_window:
        #          recent = self.val_history[-self.convergence_window:]
        #          计算标准差: std = (sum((x-mean)^2) / n)^0.5
        #          if std < self.convergence_threshold:
        #              self.should_stop = True
        #              self.stop_reason = f"收敛: std={std:.6f}"
        #              return
        #
        #   3. 无改善计数:
        #      if self.counter >= self.patience:
        #          self.should_stop = True
        #          self.stop_reason = f"连续{self.patience}轮无改善"
        #
        # 注意:
        #   - 收敛检测的标准差直接用 Python 计算即可, 无需 numpy
        #   - mean = sum(recent) / len(recent)
        #   - variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        #   - std = variance ** 0.5
        if self.overfitting_threshold is not None and train_loss is not None:
            gap = val_loss - train_loss if self.mode == "min" else train_loss - val_loss
            if gap > self.overfitting_threshold:
                self.should_stop = True
                self.stop_reason = f"过拟合: gap={gap:.4f}"
                return

        if len(self.val_history) >= self.convergence_window:
            recent_values = self.val_history[-self.convergence_window :]
            mean = sum(recent_values) / len(recent_values)
            variance = sum((value - mean) ** 2 for value in recent_values) / len(
                recent_values
            )
            standard_deviation = variance**0.5
            if standard_deviation < self.convergence_threshold:
                self.should_stop = True
                self.stop_reason = f"收敛: std={standard_deviation:.6f}"
                return

        if self.counter >= self.patience:
            self.should_stop = True
            self.stop_reason = f"连续{self.patience}轮无改善"

    def reset(self) -> None:
        """重置早停状态, 用于重新训练"""
        # 步骤:
        #   1. self.counter = 0
        #   2. self.best_score = None
        #   3. self.best_epoch = 0
        #   4. self.should_stop = False
        #   5. self.stop_reason = ""
        #   6. self.val_history = []
        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch = 0
        self.should_stop = False
        self.stop_reason = ""
        self.val_history: list[float] = []

    def state_dict(self) -> dict:
        """
        导出早停状态, 用于 checkpoint 保存

        Returns:
            dict: 包含所有关键状态的字典
        """
        # 步骤:
        #   1. return {
        #        "counter": self.counter,
        #        "best_score": self.best_score,
        #        "best_epoch": self.best_epoch,
        #        "should_stop": self.should_stop,
        #        "stop_reason": self.stop_reason,
        #        "val_history": self.val_history,
        #      }
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
            "val_history": list(self.val_history),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        从 checkpoint 恢复早停状态

        Args:
            state (dict): state_dict() 导出的字典
        """
        # 步骤:
        #   1. self.counter = state["counter"]
        #   2. self.best_score = state["best_score"]
        #   3. self.best_epoch = state["best_epoch"]
        #   4. self.should_stop = state["should_stop"]
        #   5. self.stop_reason = state["stop_reason"]
        #   6. self.val_history = state["val_history"]
        required_keys = {
            "counter",
            "best_score",
            "best_epoch",
            "should_stop",
            "stop_reason",
            "val_history",
        }
        missing_keys = required_keys.difference(state)
        if missing_keys:
            raise KeyError(f"早停状态缺少字段: {sorted(missing_keys)}")

        self.counter = int(state["counter"])
        self.best_score = state["best_score"]
        self.best_epoch = int(state["best_epoch"])
        self.should_stop = bool(state["should_stop"])
        self.stop_reason = str(state["stop_reason"])
        self.val_history = list(state["val_history"])
