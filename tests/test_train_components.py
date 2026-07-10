"""训练组件和轻量训练闭环测试."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.defaults import TrainConfig
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.logger import Logger
from src.train.trainer import Trainer


class TinyTranslationModel(nn.Module):
    """用于测试训练流程的极小序列模型

    使用 Embedding + Linear 组成一个可训练的玩具模型,
    避免加载真实权重,确保测试轻量且可复现.
    """

    def __init__(self, vocab_size: int = 12) -> None:
        super().__init__()
        # 8 维词嵌入 + 线性投影到词表大小
        self.embedding = nn.Embedding(vocab_size, 8)
        self.projection = nn.Linear(8, vocab_size)
        self.config = {"vocab_size": vocab_size}

    def forward(
        self, source_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        # 取 source 首 token 作为上下文,与 target embedding 相加后投影
        source_context = self.embedding(source_ids[:, :1])
        return self.projection(self.embedding(target_ids) + source_context)


def create_tiny_dataloader() -> DataLoader:
    """创建无需外部文件的微型 DataLoader

    手动构造两个样本,每个样本包含:
      - source_ids: 源语言 token 序列
      - target_input_ids: 目标语言解码器输入序列
      - target_output_ids: 目标语言 ground-truth 输出序列
    """
    # 用两个简单样本构造 DataLoader,batch_size=2 一次喂入全部数据
    samples = [
        {
            "source_ids": torch.tensor([4, 3]),
            "target_input_ids": torch.tensor([2, 5]),
            "target_output_ids": torch.tensor([5, 3]),
        },
        {
            "source_ids": torch.tensor([6, 3]),
            "target_input_ids": torch.tensor([2, 7]),
            "target_output_ids": torch.tensor([7, 3]),
        },
    ]
    return DataLoader(samples, batch_size=2)


def test_early_stopping_state_round_trip_and_patience() -> None:
    """验证早停机制:patience 到期后应触发停止,且状态可序列化/反序列化往返"""
    # patience=2, convergence_threshold=0.0 表示损失连续上升 2 个 epoch 即停止
    early_stopping = EarlyStopping(
        patience=2,
        convergence_window=3,
        convergence_threshold=0.0,
    )
    # epoch 0 损失 1.0,为当前最优,应继续训练
    assert early_stopping(1.0, epoch=0)
    # epoch 1 损失 1.1,超过阈值,不满足收敛条件
    assert not early_stopping(1.1, epoch=1)
    # epoch 2 损失 1.2,patience 耗尽,触发停止
    assert not early_stopping(1.2, epoch=2)
    assert early_stopping.should_stop

    # 验证状态字典往返一致性
    restored = EarlyStopping(convergence_window=3)
    restored.load_state_dict(early_stopping.state_dict())
    assert restored.state_dict() == early_stopping.state_dict()


def test_checkpoint_restores_model_and_metadata(tmp_path) -> None:
    """验证保存 → 清空参数 → 加载后模型权重和元数据完全恢复

    测试流程:
      1. 保存 checkpoint(含 model、optimizer、元数据)
      2. 将模型参数全部置零模拟"丢失"
      3. 加载 checkpoint 恢复参数
      4. 断言权重逐元素一致、元数据字段正确
    """
    model = TinyTranslationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 备份原始权重用于后续比对
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=8,
        best_metric=0.75,
        is_best=True,
        checkpoint_dir=tmp_path,
    )
    # 将模型参数全部置零,模拟恢复前状态
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    metadata = load_checkpoint(tmp_path / "best_model.pth", model, optimizer=optimizer)

    # 验证元数据
    assert metadata["epoch"] == 2
    assert metadata["global_step"] == 8
    # 验证权重逐元素一致
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])


def test_logger_writes_file(tmp_path) -> None:
    """验证 Logger 启动后写入的日志内容会落盘到文件"""
    # 禁用 TensorBoard 以减少测试依赖
    logger = Logger(
        log_dir=tmp_path / "logs",
        tensorboard_dir=tmp_path / "tensorboard",
        enable_tensorboard=False,
    ).start()
    logger.info("训练日志测试")
    log_file = logger.log_file
    logger.close()

    # 断言日志文件存在且包含写入内容
    assert log_file is not None
    assert "训练日志测试" in log_file.read_text(encoding="utf-8")


def test_trainer_runs_and_saves_checkpoints(tmp_path) -> None:
    """端到端训练闭环:构造 Trainer → 跑 2 个 epoch → 验证损失记录和 checkpoint 生成"""
    dataloader = create_tiny_dataloader()
    # 使用最小化配置以加速测试:2 epoch、禁用 AMP/TensorBoard、CPU 运行
    config = TrainConfig.withOverrides(
        epoch_count=2,
        total_training_steps=2,
        validation_interval=1,
        scheduler_type="constant",
        enable_mixed_precision=False,
        enable_tensorboard=False,
        log_interval=1,
        early_stopping_patience=10,
        early_stopping_convergence_threshold=0.0,
        show_progress=False,
    )
    logger = Logger(
        log_dir=tmp_path / "logs",
        tensorboard_dir=tmp_path / "tensorboard",
        enable_tensorboard=False,
    )
    trainer = Trainer(
        model=TinyTranslationModel(),
        train_loader=dataloader,
        val_loader=dataloader,
        config=config,
        device="cpu",
        logger=logger,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    history = trainer.train()

    # 2 个 epoch 应记录 2 条训练损失
    assert len(history["train_loss"]) == 2
    # 应生成 last_model.pth 和 best_model.pth 两个检查点
    assert (tmp_path / "checkpoints" / "last_model.pth").is_file()
    assert (tmp_path / "checkpoints" / "best_model.pth").is_file()


def test_checkpoint_missing_file_is_explicit(tmp_path) -> None:
    """加载不存在的 checkpoint 文件时应显式抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pth", TinyTranslationModel())
