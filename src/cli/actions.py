"""
CLI 与交互菜单共用的任务管线

这里负责把已经构建好的配置对象转换为 DataLoader、模型、Trainer、Evaluator
和 Translator.参数解析留在 parser.py,交互输入留在 menu.py,CLI 主任务分发
留在 main.py,因此业务逻辑只保留一份.
"""

from pathlib import Path

import torch

from configs import paths
from configs.defaults import (
    DataLoaderConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
)
from src.data.dataloader import create_dataloaders, prepare_data_pipeline
from src.data.tokenizer import SentencePieceTokenizer
from src.evaluate.evaluator import Evaluator
from src.evaluate.visualize import save_training_curves
from src.inference.translator import load_translator
from src.model.transformer import Transformer
from src.train.trainer import Trainer


def prepare_data(force: bool = False) -> None:
    """执行 raw -> interim -> tokenizer -> processed 的完整数据管线."""
    # force=True 时各阶段忽略已有产物,适合配置变更后重新生成数据.
    prepare_data_pipeline(force=force)


def train_model(
    model_config: type[ModelConfig] = ModelConfig,
    train_config: type[TrainConfig] = TrainConfig,
    loader_config: type[DataLoaderConfig] = DataLoaderConfig,
    resume_from: Path | None = None,
) -> dict[str, list[float]]:
    """根据完整配置创建依赖并执行训练."""
    # 所有配置均为静态类,业务层只读取类属性,不创建配置实例.

    # DataLoader 输出字典中的每个 Tensor 均为二维 long Tensor:
    # source_ids=(batch, source_length),target_*=(batch, target_length).
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=loader_config.batch_size,
        num_workers=loader_config.worker_count,
        auto_prepare=loader_config.auto_prepare,
        force_prepare=loader_config.force_prepare,
    )
    # 第三个返回值为 test_loader,训练阶段不需要,用 _ 丢弃.

    # SentencePiece 文件由 paths 统一管理;实际词表大小只能在模型加载后确定.
    tokenizer = SentencePieceTokenizer(paths.TOKENIZER_MODEL_PATH)

    if resume_from is not None:
        # 恢复训练时必须使用 checkpoint 中保存的模型结构,否则参数形状可能不匹配.
        model = load_model_from_checkpoint(resume_from, train_config.device)
        # 后续 best/last checkpoint 与原 checkpoint 放在同一目录,便于实验隔离.
        checkpoint_dir = Path(resume_from).parent
    else:
        # 新实验使用当前 ModelConfig,并用真实 tokenizer 覆盖词表与 padding id.
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_feedforward=model_config.d_feedforward,
            encoder_num_layers=model_config.encoder_num_layers,
            decoder_num_layers=model_config.decoder_num_layers,
            dropout=model_config.dropout,
            max_seq_length=model_config.max_seq_length,
            pad_id=tokenizer.pad_id,
            norm_first=model_config.norm_first,
        )
        # 新实验始终写入 configs.paths 声明的统一 checkpoints 目录.
        checkpoint_dir = paths.CHECKPOINTS_DIR

    # Trainer 是 train 包内唯一调度者,负责组合互不依赖的训练组件.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=train_config.device,
        resume_from=resume_from,
        config=train_config,
        checkpoint_dir=checkpoint_dir,
    )
    # history 中每个列表按 epoch 记录 train_loss、val_loss 和 val_accuracy.
    history = trainer.train()
    # 训练结束后将 loss 曲线写入统一 figures 目录,便于比较实验结果.
    figure_path = save_training_curves(history)
    print(f"训练曲线: {figure_path}")
    # 返回历史指标,测试和 notebook 可以直接复用而不解析日志文本.
    return history


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device | str
) -> Transformer:
    """按 checkpoint 内的模型配置重建 Transformer 并加载权重."""
    # 接受字符串或 Path,并统一转换,避免调用方手工处理平台路径分隔符.
    checkpoint_path = Path(checkpoint_path)
    # 文件缺失时立即报错,避免后续出现含义不清晰的 torch.load 异常.
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    # map_location 使 GPU 保存的 checkpoint 可以在 CPU 环境中评估或翻译.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    # 新版 checkpoint 的 config.model 保存 Transformer 构造参数.
    model_config = checkpoint.get("config", {}).get("model", {})
    if not model_config:
        # 兼容没有模型配置的旧 checkpoint:至少从 tokenizer 恢复词表信息.
        tokenizer = SentencePieceTokenizer(paths.TOKENIZER_MODEL_PATH)
        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        }

    # 先按相同结构创建空模型,再加载 state_dict 中的实际参数张量.
    model = Transformer.from_config(model_config)
    # load_state_dict 会严格校验参数名和形状,不匹配时立即报错.
    model.load_state_dict(checkpoint["model_state_dict"])
    # 调用方仍可决定 train/eval 模式和最终设备,因此这里只返回模型对象.
    return model


def evaluate_model(
    checkpoint_path: Path = paths.BEST_MODEL_PATH,
    split: str = "test",
    loader_config: type[DataLoaderConfig] = DataLoaderConfig,
    device: torch.device | str = TrainConfig.device,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    """加载指定数据切分和 checkpoint,并计算结构化评估指标."""
    # split 在 parser 中已经限制取值;此处保留运行时校验供 Python API 使用.
    if split not in {"train", "val", "test"}:
        raise ValueError("split 必须是 train、val 或 test")
    # DataLoader 配置通过静态类传入,默认直接使用全局默认配置类.
    # 一次创建三个 DataLoader,再通过映射选择目标切分,保持数据参数一致.
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=loader_config.batch_size,
        num_workers=loader_config.worker_count,
        auto_prepare=loader_config.auto_prepare,
        force_prepare=loader_config.force_prepare,
    )
    loader_by_split = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    # checkpoint 中的模型结构和权重必须成对恢复.
    model = load_model_from_checkpoint(checkpoint_path, device)
    # Evaluator 不依赖 Trainer,仅接收模型、设备和待遍历的数据.
    evaluator = Evaluator(model=model, device=device)
    metrics = evaluator.evaluate(
        loader_by_split[split],
        max_batches=max_batches,
        show_progress=show_progress,
    )
    # 终端输出保留四位小数,返回值仍保留完整浮点精度.
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    return metrics


def translate_text(
    text: str,
    checkpoint_path: Path = paths.BEST_MODEL_PATH,
    device: torch.device | str = TrainConfig.device,
    inference_config: type[InferenceConfig] = InferenceConfig,
) -> str:
    """加载模型和 tokenizer,并执行一条英文到中文的贪心翻译."""
    # 推理配置控制最大目标长度和 token 级 tqdm 是否显示.
    # load_translator 负责恢复与 checkpoint 配套的 Transformer 权重.
    translator = load_translator(
        checkpoint_path=checkpoint_path,
        tokenizer_path=paths.TOKENIZER_MODEL_PATH,
        device=device,
        max_generation_length=inference_config.max_generation_length,
        show_progress=inference_config.show_progress,
    )
    # Translator 内部先编码 source,再逐 token 自回归生成 target.
    translation = translator.translate(text)
    # CLI 和 menu 都需要立即展示结果,因此在共享管线中统一输出.
    print(translation)
    # 返回文本便于测试、批处理脚本或 notebook 调用.
    return translation
