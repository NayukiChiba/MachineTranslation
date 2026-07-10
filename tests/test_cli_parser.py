"""
CLI 参数覆盖和配置构建测试.

测试范围:
1. 默认参数是否正确继承 configs/defaults.py 中的配置常量
2. 命令行传入的参数是否正确映射到对应的 Config 对象字段
3. eval 和 translate 子命令的特殊参数处理
4. 非法参数组合是否在构建配置阶段正确报错
"""

import pytest

from configs.defaults import DataLoaderConfig, InferenceConfig, ModelConfig, TrainConfig
from main import (
    build_inference_config,
    build_loader_config,
    build_model_config,
    build_train_config,
)
from src.cli.parser import create_parser


def test_train_defaults_match_config_objects() -> None:
    """train 不传选项时应完整复用 defaults.py."""
    # 仅传入子命令名,不传入任何可选参数
    args = create_parser().parse_args(["train"])

    # 验证解析后的参数值与配置对象的默认值完全一致
    assert args.d_model == ModelConfig.d_model
    assert args.batch_size == DataLoaderConfig.batch_size
    assert args.epochs == TrainConfig.epoch_count
    assert args.optimizer == TrainConfig.optimizer_type
    assert args.scheduler == TrainConfig.scheduler_type


def test_train_options_build_complete_configs() -> None:
    """模型、数据、优化器和调度器参数应映射到正确字段."""
    # 传入完整的训练参数组合,覆盖模型、数据加载器、训练器三类配置
    args = create_parser().parse_args(
        [
            "train",
            "--d-model",
            "64",
            "--num-heads",
            "4",
            "--d-feedforward",
            "128",
            "--encoder-layers",
            "2",
            "--decoder-layers",
            "3",
            "--batch-size",
            "8",
            "--num-workers",
            "1",
            "--epochs",
            "5",
            "--total-steps",
            "1000",
            "--optimizer",
            "adam",
            "--lr",
            "0.001",
            "--scheduler",
            "cosine_warmup",
            "--warmup-steps",
            "100",
            "--patience",
            "7",
            "--no-mixed-precision",
            "--no-tensorboard",
            "--no-progress",
        ]
    )

    # 将解析结果分别构建为三类配置对象
    model_config = build_model_config(args)
    loader_config = build_loader_config(args)
    train_config = build_train_config(args)

    # 验证模型配置字段
    assert model_config.d_model == 64
    assert model_config.decoder_num_layers == 3
    # 验证数据加载器配置字段
    assert loader_config.batch_size == 8
    assert loader_config.worker_count == 1
    # 验证训练器配置字段
    assert train_config.epoch_count == 5
    assert train_config.optimizer_type == "adam"
    assert train_config.learning_rate == pytest.approx(0.001)
    assert train_config.scheduler_warmup_steps == 100
    assert train_config.early_stopping_patience == 7
    # 验证布尔开关参数(--no-xxx 应映射为 False)
    assert not train_config.enable_mixed_precision
    assert not train_config.enable_tensorboard
    assert not train_config.show_progress


def test_eval_supports_split_and_progress_override() -> None:
    """eval 应支持选择数据切分、批次上限和关闭 tqdm."""
    # 构造 eval 子命令的完整参数
    args = create_parser().parse_args(
        [
            "eval",
            "--checkpoint",
            "best.pth",
            "--split",
            "val",
            "--batch-size",
            "16",
            "--max-batches",
            "3",
            "--no-progress",
        ]
    )

    # 验证子命令识别
    assert args.command == "eval"
    # 验证检查点路径和切分参数
    assert args.checkpoint == "best.pth"
    assert args.split == "val"
    # 验证评估特有的批次上限和批量大小
    assert args.batch_size == 16
    assert args.max_batches == 3
    # --no-progress 应关闭进度条
    assert not args.progress


def test_translate_supports_single_and_interactive_modes() -> None:
    """translate 文本可选;省略时由 translate_main 进入交互循环."""
    parser = create_parser()

    # 单次翻译模式:传入待翻译文本和最大生成长度
    single_args = parser.parse_args(["translate", "hello", "--max-length", "20"])
    # 交互模式:不传入文本,仅关闭进度条
    interactive_args = parser.parse_args(["translate", "--no-progress"])

    # 单次模式下 text 非空,推理配置使用指定的生成长度上限
    assert single_args.text == "hello"
    assert build_inference_config(single_args) == InferenceConfig(
        max_generation_length=20,
        show_progress=True,
    )
    # 交互模式下 text 为 None,由上层 translate_main 进入交互循环
    assert interactive_args.text is None
    assert not interactive_args.progress


def test_invalid_model_dimensions_fail_when_building_config() -> None:
    """d_model 不能整除 num_heads 时应在加载数据前失败."""
    # 构造非法参数组合:d_model=63 不能被 num_heads=8 整除
    args = create_parser().parse_args(["train", "--d-model", "63", "--num-heads", "8"])

    # 期望在构建配置阶段就抛出 ValueError,而非在后续阶段静默失败
    with pytest.raises(ValueError, match="整除"):
        build_model_config(args)
