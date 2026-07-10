"""
MachineTranslation 唯一程序入口

用法:
    python main.py train [options]
    python main.py eval [options]
    python main.py translate [text] [options]
    python main.py prepare [--force]
    python main.py                       # 无参数时进入交互菜单

入口结构参考相邻 Attention 项目:parser.py 只声明参数,main.py 提供各任务的
主入口并负责分发,menu.py 负责无参数时的交互式工作流.
"""

# 标准库
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# 项目配置:所有超参数和路径的统一入口
from configs.defaults import (
    DataLoaderConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
)

# CLI 动作层:封装训练、评估、翻译、数据准备的完整流程
from src.cli.actions import evaluate_model, prepare_data, train_model, translate_text

# CLI 交互层:菜单模式与命令行参数解析
from src.cli.menu import show_menu
from src.cli.parser import create_parser

# Windows 终端的默认编码可能不是 UTF-8;显式调整后中文日志不会出现乱码.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
# stderr 同样会输出 argparse 错误和 Python 异常,因此需要保持相同编码.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def configureModel(args: argparse.Namespace) -> type[ModelConfig]:
    """根据 train 子命令返回经过校验的静态 ModelConfig 类."""
    # 每一个字段都显式映射,避免 argparse 名称变化时静默漏掉配置.
    return ModelConfig.withOverrides(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_feedforward=args.d_feedforward,
        encoder_num_layers=args.encoder_layers,
        decoder_num_layers=args.decoder_layers,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        norm_first=args.norm_first,
    )


def configureLoader(args: argparse.Namespace) -> type[DataLoaderConfig]:
    """根据当前子命令返回静态 DataLoaderConfig 类."""
    # train、eval 都有 batch_size 和 num_workers,其他入口使用默认值.
    batch_size = getattr(args, "batch_size", DataLoaderConfig.batch_size)
    # Windows 默认 worker_count=0,用户可以在 CLI 中按机器能力覆盖.
    worker_count = getattr(args, "num_workers", DataLoaderConfig.worker_count)
    # 训练入口支持强制重建数据;评估入口默认只补齐缺失产物.
    force_prepare = getattr(args, "force_prepare", False)
    return DataLoaderConfig.withOverrides(
        batch_size=batch_size,
        worker_count=worker_count,
        auto_prepare=True,
        force_prepare=force_prepare,
    )


def configureTrain(args: argparse.Namespace) -> type[TrainConfig]:
    """根据 CLI 参数返回经过校验的静态 TrainConfig 类."""
    return TrainConfig.withOverrides(
        epoch_count=args.epochs,
        total_training_steps=args.total_steps,
        validation_interval=args.validation_interval,
        validation_batch_count=args.validation_batches,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer_beta1=args.beta1,
        optimizer_beta2=args.beta2,
        optimizer_eps=args.optimizer_eps,
        sgd_momentum=args.momentum,
        scheduler_type=args.scheduler,
        scheduler_warmup_steps=args.warmup_steps,
        scheduler_min_learning_rate_ratio=args.min_lr_ratio,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        early_stopping_overfitting_threshold=args.overfitting_threshold,
        early_stopping_convergence_window=args.convergence_window,
        early_stopping_convergence_threshold=args.convergence_threshold,
        gradient_clip_norm=args.grad_clip,
        enable_mixed_precision=args.mixed_precision,
        log_interval=args.log_interval,
        device=args.device,
        random_seed=args.seed,
        enable_tensorboard=args.tensorboard,
        show_progress=args.progress,
    )


def configureInference(args: argparse.Namespace) -> type[InferenceConfig]:
    """根据 translate 子命令返回静态 InferenceConfig 类."""
    # 推理阶段只关心生成长度和进度显示,模型结构参数从 checkpoint 自动恢复.
    return InferenceConfig.withOverrides(
        max_generation_length=args.max_length,
        show_progress=args.progress,
    )


def prepare_main(args: argparse.Namespace) -> None:
    """数据准备主入口."""
    # prepare_data 会按 raw、interim、tokenizer、processed 的顺序执行.
    # force=True 时跳过存在性检查,强制重新执行所有步骤.
    prepare_data(force=args.force)


def train_main(args: argparse.Namespace) -> None:
    """训练主入口:构建配置、展示配置、创建训练管线并开始训练."""
    # 先构建并校验模型配置,非法维度会在读取大数据前快速失败.
    model_config = configureModel(args)
    # DataLoader 配置决定 batch 内张量第一维和数据读取并发数.
    loader_config = configureLoader(args)
    # 训练配置覆盖 optimizer、scheduler、early stopping、AMP 和日志行为.
    train_config = configureTrain(args)

    # 明确展示最终生效值,方便把终端输出直接作为实验记录.
    print("=" * 60)
    print("模型配置")
    model_config.printSummary()
    print("\n数据加载配置")
    loader_config.printSummary()
    print("\n训练配置")
    train_config.printSummary()
    print("=" * 60)

    # argparse 返回字符串路径;业务层统一使用 pathlib.Path.
    resume_from = Path(args.resume) if args.resume else None
    # 共享 action 创建 tokenizer、DataLoader、Transformer 和 Trainer.
    train_model(
        model_config=model_config,
        train_config=train_config,
        loader_config=loader_config,
        resume_from=resume_from,
    )


def eval_main(args: argparse.Namespace) -> None:
    """评估主入口:选择数据切分并计算 loss、困惑度和 token accuracy."""
    # eval 只需要 DataLoader 参数,不会修改 checkpoint 中保存的模型结构.
    loader_config = configureLoader(args)
    print("=" * 60)
    print(f"评估数据集: {args.split}")
    print(f"检查点: {args.checkpoint}")
    loader_config.printSummary()
    print("=" * 60)

    # show_progress 控制评估 batch 级 tqdm,便于 CI 或日志文件关闭动态输出.
    evaluate_model(
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        loader_config=loader_config,
        device=args.device,
        max_batches=args.max_batches,
        show_progress=args.progress,
    )


def translate_main(args: argparse.Namespace) -> None:
    """翻译主入口:支持命令行单句和持续交互两种模式."""
    # 推理配置控制目标序列最大长度和 token 级进度条.
    inference_config = configureInference(args)
    # checkpoint 结构中保存模型参数,tokenizer 路径则来自 configs.paths.
    checkpoint_path = Path(args.checkpoint)

    if args.text:
        # 指定 text 时只翻译一次,适合 shell 脚本和批处理调用.
        translate_text(
            text=args.text,
            checkpoint_path=checkpoint_path,
            device=args.device,
            inference_config=inference_config,
        )
        return

    # 未指定 text 时进入循环,避免每翻译一句都重新加载 Python 进程.
    print("进入交互翻译模式,输入 quit、exit 或 q 退出")
    while True:
        # strip 去掉命令行输入两端空白,空行不会送入 tokenizer.
        source_text = input("\n请输入英文文本: ").strip()
        if source_text.lower() in {"quit", "exit", "q"}:
            print("退出翻译模式")
            return
        if not source_text:
            print("输入文本不能为空")
            continue
        translate_text(
            text=source_text,
            checkpoint_path=checkpoint_path,
            device=args.device,
            inference_config=inference_config,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """唯一分发函数:无参数进入 menu,有参数进入对应 CLI 主入口.

    返回值为 0 表示正常结束,非 0 表示异常退出,
    便于 shell 脚本和 CI 系统判断执行结果.
    """
    # 测试可以显式传入 argv;真实启动时读取 sys.argv[1:].
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        # 与 Attention 项目一致,直接运行 main.py 时展示交互菜单.
        show_menu()
        return 0

    # CLI 模式先由 parser 校验类型和 choices,再由下面的分支执行任务.
    parser = create_parser()
    args = parser.parse_args(arguments)

    # 根据子命令分发到对应的主入口函数.
    if args.command == "prepare":
        prepare_main(args)  # 数据下载与预处理
    elif args.command == "train":
        train_main(args)  # 模型训练
    elif args.command in {"eval", "evaluate"}:
        eval_main(args)  # 模型评估(支持 eval / evaluate 两个别名)
    elif args.command == "translate":
        translate_main(args)  # 单句或交互式翻译
    else:
        # 有参数但没有子命令时显示帮助,而不是静默结束.
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    # 将返回码交给操作系统,便于 shell 和 CI 判断执行是否成功.
    raise SystemExit(main())
