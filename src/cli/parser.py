"""
命令行参数解析器

本模块只负责声明 CLI 的语法和默认值,不创建模型、不读取数据,也不启动训练.
真正的 train / eval / translate 主入口位于项目根目录 main.py,职责划分与
相邻 Attention 项目保持一致.

用法:
    python main.py train [options]
    python main.py eval [options]
    python main.py translate [text] [options]
    python main.py prepare [--force]
    python main.py
"""

# 标准库:命令行参数解析
import argparse

# 项目配置:路径常量与各模块默认值
from configs import paths
from configs.defaults import (
    DataLoaderConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
)


def create_parser() -> argparse.ArgumentParser:
    """创建包含 train、eval、translate 和 prepare 的主解析器."""
    # 每次创建解析器时都实例化独立默认配置,避免 CLI 修改类级默认值.
    model_defaults = ModelConfig()
    # DataLoader 参数单独归类,避免把数据并发设置混入模型配置.
    loader_defaults = DataLoaderConfig()
    # 训练默认值覆盖训练循环、优化器、调度器、早停和日志.
    train_defaults = TrainConfig()
    # 推理默认值只负责控制自回归生成过程.
    inference_defaults = InferenceConfig()

    # RawDescriptionHelpFormatter 会保留帮助文本中的换行和分组结构.
    parser = argparse.ArgumentParser(
        description="Encoder-Decoder Transformer 中英机器翻译",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 子命令用于把不同任务的参数空间隔离,避免 translate 接收到训练参数.
    subparsers = parser.add_subparsers(dest="command", help="任务子命令")

    # ------------------------------------------------------------------
    # prepare:数据下载、清洗、分词器训练和 token 编码.
    # ------------------------------------------------------------------
    prepare_parser = subparsers.add_parser("prepare", help="准备完整训练数据")
    # --force 用于忽略已有文件并重新执行整个数据管线.
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并生成所有数据产物",
    )

    # ------------------------------------------------------------------
    # train:模型结构与训练行为均可通过命令行覆盖.
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="训练翻译模型")

    # 模型结构参数会直接映射到 ModelConfig 和 Transformer 构造参数.
    model_group = train_parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--d-model",
        type=int,
        default=model_defaults.d_model,
        help="token 表示维度(默认: %(default)s)",
    )
    model_group.add_argument(
        "--num-heads",
        type=int,
        default=model_defaults.num_heads,
        help="多头注意力头数,必须整除 d_model(默认: %(default)s)",
    )
    model_group.add_argument(
        "--d-feedforward",
        type=int,
        default=model_defaults.d_feedforward,
        help="前馈网络中间维度(默认: %(default)s)",
    )
    model_group.add_argument(
        "--encoder-layers",
        type=int,
        default=model_defaults.encoder_num_layers,
        help="Encoder 堆叠层数(默认: %(default)s)",
    )
    model_group.add_argument(
        "--decoder-layers",
        type=int,
        default=model_defaults.decoder_num_layers,
        help="Decoder 堆叠层数(默认: %(default)s)",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=model_defaults.dropout,
        help="Dropout 概率(默认: %(default)s)",
    )
    model_group.add_argument(
        "--max-seq-length",
        type=int,
        default=model_defaults.max_seq_length,
        help="位置编码支持的最大序列长度(默认: %(default)s)",
    )
    model_group.add_argument(
        "--norm-first",
        choices=["pre", "post"],
        default=model_defaults.norm_first,
        help="LayerNorm 放置方式(默认: %(default)s)",
    )

    # DataLoader 参数会影响显存占用、吞吐量和首次运行的数据准备行为.
    data_group = train_parser.add_argument_group("数据加载配置")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=loader_defaults.batch_size,
        help="训练和验证批次大小(默认: %(default)s)",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=loader_defaults.worker_count,
        help="DataLoader worker 数量(默认: %(default)s)",
    )
    data_group.add_argument(
        "--force-prepare",
        action="store_true",
        default=loader_defaults.force_prepare,
        help="训练前强制重新执行数据准备管线",
    )

    # 训练循环参数决定停止条件、验证成本、数值稳定性和日志频率.
    loop_group = train_parser.add_argument_group("训练循环配置")
    loop_group.add_argument(
        "--epochs",
        type=int,
        default=train_defaults.epoch_count,
        help="最大训练轮数(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--total-steps",
        type=int,
        default=train_defaults.total_training_steps,
        help="最大参数更新步数(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--validation-interval",
        type=int,
        default=train_defaults.validation_interval,
        help="预留的 step 级验证间隔(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--validation-batches",
        type=int,
        default=train_defaults.validation_batch_count,
        help="每轮最多验证的 batch 数(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--grad-clip",
        type=float,
        default=train_defaults.gradient_clip_norm,
        help="梯度范数裁剪阈值(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--seed",
        type=int,
        default=train_defaults.random_seed,
        help="随机种子(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--device",
        default=train_defaults.device,
        help="运行设备,例如 cpu、cuda 或 cuda:0(默认: %(default)s)",
    )
    loop_group.add_argument(
        "--log-interval",
        type=int,
        default=train_defaults.log_interval,
        help="每隔多少 step 写入一次训练指标(默认: %(default)s)",
    )
    # BooleanOptionalAction 自动生成 --mixed-precision / --no-mixed-precision 一对开关.
    loop_group.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=train_defaults.enable_mixed_precision,
        help="启用或关闭 CUDA 自动混合精度",
    )
    # --tensorboard / --no-tensorboard 开关对.
    loop_group.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=train_defaults.enable_tensorboard,
        help="启用或关闭 TensorBoard 日志",
    )
    # --progress / --no-progress 开关对.
    loop_group.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=train_defaults.show_progress,
        help="启用或关闭 tqdm 进度条",
    )

    # 优化器参数完整映射到 create_optimizer,便于通过 CLI 复现实验.
    optimizer_group = train_parser.add_argument_group("优化器配置")
    optimizer_group.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default=train_defaults.optimizer_type,
        help="优化器类型(默认: %(default)s)",
    )
    optimizer_group.add_argument(
        "--lr",
        type=float,
        default=train_defaults.learning_rate,
        help="初始学习率(默认: %(default)s)",
    )
    optimizer_group.add_argument(
        "--weight-decay",
        type=float,
        default=train_defaults.weight_decay,
        help="权重衰减系数(默认: %(default)s)",
    )
    optimizer_group.add_argument(
        "--beta1",
        type=float,
        default=train_defaults.optimizer_beta1,
        help="Adam 一阶动量衰减率(默认: %(default)s)",
    )
    optimizer_group.add_argument(
        "--beta2",
        type=float,
        default=train_defaults.optimizer_beta2,
        help="Adam 二阶动量衰减率(默认: %(default)s)",
    )
    optimizer_group.add_argument(
        "--optimizer-eps",
        type=float,
        default=train_defaults.optimizer_eps,
        help="优化器数值稳定项(默认: %(default)s)",
    )
    optimizer_group.add_argument(
        "--momentum",
        type=float,
        default=train_defaults.sgd_momentum,
        help="SGD 动量系数(默认: %(default)s)",
    )

    # 调度器参数决定每一步 optimizer.step() 之后如何修改学习率.
    scheduler_group = train_parser.add_argument_group("学习率调度配置")
    scheduler_group.add_argument(
        "--scheduler",
        choices=["constant", "cosine", "step", "exponential", "cosine_warmup"],
        default=train_defaults.scheduler_type,
        help="学习率调度策略(默认: %(default)s)",
    )
    scheduler_group.add_argument(
        "--warmup-steps",
        type=int,
        default=train_defaults.scheduler_warmup_steps,
        help="线性预热步数(默认: %(default)s)",
    )
    scheduler_group.add_argument(
        "--min-lr-ratio",
        type=float,
        default=train_defaults.scheduler_min_learning_rate_ratio,
        help="最低学习率相对初始值的比例(默认: %(default)s)",
    )
    scheduler_group.add_argument(
        "--scheduler-step-size",
        type=int,
        default=train_defaults.scheduler_step_size,
        help="StepLR 的衰减步长(默认: %(default)s)",
    )
    scheduler_group.add_argument(
        "--scheduler-gamma",
        type=float,
        default=train_defaults.scheduler_gamma,
        help="阶梯或指数调度器的衰减系数(默认: %(default)s)",
    )

    # 早停参数独立分组,便于关闭过拟合检测或调整收敛敏感度.
    early_stopping_group = train_parser.add_argument_group("早停配置")
    early_stopping_group.add_argument(
        "--patience",
        type=int,
        default=train_defaults.early_stopping_patience,
        help="允许验证指标连续无改善的次数(默认: %(default)s)",
    )
    early_stopping_group.add_argument(
        "--min-delta",
        type=float,
        default=train_defaults.early_stopping_min_delta,
        help="认定指标改善所需的最小差值(默认: %(default)s)",
    )
    early_stopping_group.add_argument(
        "--overfitting-threshold",
        type=float,
        default=train_defaults.early_stopping_overfitting_threshold,
        help="训练与验证 loss 差距阈值,默认关闭",
    )
    early_stopping_group.add_argument(
        "--convergence-window",
        type=int,
        default=train_defaults.early_stopping_convergence_window,
        help="收敛检测滑动窗口大小(默认: %(default)s)",
    )
    early_stopping_group.add_argument(
        "--convergence-threshold",
        type=float,
        default=train_defaults.early_stopping_convergence_threshold,
        help="窗口内验证指标标准差阈值(默认: %(default)s)",
    )
    # resume 接受完整 Path;恢复后模型结构以 checkpoint 配置为准.
    # 该参数放在 train_parser 顶层而非某个参数组,因为它不属于模型/数据/训练的任何单一类别.
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定 checkpoint 恢复训练",
    )

    # ------------------------------------------------------------------
    # eval:选择 checkpoint、数据切分和评估规模.
    # ------------------------------------------------------------------
    # aliases 让用户既可以用 "eval" 也可以用 "evaluate" 触发评估子命令.
    eval_parser = subparsers.add_parser("eval", aliases=["evaluate"], help="评估模型")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(paths.BEST_MODEL_PATH),
        help="checkpoint 路径(默认: %(default)s)",
    )
    eval_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="评估数据切分(默认: %(default)s)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=loader_defaults.batch_size,
        help="评估批次大小(默认: %(default)s)",
    )
    eval_parser.add_argument(
        "--num-workers",
        type=int,
        default=loader_defaults.worker_count,
        help="评估 DataLoader worker 数量(默认: %(default)s)",
    )
    eval_parser.add_argument(
        "--device",
        default=train_defaults.device,
        help="评估设备(默认: %(default)s)",
    )
    # 默认值为 None 表示不设上限,遍历整个数据切分.
    eval_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="最多评估的 batch 数,默认遍历完整切分",
    )
    # --progress / --no-progress 开关对,评估时默认显示进度条.
    eval_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或关闭 tqdm 进度条",
    )

    # ------------------------------------------------------------------
    # translate:既支持命令行单句,也支持无 text 时进入交互循环.
    # ------------------------------------------------------------------
    translate_parser = subparsers.add_parser("translate", help="翻译英文文本")
    # nargs="?" 使 text 成为可选位置参数:提供时直接翻译,省略时进入交互模式.
    translate_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="待翻译文本;省略时进入交互翻译模式",
    )
    translate_parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(paths.BEST_MODEL_PATH),
        help="checkpoint 路径(默认: %(default)s)",
    )
    translate_parser.add_argument(
        "--device",
        default=train_defaults.device,
        help="推理设备(默认: %(default)s)",
    )
    translate_parser.add_argument(
        "--max-length",
        type=int,
        default=inference_defaults.max_generation_length,
        help="最多生成的目标 token 数(默认: %(default)s)",
    )
    # --progress / --no-progress 开关对,推理时默认跟随 InferenceConfig 的设置.
    translate_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=inference_defaults.show_progress,
        help="启用或关闭 token 级 tqdm 进度条",
    )

    # 返回解析器本身,参数解析和任务分发由 main.py 完成.
    return parser


# 保留旧名称作为兼容入口,但新代码和 README 统一使用 create_parser.
build_parser = create_parser
