"""
configs/defaults.py
项目默认配置模块

功能:
1. 管理数据处理默认参数
2. 后续扩展模型和训练默认参数
"""

from typing import Any, Literal, Self

import torch


class StaticConfig:
    """静态配置基类,所有配置只允许通过类属性访问."""

    def __new__(cls) -> Self:
        """禁止创建配置实例,避免静态配置与实例配置混用."""
        raise TypeError(f"{cls.__name__} 是静态配置类,不能实例化")

    @classmethod
    def getFieldNames(cls) -> tuple[str, ...]:
        """按声明顺序返回配置字段名."""
        fieldNames: list[str] = []
        for configClass in reversed(cls.__mro__):
            if not issubclass(configClass, StaticConfig):
                continue
            for fieldName in configClass.__dict__.get("__annotations__", {}):
                if fieldName not in fieldNames:
                    fieldNames.append(fieldName)
        return tuple(fieldNames)

    @classmethod
    def withOverrides(cls, **overrides: Any) -> type[Self]:
        """返回带覆盖值的新静态配置类,不会创建配置实例."""
        unknownFields = set(overrides) - set(cls.getFieldNames())
        if unknownFields:
            unknownNames = ", ".join(sorted(unknownFields))
            raise AttributeError(f"未知配置字段: {unknownNames}")
        configuredClass = type(
            cls.__name__,
            (cls,),
            {"__module__": cls.__module__, **overrides},
        )
        configuredClass.validate()
        return configuredClass

    @classmethod
    def toDict(cls) -> dict[str, Any]:
        """将静态配置导出为普通字典."""
        return {fieldName: getattr(cls, fieldName) for fieldName in cls.getFieldNames()}

    @classmethod
    def validate(cls) -> None:
        """校验配置值,具体约束由子类实现."""

    @classmethod
    def printSummary(cls) -> None:
        """逐项打印当前静态配置."""
        print(f"[{cls.__name__}]")
        for fieldName, value in cls.toDict().items():
            print(f"  {fieldName} = {value!r}")


class DataConfig(StaticConfig):
    """数据集配置"""

    # 英文句子最大词数,过长样本会增加显存和训练成本
    max_english_words: int = 128

    # 中文句子最大字符数,第一阶段按字符长度估算目标序列长度
    max_chinese_chars: int = 160

    # 中英文长度比例上限,用于过滤明显错位的平行句对
    max_length_ratio: float = 4.0

    # source 序列最大 token 数,包含 <eos>
    max_source_tokens: int = 128

    # target 序列最大 token 数,包含 <bos> 或 <eos>
    max_target_tokens: int = 128

    @classmethod
    def validate(cls) -> None:
        """校验文本和 token 长度限制,避免数据阶段产生空序列."""
        if cls.max_english_words <= 0 or cls.max_chinese_chars <= 0:
            raise ValueError("原始文本长度限制必须大于 0")
        if cls.max_length_ratio < 1:
            raise ValueError("中英文长度比例上限不能小于 1")
        if cls.max_source_tokens <= 1 or cls.max_target_tokens <= 1:
            raise ValueError("token 长度上限必须大于 1,以容纳特殊 token")


class TokenizerConfig(StaticConfig):
    """SentencePiece 分词器配置"""

    # 共享子词词表大小,第一版控制在 16k,兼顾效果和训练成本
    vocab_size: int = 16000

    # BPE 是机器翻译常见子词建模方式
    model_type: str = "bpe"

    # 中英混合语料需要较高字符覆盖率,避免中文字符被大量映射为 unk
    character_coverage: float = 0.9995

    # 限制参与训练 tokenizer 的句子数,避免训练 tokenizer 过慢
    input_sentence_size: int = 1000000

    # 固定特殊 token id,后续 Dataset 和 loss 会依赖 pad_id
    pad_id: int = 0
    unknown_id: int = 1
    begin_of_sentence_id: int = 2
    end_of_sentence_id: int = 3

    pad_token: str = "<pad>"
    unknown_token: str = "<unk>"
    begin_of_sentence_token: str = "<bos>"
    end_of_sentence_token: str = "<eos>"

    @classmethod
    def validate(cls) -> None:
        """校验 SentencePiece 参数和特殊 token id."""
        if cls.vocab_size <= 4:
            raise ValueError("词表大小必须大于特殊 token 数量")
        if not 0 < cls.character_coverage <= 1:
            raise ValueError("character_coverage 必须在 (0, 1] 范围内")
        special_ids = {
            cls.pad_id,
            cls.unknown_id,
            cls.begin_of_sentence_id,
            cls.end_of_sentence_id,
        }
        if len(special_ids) != 4:
            raise ValueError("四个特殊 token id 不能重复")


class DataLoaderConfig(StaticConfig):
    """DataLoader 默认配置"""

    # 默认 batch 大小,先用较保守的值,避免显存压力过大
    batch_size: int = 32

    # Windows 下多进程 DataLoader 容易引入额外复杂度,默认先用单进程
    worker_count: int = 0

    # 默认自动准备数据管线,保证 clone 项目后能直接获取 DataLoader
    auto_prepare: bool = True

    # 默认不强制重跑管线,避免重复下载和重复训练分词器
    force_prepare: bool = False

    @classmethod
    def validate(cls) -> None:
        """校验 DataLoader 并行和批次参数."""
        if cls.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if cls.worker_count < 0:
            raise ValueError("worker_count 不能小于 0")


class ModelConfig(StaticConfig):
    """Transformer 模型默认配置"""

    # 模型隐藏维度,embedding、attention 和前馈网络输入输出都使用该维度
    d_model: int = 512

    # 多头注意力的头数,必须能整除 d_model
    num_heads: int = 8

    # 前馈网络中间层维度,通常是 d_model 的 4 倍
    d_feedforward: int = 2048

    # Encoder 和 Decoder 的堆叠层数
    encoder_num_layers: int = 6
    decoder_num_layers: int = 6

    # Dropout 概率,用于 embedding、attention 输出和前馈网络
    dropout: float = 0.1

    # 位置编码支持的最大序列长度
    max_seq_length: int = 5000

    # LayerNorm的放置策略
    norm_first: Literal["pre", "post"] = "pre"

    @classmethod
    def validate(cls) -> None:
        """校验 Transformer 各维度之间的必要约束."""
        if cls.d_model <= 0 or cls.num_heads <= 0:
            raise ValueError("d_model 和 num_heads 必须大于 0")
        if cls.d_model % cls.num_heads != 0:
            raise ValueError("d_model 必须能够被 num_heads 整除")
        if cls.d_feedforward <= 0:
            raise ValueError("d_feedforward 必须大于 0")
        if cls.encoder_num_layers <= 0 or cls.decoder_num_layers <= 0:
            raise ValueError("Encoder 和 Decoder 层数必须大于 0")
        if not 0 <= cls.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        if cls.max_seq_length <= 0:
            raise ValueError("max_seq_length 必须大于 0")
        if cls.norm_first not in {"pre", "post"}:
            raise ValueError("norm_first 必须是 pre 或 post")


class TrainConfig(StaticConfig):
    """训练流程默认配置"""

    # 默认训练轮数,第一版先保持保守,便于快速跑通流程
    epoch_count: int = 10

    # 总训练步数,step 级调度器和训练停止条件都会使用该值
    total_training_steps: int = 100000

    # 每隔多少个 step 执行一次快速验证并保存 best / last
    validation_interval: int = 100

    # 每次快速验证最多使用多少个 batch
    validation_batch_count: int = 100

    # 优化器类型,支持 adam / adamw / sgd
    optimizer_type: Literal["adam", "adamw", "sgd"] = "adamw"

    # 初始学习率
    learning_rate: float = 1e-4

    # 权重衰减,默认关闭,便于先跑通最小训练流程
    weight_decay: float = 0.0

    # Adam / AdamW 动量参数
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999

    # Adam / AdamW 数值稳定项
    optimizer_eps: float = 1e-8

    # SGD 动量参数
    sgd_momentum: float = 0.9

    # 学习率调度器类型,支持 constant / cosine / step / exponential / cosine_warmup
    scheduler_type: Literal[
        "constant", "cosine", "step", "exponential", "cosine_warmup"
    ] = "cosine_warmup"

    # warmup 步数,cosine_warmup 使用
    scheduler_warmup_steps: int = 500

    # 最小学习率比例,cosine / exponential / cosine_warmup 使用
    scheduler_min_learning_rate_ratio: float = 0.01

    # StepLR 每隔多少个 step 衰减一次
    scheduler_step_size: int = 1000

    # StepLR / ExponentialLR 学习率衰减比例
    scheduler_gamma: float = 0.8

    # early stopping 容忍验证集无提升的检查次数
    early_stopping_patience: int = 200

    # 判断验证集提升时需要超过的最小差值
    early_stopping_min_delta: float = 1e-4

    # early stopping 优化方向,loss 使用 min,accuracy 可使用 max
    early_stopping_mode: str = "min"

    # 过拟合阈值,None 表示关闭过拟合检测
    early_stopping_overfitting_threshold: float | None = None

    # 收敛检测窗口大小
    early_stopping_convergence_window: int = 3

    # 收敛检测阈值,窗口内标准差低于该值认为已收敛
    early_stopping_convergence_threshold: float = 1e-4

    # 梯度裁剪阈值,避免训练初期梯度爆炸
    gradient_clip_norm: float = 1.0

    # CUDA 下默认启用自动混合精度,CPU 会自动关闭
    enable_mixed_precision: bool = True

    # 每隔多少个 batch 打印一次训练日志
    log_interval: int = 50

    # 默认训练设备,直接由 torch.cuda.is_available() 决定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 随机种子,保证实验尽量可复现
    random_seed: int = 42

    # logger 名称
    logger_name: str = "training"

    # 是否启用 TensorBoard,如果环境缺少 tensorboard 会自动降级
    enable_tensorboard: bool = True

    # 是否显示 tqdm 进度条;测试或日志重定向场景可通过 CLI 关闭
    show_progress: bool = True

    @classmethod
    def validate(cls) -> None:
        """校验训练、优化器、调度器和早停参数."""
        if cls.epoch_count <= 0 or cls.total_training_steps <= 0:
            raise ValueError("epoch_count 和 total_training_steps 必须大于 0")
        if cls.validation_interval <= 0 or cls.validation_batch_count <= 0:
            raise ValueError("验证间隔和验证 batch 数必须大于 0")
        if cls.learning_rate <= 0 or cls.weight_decay < 0:
            raise ValueError("learning_rate 必须大于 0,weight_decay 不能小于 0")
        if cls.optimizer_type not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer_type 必须是 adam、adamw 或 sgd")
        if not 0 < cls.optimizer_beta1 < 1 or not 0 < cls.optimizer_beta2 < 1:
            raise ValueError("Adam beta 参数必须在 (0, 1) 范围内")
        if cls.optimizer_eps <= 0:
            raise ValueError("optimizer_eps 必须大于 0")
        if cls.sgd_momentum < 0:
            raise ValueError("sgd_momentum 不能小于 0")
        if cls.scheduler_type not in {
            "constant",
            "cosine",
            "step",
            "exponential",
            "cosine_warmup",
        }:
            raise ValueError("scheduler_type 不受支持")
        if cls.scheduler_warmup_steps < 0:
            raise ValueError("scheduler_warmup_steps 不能小于 0")
        if (
            cls.scheduler_type == "cosine_warmup"
            and cls.scheduler_warmup_steps > cls.total_training_steps
        ):
            raise ValueError("scheduler_warmup_steps 不能超过总训练步数")
        if not 0 <= cls.scheduler_min_learning_rate_ratio <= 1:
            raise ValueError("最小学习率比例必须在 [0, 1] 范围内")
        if cls.scheduler_step_size <= 0 or cls.scheduler_gamma <= 0:
            raise ValueError("调度器 step_size 和 gamma 必须大于 0")
        if cls.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience 必须大于 0")
        if cls.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta 不能小于 0")
        if cls.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode 必须是 min 或 max")
        if cls.early_stopping_convergence_window <= 1:
            raise ValueError("收敛检测窗口必须大于 1")
        if cls.early_stopping_convergence_threshold < 0:
            raise ValueError("收敛检测阈值不能小于 0")
        if cls.gradient_clip_norm <= 0 or cls.log_interval <= 0:
            raise ValueError("梯度裁剪阈值和日志间隔必须大于 0")


class InferenceConfig(StaticConfig):
    """推理流程默认配置"""

    # 自回归生成的最大 token 数,包含起始 token 后续生成部分
    max_generation_length: int = 128

    # 是否在自回归解码时显示 token 级进度条
    show_progress: bool = True

    @classmethod
    def validate(cls) -> None:
        """校验自回归生成长度."""
        if cls.max_generation_length <= 0:
            raise ValueError("max_generation_length 必须大于 0")


# 静态配置不会经历实例初始化,因此在模块加载时显式校验默认值.
DataConfig.validate()
TokenizerConfig.validate()
DataLoaderConfig.validate()
ModelConfig.validate()
TrainConfig.validate()
InferenceConfig.validate()
