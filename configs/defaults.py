"""
configs/defaults.py
项目默认配置模块

功能：
1. 管理数据处理默认参数
2. 后续扩展模型和训练默认参数
"""

import torch


class DataConfig:
    """数据集配置"""

    # 英文句子最大词数，过长样本会增加显存和训练成本
    max_english_words: int = 128

    # 中文句子最大字符数，第一阶段按字符长度估算目标序列长度
    max_chinese_chars: int = 160

    # 中英文长度比例上限，用于过滤明显错位的平行句对
    max_length_ratio: float = 4.0

    # source 序列最大 token 数，包含 <eos>
    max_source_tokens: int = 128

    # target 序列最大 token 数，包含 <bos> 或 <eos>
    max_target_tokens: int = 128


class TokenizerConfig:
    """SentencePiece 分词器配置"""

    # 共享子词词表大小，第一版控制在 16k，兼顾效果和训练成本
    vocab_size: int = 16000

    # BPE 是机器翻译常见子词建模方式
    model_type: str = "bpe"

    # 中英混合语料需要较高字符覆盖率，避免中文字符被大量映射为 unk
    character_coverage: float = 0.9995

    # 限制参与训练 tokenizer 的句子数，避免训练 tokenizer 过慢
    input_sentence_size: int = 1000000

    # 固定特殊 token id，后续 Dataset 和 loss 会依赖 pad_id
    pad_id: int = 0
    unknown_id: int = 1
    begin_of_sentence_id: int = 2
    end_of_sentence_id: int = 3

    pad_token: str = "<pad>"
    unknown_token: str = "<unk>"
    begin_of_sentence_token: str = "<bos>"
    end_of_sentence_token: str = "<eos>"


class DataLoaderConfig:
    """DataLoader 默认配置"""

    # 默认 batch 大小，先用较保守的值，避免显存压力过大
    batch_size: int = 32

    # Windows 下多进程 DataLoader 容易引入额外复杂度，默认先用单进程
    worker_count: int = 0

    # 默认自动准备数据管线，保证 clone 项目后能直接获取 DataLoader
    auto_prepare: bool = True

    # 默认不强制重跑管线，避免重复下载和重复训练分词器
    force_prepare: bool = False


class ModelConfig:
    """Transformer 模型默认配置"""

    # 模型隐藏维度，embedding、attention 和前馈网络输入输出都使用该维度
    hidden_dim: int = 512

    # 多头注意力的头数，必须能整除 hidden_dim
    attention_head_count: int = 8

    # 前馈网络中间层维度，通常是 hidden_dim 的 4 倍
    feedforward_dim: int = 2048

    # Encoder 和 Decoder 的堆叠层数
    encoder_layer_count: int = 6
    decoder_layer_count: int = 6

    # Dropout 概率，用于 embedding、attention 输出和前馈网络
    dropout: float = 0.1

    # 位置编码支持的最大序列长度
    max_sequence_length: int = 5000


class TrainConfig:
    """训练流程默认配置"""

    # 默认训练轮数，第一版先保持保守，便于快速跑通流程
    epoch_count: int = 10

    # 总训练步数，step 级调度器和训练停止条件都会使用该值
    total_training_steps: int = 100000

    # 每隔多少个 step 执行一次快速验证并保存 best / last
    validation_interval: int = 100

    # 每次快速验证最多使用多少个 batch
    validation_batch_count: int = 100

    # 优化器类型，支持 adam / adamw / sgd
    optimizer_type: str = "adamw"

    # 初始学习率
    learning_rate: float = 1e-4

    # 权重衰减，默认关闭，便于先跑通最小训练流程
    weight_decay: float = 0.0

    # Adam / AdamW 动量参数
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999

    # Adam / AdamW 数值稳定项
    optimizer_eps: float = 1e-8

    # SGD 动量参数
    sgd_momentum: float = 0.9

    # 学习率调度器类型，支持 constant / cosine / step / exponential / cosine_warmup
    scheduler_type: str = "cosine_warmup"

    # warmup 步数，cosine_warmup 使用
    scheduler_warmup_steps: int = 500

    # 最小学习率比例，cosine / exponential / cosine_warmup 使用
    scheduler_min_learning_rate_ratio: float = 0.01

    # StepLR 每隔多少个 step 衰减一次
    scheduler_step_size: int = 1000

    # StepLR / ExponentialLR 学习率衰减比例
    scheduler_gamma: float = 0.8

    # early stopping 容忍验证集无提升的检查次数
    early_stopping_patience: int = 200

    # 判断验证集提升时需要超过的最小差值
    early_stopping_min_delta: float = 1e-4

    # early stopping 优化方向，loss 使用 min，accuracy 可使用 max
    early_stopping_mode: str = "min"

    # 过拟合阈值，None 表示关闭过拟合检测
    early_stopping_overfitting_threshold: float | None = None

    # 收敛检测窗口大小
    early_stopping_convergence_window: int = 3

    # 收敛检测阈值，窗口内标准差低于该值认为已收敛
    early_stopping_convergence_threshold: float = 1e-4

    # 梯度裁剪阈值，避免训练初期梯度爆炸
    gradient_clip_norm: float = 1.0

    # CUDA 下默认启用自动混合精度，CPU 会自动关闭
    enable_mixed_precision: bool = True

    # 每隔多少个 batch 打印一次训练日志
    log_interval: int = 50

    # 默认训练设备，直接由 torch.cuda.is_available() 决定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 随机种子，保证实验尽量可复现
    random_seed: int = 42

    # logger 名称
    logger_name: str = "training"

    # 是否启用 TensorBoard，如果环境缺少 tensorboard 会自动降级
    enable_tensorboard: bool = True


class InferenceConfig:
    """推理流程默认配置"""

    # 自回归生成的最大 token 数，包含起始 token 后续生成部分
    max_generation_length: int = 128
