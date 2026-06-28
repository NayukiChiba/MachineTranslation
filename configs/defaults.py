"""
configs/defaults.py
项目默认配置模块

功能：
1. 管理数据处理默认参数
2. 后续扩展模型和训练默认参数
"""


class DataConfig:
    """数据集配置"""

    # 英文句子最大词数，过长样本会增加显存和训练成本
    MAX_ENGLISH_WORDS: int = 128

    # 中文句子最大字符数，第一阶段按字符长度估算目标序列长度
    MAX_CHINESE_CHARS: int = 160

    # 中英文长度比例上限，用于过滤明显错位的平行句对
    MAX_LENGTH_RATIO: float = 4.0

    # source 序列最大 token 数，包含 <eos>
    MAX_SOURCE_TOKENS: int = 128

    # target 序列最大 token 数，包含 <bos> 或 <eos>
    MAX_TARGET_TOKENS: int = 128


class TokenizerConfig:
    """SentencePiece 分词器配置"""

    # 共享子词词表大小，第一版控制在 16k，兼顾效果和训练成本
    VOCAB_SIZE: int = 16000

    # BPE 是机器翻译常见子词建模方式
    MODEL_TYPE: str = "bpe"

    # 中英混合语料需要较高字符覆盖率，避免中文字符被大量映射为 unk
    CHARACTER_COVERAGE: float = 0.9995

    # 限制参与训练 tokenizer 的句子数，避免训练 tokenizer 过慢
    INPUT_SENTENCE_SIZE: int = 1000000

    # 固定特殊 token id，后续 Dataset 和 loss 会依赖 pad_id
    PAD_ID: int = 0
    UNK_ID: int = 1
    BOS_ID: int = 2
    EOS_ID: int = 3

    PAD_TOKEN: str = "<pad>"
    UNK_TOKEN: str = "<unk>"
    BOS_TOKEN: str = "<bos>"
    EOS_TOKEN: str = "<eos>"


class DataLoaderConfig:
    """DataLoader 默认配置"""

    # 默认 batch 大小，先用较保守的值，避免显存压力过大
    BATCH_SIZE: int = 32

    # Windows 下多进程 DataLoader 容易引入额外复杂度，默认先用单进程
    NUM_WORKERS: int = 0

    # 默认自动准备数据管线，保证 clone 项目后能直接获取 DataLoader
    AUTO_PREPARE: bool = True

    # 默认不强制重跑管线，避免重复下载和重复训练分词器
    FORCE_PREPARE: bool = False
