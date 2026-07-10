"""
项目路径配置

功能:
1. 统一管理项目目录、数据集目录和输出目录
2. 固定 Hugging Face 数据集缓存位置
3. 将 OPUS-100 英中数据集导出为项目可控的 JSONL 文件
"""

from pathlib import Path

# 项目根目录(configs 目录的上一级,即 MachineTranslation 目录)
ROOT = Path(__file__).resolve().parent.parent


def get_dir(path: Path) -> Path:
    """
    确保目录存在,不存在则创建

    Args:
        path (Path): 目标目录路径

    Returns:
        Path: 已确保存在的目录路径(原样返回)
    """
    # 目录不存在时递归创建(含所有父目录)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# 数据根目录可通过环境变量覆盖,默认保存在项目内,避免依赖固定磁盘路径
DATASETS_DIR = get_dir(Path("D:/Datasets/MachineTranslation"))

# Hugging Face 下载和缓存目录
DOWNLOADS_DIR = get_dir(DATASETS_DIR / "downloads")  # 通用下载缓存目录
HUGGINGFACE_CACHE_DIR = get_dir(
    DOWNLOADS_DIR / "huggingface"
)  # Hugging Face 专属缓存目录

# Hugging Face 会在缓存目录中自动管理这些子目录
HUGGINGFACE_DATASETS_DIR = get_dir(HUGGINGFACE_CACHE_DIR / "datasets")  # 数据集缓存目录
HUGGINGFACE_DOWNLOADS_DIR = get_dir(
    HUGGINGFACE_CACHE_DIR / "downloads"
)  # 下载文件缓存目录
HUGGINGFACE_MODULES_DIR = get_dir(
    HUGGINGFACE_CACHE_DIR / "modules"
)  # 自定义模块缓存目录

# 项目接管后的数据目录(原始 -> 中间 -> 处理后的三级流水线)
RAW_DATASETS_DIR = get_dir(
    DATASETS_DIR / "raw"
)  # 原始数据集目录(Hugging Face 导出后的原始数据)
INTERIM_DATASETS_DIR = get_dir(
    DATASETS_DIR / "interim"
)  # 中间数据集目录(经初步清洗/过滤后的数据)
PROCESSED_DATASETS_DIR = get_dir(
    DATASETS_DIR / "processed"
)  # 处理完成数据集目录(可直接用于训练的数据)

# OPUS-100 英中数据集配置
OPUS100_DATASET_NAME = "Helsinki-NLP/opus-100"  # Hugging Face 数据集标识符
OPUS100_LANGUAGE_PAIR = "en-zh"  # 语言对:英语 -> 中文
OPUS100_DATASET_DIR_NAME = "opus100_en_zh"  # 数据集在项目中的目录名

# 原始 JSONL 数据集文件(从 Hugging Face 下载后直接导出的数据)
RAW_OPUS100_DATASET_DIR = get_dir(RAW_DATASETS_DIR / OPUS100_DATASET_DIR_NAME)
RAW_TRAIN_DATASET_PATH = RAW_OPUS100_DATASET_DIR / "train.jsonl"  # 训练集
RAW_VAL_DATASET_PATH = RAW_OPUS100_DATASET_DIR / "validation.jsonl"  # 验证集
RAW_TEST_DATASET_PATH = RAW_OPUS100_DATASET_DIR / "test.jsonl"  # 测试集

# 中间处理 JSONL 数据集文件(经过初步清洗和过滤后的数据)
INTERIM_OPUS100_DATASET_DIR = get_dir(INTERIM_DATASETS_DIR / OPUS100_DATASET_DIR_NAME)
INTERIM_TRAIN_DATASET_PATH = (
    INTERIM_OPUS100_DATASET_DIR / "train.jsonl"
)  # 训练集(中间态)
INTERIM_VAL_DATASET_PATH = (
    INTERIM_OPUS100_DATASET_DIR / "validation.jsonl"
)  # 验证集(中间态)
INTERIM_TEST_DATASET_PATH = INTERIM_OPUS100_DATASET_DIR / "test.jsonl"  # 测试集(中间态)

# 处理后 JSONL 数据集文件(已完成分词、编码等预处理,可直接用于训练)
PROCESSED_OPUS100_DATASET_DIR = get_dir(
    PROCESSED_DATASETS_DIR / OPUS100_DATASET_DIR_NAME
)
PROCESSED_TRAIN_DATASET_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "train.jsonl"
)  # 训练集(最终态)
PROCESSED_VAL_DATASET_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "validation.jsonl"
)  # 验证集(最终态)
PROCESSED_TEST_DATASET_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "test.jsonl"
)  # 测试集(最终态)

# SentencePiece 分词器文件(训练和推理所需的全部资源)
TOKENIZER_CORPUS_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "tokenizer_corpus.txt"
)  # 训练分词器用的语料文本
TOKENIZER_MODEL_PREFIX_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "tokenizer"
)  # 分词器模型文件名前缀(不含扩展名)
TOKENIZER_MODEL_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "tokenizer.model"
)  # 分词器模型文件(二进制格式)
TOKENIZER_VOCAB_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "tokenizer.vocab"
)  # 分词器词汇表文件(SentencePiece 原生格式)
TOKENIZER_VOCAB_JSON_PATH = (
    PROCESSED_OPUS100_DATASET_DIR / "tokenizer_vocab.json"
)  # 分词器词汇表文件(JSON 格式,便于程序读取)

# 产出位置(所有训练和推理产出的根目录)
OUTPUTS_DIR = get_dir(ROOT / "outputs")

# 各类型产出子目录
CHECKPOINTS_DIR = get_dir(
    OUTPUTS_DIR / "checkpoints"
)  # 模型检查点保存目录(含训练中间状态)
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")  # 训练日志目录
TENSORBOARD_DIR = get_dir(OUTPUTS_DIR / "tensorboard")  # TensorBoard 事件文件目录
FIGURES_DIR = get_dir(OUTPUTS_DIR / "figures")  # 可视化图表输出目录
PREDICTIONS_DIR = get_dir(OUTPUTS_DIR / "predictions")  # 模型预测结果输出目录

# 模型保存位置(checkpoints 子目录下的固定文件名)
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"  # 验证集上表现最佳的模型权重
LAST_MODEL_PATH = (
    CHECKPOINTS_DIR / "last_model.pth"
)  # 最近一次保存的模型权重(用于断点续训)
