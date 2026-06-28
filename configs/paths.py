"""
项目路径配置

功能：
1. 统一管理项目目录、数据集目录和输出目录
2. 固定 Hugging Face 数据集缓存位置
3. 将 OPUS-100 英中数据集导出为项目可控的 JSONL 文件
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_dir(path: Path) -> Path:
    """确保目录存在，不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# 项目外部数据根目录
DATASETS_DIR = get_dir(Path("D:/Datasets/MachineTranslation"))

# Hugging Face 下载和缓存目录
DOWNLOADS_DIR = get_dir(DATASETS_DIR / "downloads")
HUGGINGFACE_CACHE_DIR = get_dir(DOWNLOADS_DIR / "huggingface")

# Hugging Face 会在缓存目录中自动管理这些子目录
HUGGINGFACE_DATASETS_DIR = get_dir(HUGGINGFACE_CACHE_DIR / "datasets")
HUGGINGFACE_DOWNLOADS_DIR = get_dir(HUGGINGFACE_CACHE_DIR / "downloads")
HUGGINGFACE_MODULES_DIR = get_dir(HUGGINGFACE_CACHE_DIR / "modules")

# 项目接管后的数据目录
RAW_DATASETS_DIR = get_dir(DATASETS_DIR / "raw")
INTERIM_DATASETS_DIR = get_dir(DATASETS_DIR / "interim")
PROCESSED_DATASETS_DIR = get_dir(DATASETS_DIR / "processed")

# OPUS-100 英中数据集配置
OPUS100_DATASET_NAME = "Helsinki-NLP/opus-100"
OPUS100_LANGUAGE_PAIR = "en-zh"
OPUS100_DATASET_DIR_NAME = "opus100_en_zh"

# 原始 JSONL 数据集文件
RAW_OPUS100_DATASET_DIR = get_dir(RAW_DATASETS_DIR / OPUS100_DATASET_DIR_NAME)
RAW_TRAIN_DATASET_PATH = RAW_OPUS100_DATASET_DIR / "train.jsonl"
RAW_VAL_DATASET_PATH = RAW_OPUS100_DATASET_DIR / "validation.jsonl"
RAW_TEST_DATASET_PATH = RAW_OPUS100_DATASET_DIR / "test.jsonl"

# 中间处理 JSONL 数据集文件
INTERIM_OPUS100_DATASET_DIR = get_dir(INTERIM_DATASETS_DIR / OPUS100_DATASET_DIR_NAME)
INTERIM_TRAIN_DATASET_PATH = INTERIM_OPUS100_DATASET_DIR / "train.jsonl"
INTERIM_VAL_DATASET_PATH = INTERIM_OPUS100_DATASET_DIR / "validation.jsonl"
INTERIM_TEST_DATASET_PATH = INTERIM_OPUS100_DATASET_DIR / "test.jsonl"

# 处理后 JSONL 数据集文件
PROCESSED_OPUS100_DATASET_DIR = get_dir(
    PROCESSED_DATASETS_DIR / OPUS100_DATASET_DIR_NAME
)
PROCESSED_TRAIN_DATASET_PATH = PROCESSED_OPUS100_DATASET_DIR / "train.jsonl"
PROCESSED_VAL_DATASET_PATH = PROCESSED_OPUS100_DATASET_DIR / "validation.jsonl"
PROCESSED_TEST_DATASET_PATH = PROCESSED_OPUS100_DATASET_DIR / "test.jsonl"

# SentencePiece 分词器文件
TOKENIZER_CORPUS_PATH = PROCESSED_OPUS100_DATASET_DIR / "tokenizer_corpus.txt"
TOKENIZER_MODEL_PREFIX_PATH = PROCESSED_OPUS100_DATASET_DIR / "tokenizer"
TOKENIZER_MODEL_PATH = PROCESSED_OPUS100_DATASET_DIR / "tokenizer.model"
TOKENIZER_VOCAB_PATH = PROCESSED_OPUS100_DATASET_DIR / "tokenizer.vocab"
TOKENIZER_VOCAB_JSON_PATH = PROCESSED_OPUS100_DATASET_DIR / "tokenizer_vocab.json"

# 产出位置
OUTPUTS_DIR = get_dir(ROOT / "outputs")

CHECKPOINTS_DIR = get_dir(OUTPUTS_DIR / "checkpoints")
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")
TENSORBOARD_DIR = get_dir(OUTPUTS_DIR / "tensorboard")
FIGURES_DIR = get_dir(OUTPUTS_DIR / "figures")

# 模型保存位置
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"
