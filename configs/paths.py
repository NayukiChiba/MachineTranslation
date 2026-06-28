from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_dir(path: Path) -> Path:
    """确保目录存在,不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# 数据集位置
DATASETS_DIR = get_dir(Path("D:/Datasets/MathineTranslation"))
# 下载位置
DOWNLOADS_DIR = get_dir(DATASETS_DIR / "downloads")
# 原始数据集位置
RAW_DATASETS_DIR = get_dir(DATASETS_DIR / "raw")
# 处理后数据集位置
PROCESSED_DATASETS_DIR = get_dir(DATASETS_DIR / "processed")
# 中间处理数据集位置
INTERIM_DATASETS_DIR = get_dir(DATASETS_DIR / "interim")

# 产出位置
OUTPUTS_DIR = get_dir(ROOT / "outputs")

CHECKPOINTS_DIR = get_dir(OUTPUTS_DIR / "checkpoints")
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")
TENSORBOARD_DIR = get_dir(OUTPUTS_DIR / "tensorboard")
FIGURES_DIR = get_dir(OUTPUTS_DIR / "figures")

# 模型保存位置
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"

# 下载的数据集文件
# DATASET_ZIP_PATH = DOWNLOADS_DIR / "THUCNews.zip"
# THUCNEWS_RAW_DIR = RAW_DATASETS_DIR / "THUCNews"

# 中间处理数据集文件
INTERIM_TRAIN_DATASET_PATH = INTERIM_DATASETS_DIR / "train.txt"
INTERIM_VAL_DATASET_PATH = INTERIM_DATASETS_DIR / "val.txt"
INTERIM_TEST_DATASET_PATH = INTERIM_DATASETS_DIR / "test.txt"

# 处理后数据集文件
PROCESSED_TRAIN_DATASET_PATH = PROCESSED_DATASETS_DIR / "train_processed.txt"
PROCESSED_VAL_DATASET_PATH = PROCESSED_DATASETS_DIR / "val_processed.txt"
PROCESSED_TEST_DATASET_PATH = PROCESSED_DATASETS_DIR / "test_processed.txt"
