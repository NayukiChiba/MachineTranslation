"""
从 Hugging Face 缓存导出 OPUS-100 中英数据集

功能：
1. 读取 Hugging Face 已缓存的 OPUS-100 en-zh 数据集
2. 将 train、validation、test 导出为 JSONL 文件
3. 保存到项目 raw 数据集目录

使用方法：
    uv run python -m src.data.download
"""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import load_dataset

from configs import paths


def save_split_to_jsonl(
    dataset_split: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """
    保存数据集切分为 JSONL 文件

    Args:
        dataset_split: Hugging Face 数据集切分
        output_path: 输出 JSONL 文件路径
    """
    with output_path.open("w", encoding="utf-8") as file:
        for item in dataset_split:
            translation = item["translation"]
            data_item = {
                "en": translation["en"],
                "zh": translation["zh"],
            }

            file.write(json.dumps(data_item, ensure_ascii=False) + "\n")


def download_raw_dataset() -> None:
    """从 Hugging Face 下载或缓存中导出原始 JSONL 数据集"""
    dataset = load_dataset(
        paths.OPUS100_DATASET_NAME,
        paths.OPUS100_LANGUAGE_PAIR,
        cache_dir=str(paths.HUGGINGFACE_CACHE_DIR),
    )

    print(dataset)

    save_split_to_jsonl(dataset["train"], paths.RAW_TRAIN_DATASET_PATH)
    save_split_to_jsonl(dataset["validation"], paths.RAW_VAL_DATASET_PATH)
    save_split_to_jsonl(dataset["test"], paths.RAW_TEST_DATASET_PATH)

    print("OPUS-100 中英数据集导出完成")
    print(f"训练集: {paths.RAW_TRAIN_DATASET_PATH}")
    print(f"验证集: {paths.RAW_VAL_DATASET_PATH}")
    print(f"测试集: {paths.RAW_TEST_DATASET_PATH}")


def main() -> None:
    """主函数：生成 raw 数据集"""
    download_raw_dataset()


if __name__ == "__main__":
    main()
