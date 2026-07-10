"""
从 Hugging Face 缓存导出 OPUS-100 中英数据集

功能:
1. 读取 Hugging Face 已缓存的 OPUS-100 en-zh 数据集
2. 将 train、validation、test 导出为 JSONL 文件
3. 保存到项目 raw 数据集目录

使用方法:
    uv run python -m src.data.download
"""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import load_dataset
from tqdm.auto import tqdm

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
    # Hugging Face Dataset 实现 __len__;通用 Iterable 则允许 total=None.
    item_count = len(dataset_split) if hasattr(dataset_split, "__len__") else None
    # 输出文件采用一行一个 JSON 对象的 JSONL 格式,便于流式处理.
    with output_path.open("w", encoding="utf-8") as file:
        # tqdm 以样本为单位展示当前切分的导出进度.
        for item in tqdm(
            dataset_split,
            total=item_count,
            desc=f"export {output_path.stem}",
            unit="sample",
            dynamic_ncols=True,
        ):
            # OPUS-100 把双语句对存放在 translation 子字典中.
            translation = item["translation"]
            # 只保留本项目需要的英文 source 和中文 target.
            data_item = {
                "en": translation["en"],
                "zh": translation["zh"],
            }

            # ensure_ascii=False 保留中文,便于直接检查原始文件.
            file.write(json.dumps(data_item, ensure_ascii=False) + "\n")


def download_raw_dataset() -> None:
    """从 Hugging Face 下载或缓存中导出原始 JSONL 数据集"""
    # 从 Hugging Face 加载 OPUS-100 中英数据集,若本地已缓存则直接使用缓存
    dataset = load_dataset(
        paths.OPUS100_DATASET_NAME,
        paths.OPUS100_LANGUAGE_PAIR,
        cache_dir=str(paths.HUGGINGFACE_CACHE_DIR),
    )

    # 打印数据集结构,便于确认切分名称和数据量
    print(dataset)

    # 分别导出训练集、验证集和测试集为 JSONL 文件
    save_split_to_jsonl(dataset["train"], paths.RAW_TRAIN_DATASET_PATH)
    save_split_to_jsonl(dataset["validation"], paths.RAW_VAL_DATASET_PATH)
    save_split_to_jsonl(dataset["test"], paths.RAW_TEST_DATASET_PATH)

    # 输出导出完成的提示及各个输出文件路径
    print("OPUS-100 中英数据集导出完成")
    print(f"训练集: {paths.RAW_TRAIN_DATASET_PATH}")
    print(f"验证集: {paths.RAW_VAL_DATASET_PATH}")
    print(f"测试集: {paths.RAW_TEST_DATASET_PATH}")


def main() -> None:
    """主函数:生成 raw 数据集"""
    download_raw_dataset()


if __name__ == "__main__":
    main()
