"""
生成 interim 数据集模块

功能:
1. 读取 raw JSONL 数据
2. 清洗英文和中文文本
3. 过滤无效样本
4. 去除重复样本
5. 保存到 interim JSONL 数据集

使用方法:
    uv run -m src.data.interim
"""

import json
import re
from pathlib import Path

from configs import paths
from configs.defaults import DataConfig
from src.data.utils import load_jsonl_item


def normalize_english_text(text: str) -> str:
    """
    规范化英文文本

    Args:
        text: 英文文本

    Returns:
        规范化后的英文文本
    """
    # 去除首尾空格
    text = text.strip()
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_chinese_text(text: str) -> str:
    """
    规范化中文文本

    Args:
        text: 中文文本

    Returns:
        规范化后的中文文本
    """
    # 去除首尾空格
    text = text.strip()
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)

    return text


def count_english_words(text: str) -> int:
    """
    统计英文文本中的单词数量

    Args:
        text: 英文文本

    Returns:
        单词数量
    """
    if not text:
        return 0

    return len(text.split())


def count_chinese_chars(text: str) -> int:
    """
    统计中文文本中的字符数量

    Args:
        text: 中文文本

    Returns:
        字符数量
    """
    if not text:
        return 0

    return len(text)


def is_valid_sample(english_text: str, chinese_text: str) -> bool:
    """
    判断中英文样本是否有效

    Args:
        english_text: 英文文本
        chinese_text: 中文文本

    Returns:
        是否有效
    """
    english_word_count = count_english_words(english_text)
    chinese_char_count = count_chinese_chars(chinese_text)

    if (
        english_word_count == 0
        or chinese_char_count == 0
        or english_word_count > DataConfig.MAX_ENGLISH_WORDS
        or chinese_char_count > DataConfig.MAX_CHINESE_CHARS
        or max(
            english_word_count / chinese_char_count,
            chinese_char_count / english_word_count,
        )
        > DataConfig.MAX_LENGTH_RATIO
    ):
        return False

    return True


def create_interim_file(input_path: Path, output_path: Path) -> None:
    """
    清洗单个 JSONL 文件

    Args:
        input_path: 输入 raw JSONL 文件路径
        output_path: 输出 interim JSONL 文件路径
    """
    total_count = 0
    valid_count = 0
    invalid_count = 0
    duplicate_count = 0
    seen_samples: set[tuple[str, str]] = set()

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            total_count += 1
            item = load_jsonl_item(line)

            if item is None:
                invalid_count += 1
                continue
            english_text = normalize_english_text(str(item.get("en", "")))
            chinese_text = normalize_chinese_text(str(item.get("zh", "")))

            if not is_valid_sample(english_text, chinese_text):
                invalid_count += 1
                continue
            sample_key = (english_text, chinese_text)

            if sample_key in seen_samples:
                duplicate_count += 1
                continue

            seen_samples.add(sample_key)

            data_item = {
                "en": english_text,
                "zh": chinese_text,
            }

            output_file.write(json.dumps(data_item, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"总样本数: {total_count}")
    print(f"有效样本数: {valid_count}")
    print(f"无效样本数: {invalid_count}")
    print(f"重复样本数: {duplicate_count}")
    print("-" * 60)


def create_interim_splits() -> None:
    """生成 train / validation / test 三个 interim 数据集"""
    create_interim_file(
        paths.RAW_TRAIN_DATASET_PATH,
        paths.INTERIM_TRAIN_DATASET_PATH,
    )
    create_interim_file(
        paths.RAW_VAL_DATASET_PATH,
        paths.INTERIM_VAL_DATASET_PATH,
    )
    create_interim_file(
        paths.RAW_TEST_DATASET_PATH,
        paths.INTERIM_TEST_DATASET_PATH,
    )

    print("OPUS-100 英中数据集清洗完成")


def main() -> None:
    """主函数：生成 interim 数据集"""
    create_interim_splits()


if __name__ == "__main__":
    main()
