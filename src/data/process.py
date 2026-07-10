"""
生成 processed 数据集模块

功能:
1. 读取 interim JSONL 文本句对
2. 使用 SentencePiece 分词器转换为 token id
3. 保存为 processed JSONL,供训练阶段直接读取

使用方法:
    uv run -m src.data.process
"""

import json
from pathlib import Path

from tqdm.auto import tqdm

from configs import paths
from configs.defaults import DataConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.data.utils import load_jsonl_item


def truncate_token_ids(token_ids: list[int], max_length: int) -> list[int]:
    """
    截断 token id 序列

    Args:
        token_ids: token id 列表
        max_length: 最大长度

    Returns:
        list[int]: 截断后的 token id 列表
    """
    return token_ids[:max_length]


def encode_translation_pair(
    english_text: str,
    chinese_text: str,
    tokenizer: SentencePieceTokenizer,
    max_source_tokens: int = DataConfig.max_source_tokens,
    max_target_tokens: int = DataConfig.max_target_tokens,
) -> dict[str, list[int]]:
    """
    将一个英中句对编码为 Transformer 训练样本

    Args:
        english_text: 英文 source 文本,送入 encoder
        chinese_text: 中文 target 文本,送入 decoder
        tokenizer: SentencePiece 分词器
        max_source_tokens: source 最大 token 数
        max_target_tokens: target 最大 token 数

    Returns:
        dict[str, list[int]]: 编码后的训练样本
    """
    # 英文 source 末尾追加 EOS,中文 target 不追加(在下方统一构造)
    source_ids = tokenizer.encode(english_text, add_eos=True)
    target_ids = tokenizer.encode(chinese_text)

    # 截断到最大长度,target 预留 1 个位置给 EOS
    source_ids = truncate_token_ids(source_ids, max_source_tokens)
    target_ids = truncate_token_ids(target_ids, max_target_tokens - 1)

    # 构造训练样本:source 用于 encoder,target_input 用于 decoder 输入,target_output 用于计算 loss
    return {
        "source_ids": source_ids,
        "target_input_ids": [tokenizer.bos_id] + target_ids,
        "target_output_ids": target_ids + [tokenizer.eos_id],
    }


def process_file(
    input_path: Path,
    output_path: Path,
    tokenizer: SentencePieceTokenizer,
) -> None:
    """
    将一个 interim JSONL 文件编码为 processed JSONL

    Args:
        input_path: 输入文本 JSONL 文件路径
        output_path: 输出 token id JSONL 文件路径
        tokenizer: SentencePiece 分词器
    """
    total_count = 0
    valid_count = 0
    invalid_count = 0

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        # 每读取一行就完成解析、SentencePiece 编码和 JSONL 写入,避免额外副本.
        for line in tqdm(
            input_file,
            desc=f"encode {input_path.stem}",
            unit="line",
            dynamic_ncols=True,
        ):
            total_count += 1
            item = load_jsonl_item(line)

            # JSON 解析失败,跳过该行
            if item is None:
                invalid_count += 1
                continue

            english_text = str(item.get("en", "")).strip()
            chinese_text = str(item.get("zh", "")).strip()

            # 任一文本为空,跳过该句对
            if not english_text or not chinese_text:
                invalid_count += 1
                continue

            # 将句对编码为 token id 并写入 processed 文件
            encoded_item = encode_translation_pair(
                english_text,
                chinese_text,
                tokenizer,
            )
            output_file.write(json.dumps(encoded_item, ensure_ascii=False) + "\n")
            valid_count += 1

    # 输出处理统计信息
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"总样本数: {total_count}")
    print(f"有效样本数: {valid_count}")
    print(f"无效样本数: {invalid_count}")
    print("-" * 60)


def process_all_splits(tokenizer: SentencePieceTokenizer) -> None:
    """
    编码 train / validation / test 三个数据切分

    Args:
        tokenizer: SentencePiece 分词器
    """
    # 训练集
    process_file(
        paths.INTERIM_TRAIN_DATASET_PATH,
        paths.PROCESSED_TRAIN_DATASET_PATH,
        tokenizer,
    )
    # 验证集
    process_file(
        paths.INTERIM_VAL_DATASET_PATH,
        paths.PROCESSED_VAL_DATASET_PATH,
        tokenizer,
    )
    # 测试集
    process_file(
        paths.INTERIM_TEST_DATASET_PATH,
        paths.PROCESSED_TEST_DATASET_PATH,
        tokenizer,
    )


def main() -> None:
    """主函数:生成 processed 数据集"""
    # 加载训练好的 SentencePiece 分词器
    tokenizer = SentencePieceTokenizer(paths.TOKENIZER_MODEL_PATH)
    process_all_splits(tokenizer)


if __name__ == "__main__":
    main()
