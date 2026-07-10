"""
SentencePiece 词表管理模块

功能:
1. 从 tokenizer.model 加载 token 与 id 的对应关系
2. 提供 token_to_id / id_to_token 查询接口
3. 提供特殊 token id 属性
4. 使用 tokenizer.vocab 导出可读 JSON,便于人工检查词表

说明:
    本项目使用 SentencePiece 作为分词器,因此不再手写 vocab.json
    作为训练入口.vocabulary.py 只提供词表查询和检查能力.
"""

import json
from pathlib import Path

import sentencepiece as spm

from configs import paths


def load_sentencepiece_vocab(vocab_path: Path) -> list[dict[str, float | int | str]]:
    """
    读取 SentencePiece 生成的 tokenizer.vocab 文件

    Args:
        vocab_path: tokenizer.vocab 文件路径

    Returns:
        list[dict[str, float | int | str]]: 带 id、token、score 的词表列表
    """
    vocab_items: list[dict[str, float | int | str]] = []

    # 逐行读取 SentencePiece vocab 文件,行号即为 token id
    with vocab_path.open("r", encoding="utf-8") as vocab_file:
        for token_id, line in enumerate(vocab_file):
            line = line.rstrip("\n")  # 去除行尾换行符

            if not line:
                continue  # 跳过空行

            # vocab 文件格式: token\t score (tab 分隔)
            parts = line.split("\t")
            token = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0  # 容错:无分数时默认为 0

            vocab_items.append(
                {
                    "id": token_id,
                    "token": token,
                    "score": score,
                }
            )

    return vocab_items


def build_vocab_json_data(vocab_path: Path) -> dict[str, object]:
    """
    构建双向词表 JSON 数据

    Args:
        vocab_path: tokenizer.vocab 文件路径

    Returns:
        dict[str, object]: 包含 token2id、id2token、scores、items 的词表数据
    """
    vocab_items = load_sentencepiece_vocab(vocab_path)

    # 构建双向映射:token -> id,id -> token,以及 token -> score
    token_to_id = {str(item["token"]): int(item["id"]) for item in vocab_items}
    id_to_token = {str(item["id"]): str(item["token"]) for item in vocab_items}
    scores = {str(item["token"]): float(item["score"]) for item in vocab_items}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "scores": scores,
        "items": vocab_items,
    }


def save_sentencepiece_vocab_json(vocab_path: Path, output_path: Path) -> None:
    """
    将 tokenizer.vocab 转换为 JSON 文件

    Args:
        vocab_path: tokenizer.vocab 文件路径
        output_path: 输出 JSON 文件路径
    """
    vocab_data = build_vocab_json_data(vocab_path)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(vocab_data, output_file, ensure_ascii=False, indent=2)


class Vocabulary:
    """基于 SentencePiece 模型的词表视图"""

    def __init__(self, model_path: Path) -> None:
        """
        初始化词表

        Args:
            model_path: SentencePiece tokenizer.model 文件路径
        """
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(
            str(model_path)
        )  # SentencePiece 要求传入字符串路径而非 Path 对象

    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return self.processor.get_piece_size()

    @property
    def pad_id(self) -> int:
        """padding token id"""
        return self.processor.pad_id()

    @property
    def unk_id(self) -> int:
        """unknown token id"""
        return self.processor.unk_id()

    @property
    def bos_id(self) -> int:
        """begin-of-sentence token id"""
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        """end-of-sentence token id"""
        return self.processor.eos_id()

    def token_to_id(self, token: str) -> int:
        """
        查询 token 对应的 id

        Args:
            token: token 文本

        Returns:
            int: token id,不存在时返回 unk_id
        """
        return self.processor.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """
        查询 id 对应的 token

        Args:
            token_id: token id

        Returns:
            str: token 文本
        """
        # 边界检查:确保 token_id 在词表范围内
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id 超出词表范围: {token_id}")

        return self.processor.id_to_piece(token_id)

    def to_token_to_id(self) -> dict[str, int]:
        """
        导出 token 到 id 的映射

        Returns:
            dict[str, int]: token -> id 映射
        """
        # 遍历词表内所有 id,构建 token -> id 映射
        return {
            self.id_to_token(token_id): token_id for token_id in range(self.vocab_size)
        }

    def to_id_to_token(self) -> dict[str, str]:
        """
        导出 id 到 token 的映射

        Returns:
            dict[str, str]: id -> token 映射,key 使用字符串便于保存 JSON
        """
        # key 使用字符串格式的 id,便于直接保存为 JSON
        return {
            str(token_id): self.id_to_token(token_id)
            for token_id in range(self.vocab_size)
        }

    def save_token_to_id_json(self, output_path: Path) -> None:
        """
        保存 token -> id 映射为 JSON 文件

        Args:
            output_path: 输出 JSON 文件路径
        """
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                self.to_token_to_id(),
                output_file,
                ensure_ascii=False,
                indent=2,
            )


def main() -> None:
    """打印词表基础信息"""
    # 加载 SentencePiece 模型并构建词表视图
    vocabulary = Vocabulary(paths.TOKENIZER_MODEL_PATH)
    # 将 tokenizer.vocab 导出为可读 JSON
    save_sentencepiece_vocab_json(
        paths.TOKENIZER_VOCAB_PATH,
        paths.TOKENIZER_VOCAB_JSON_PATH,
    )

    # 打印词表统计信息
    print(f"词表大小: {vocabulary.vocab_size}")
    print(f"pad_id: {vocabulary.pad_id}")
    print(f"unk_id: {vocabulary.unk_id}")
    print(f"bos_id: {vocabulary.bos_id}")
    print(f"eos_id: {vocabulary.eos_id}")

    # 打印前 20 个 token 用于快速检查
    print("前 20 个 token:")
    for token_id in range(min(20, vocabulary.vocab_size)):
        token = vocabulary.id_to_token(token_id)
        print(f"{token_id}: {token}")

    print(f"词表 JSON: {paths.TOKENIZER_VOCAB_JSON_PATH}")


if __name__ == "__main__":
    main()
