"""
SentencePiece 词表管理模块

功能：
1. 从 tokenizer.model 加载 token 与 id 的对应关系
2. 提供 token2id / id2token 查询接口
3. 提供特殊 token id 属性
4. 使用 tokenizer.vocab 导出可读 JSON，便于人工检查词表

说明：
    本项目使用 SentencePiece 作为分词器，因此不再手写 vocab.json
    作为训练入口。vocabulary.py 只提供词表查询和检查能力。
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

    with vocab_path.open("r", encoding="utf-8") as vocab_file:
        for token_id, line in enumerate(vocab_file):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")
            token = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0

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

    token2id = {str(item["token"]): int(item["id"]) for item in vocab_items}
    id2token = {str(item["id"]): str(item["token"]) for item in vocab_items}
    scores = {str(item["token"]): float(item["score"]) for item in vocab_items}

    return {
        "token2id": token2id,
        "id2token": id2token,
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
        self.processor.load(str(model_path))

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

    def token2id(self, token: str) -> int:
        """
        查询 token 对应的 id

        Args:
            token: token 文本

        Returns:
            int: token id，不存在时返回 unk_id
        """
        return self.processor.piece_to_id(token)

    def id2token(self, token_id: int) -> str:
        """
        查询 id 对应的 token

        Args:
            token_id: token id

        Returns:
            str: token 文本
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id 超出词表范围: {token_id}")

        return self.processor.id_to_piece(token_id)

    def to_token2id(self) -> dict[str, int]:
        """
        导出 token 到 id 的映射

        Returns:
            dict[str, int]: token -> id 映射
        """
        return {
            self.id2token(token_id): token_id for token_id in range(self.vocab_size)
        }

    def to_id2token(self) -> dict[str, str]:
        """
        导出 id 到 token 的映射

        Returns:
            dict[str, str]: id -> token 映射，key 使用字符串便于保存 JSON
        """
        return {
            str(token_id): self.id2token(token_id)
            for token_id in range(self.vocab_size)
        }

    def save_token2id_json(self, output_path: Path) -> None:
        """
        保存 token -> id 映射为 JSON 文件

        Args:
            output_path: 输出 JSON 文件路径
        """
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                self.to_token2id(),
                output_file,
                ensure_ascii=False,
                indent=2,
            )


def main() -> None:
    """打印词表基础信息"""
    vocabulary = Vocabulary(paths.TOKENIZER_MODEL_PATH)
    save_sentencepiece_vocab_json(
        paths.TOKENIZER_VOCAB_PATH,
        paths.TOKENIZER_VOCAB_JSON_PATH,
    )

    print(f"词表大小: {vocabulary.vocab_size}")
    print(f"pad_id: {vocabulary.pad_id}")
    print(f"unk_id: {vocabulary.unk_id}")
    print(f"bos_id: {vocabulary.bos_id}")
    print(f"eos_id: {vocabulary.eos_id}")

    print("前 20 个 token:")
    for token_id in range(min(20, vocabulary.vocab_size)):
        token = vocabulary.id2token(token_id)
        print(f"{token_id}: {token}")

    print(f"词表 JSON: {paths.TOKENIZER_VOCAB_JSON_PATH}")


if __name__ == "__main__":
    main()
