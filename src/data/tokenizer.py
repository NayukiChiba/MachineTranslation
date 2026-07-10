"""
SentencePiece 分词器训练与加载模块

功能:
1. 从 interim 训练集构建 tokenizer 训练语料
2. 训练中英文共享的 SentencePiece BPE 分词器
3. 提供 encode / decode 接口,供 Dataset 和推理流程复用

使用方法:
    uv run python -m src.data.tokenizer
"""

from pathlib import Path
from typing import Iterable

import sentencepiece as spm
from tqdm.auto import tqdm

from configs import paths
from configs.defaults import TokenizerConfig
from src.data.utils import load_jsonl_item


def iter_parallel_texts(input_path: Path) -> Iterable[str]:
    """
    遍历平行语料中的中英文文本

    Args:
        input_path: 输入 JSONL 文件路径

    Yields:
        str: 单条文本,英文和中文会分别产出
    """
    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            # 将 JSON 行解析为字典
            item = load_jsonl_item(line)

            # 跳过解析失败的行
            if item is None:
                continue

            # 提取英文和中文文本,缺失时默认为空字符串
            english_text = str(item.get("en", "")).strip()
            chinese_text = str(item.get("zh", "")).strip()

            # 分别产出非空文本,确保两种语言都参与 BPE 训练
            if english_text:
                yield english_text
            if chinese_text:
                yield chinese_text


def build_tokenizer_corpus(input_path: Path, output_path: Path) -> None:
    """
    构建 SentencePiece 训练语料文件

    Args:
        input_path: 清洗后的训练集 JSONL 路径
        output_path: 输出纯文本语料路径
    """
    line_count = 0

    # 将平行语料写入纯文本文件,SentencePiece 按行读取训练
    with output_path.open("w", encoding="utf-8") as output_file:
        # 每个平行句对会产出英文和中文两行,供共享 BPE 词表联合学习.
        for text in tqdm(
            iter_parallel_texts(input_path),
            desc="build tokenizer corpus",
            unit="line",
            dynamic_ncols=True,
        ):
            output_file.write(text + "\n")
            line_count += 1

    # 输出语料统计信息
    print(f"分词器训练语料: {output_path}")
    print(f"语料行数: {line_count}")


def train_sentencepiece_tokenizer(corpus_path: Path, model_prefix_path: Path) -> None:
    """
    训练 SentencePiece 分词器

    Args:
        corpus_path: SentencePiece 训练语料路径
        model_prefix_path: 输出模型前缀路径
    """
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix_path),
        # 词表与模型类型
        vocab_size=TokenizerConfig.vocab_size,
        model_type=TokenizerConfig.model_type,
        character_coverage=TokenizerConfig.character_coverage,
        # 训练数据控制
        input_sentence_size=TokenizerConfig.input_sentence_size,
        shuffle_input_sentence=True,
        # 特殊 token id 映射
        pad_id=TokenizerConfig.pad_id,
        unk_id=TokenizerConfig.unknown_id,
        bos_id=TokenizerConfig.begin_of_sentence_id,
        eos_id=TokenizerConfig.end_of_sentence_id,
        # 特殊 token 文本表示
        pad_piece=TokenizerConfig.pad_token,
        unk_piece=TokenizerConfig.unknown_token,
        bos_piece=TokenizerConfig.begin_of_sentence_token,
        eos_piece=TokenizerConfig.end_of_sentence_token,
    )

    # 输出训练结果路径
    print(f"分词器模型: {paths.TOKENIZER_MODEL_PATH}")
    print(f"分词器词表: {paths.TOKENIZER_VOCAB_PATH}")


class SentencePieceTokenizer:
    """SentencePiece 分词器封装"""

    def __init__(self, model_path: Path) -> None:
        """
        初始化分词器

        Args:
            model_path: SentencePiece 模型路径
        """
        # 创建 SentencePiece 处理器并加载预训练模型
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

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        将文本编码为 token id

        Args:
            text: 输入文本
            add_bos: 是否添加句子开始 token
            add_eos: 是否添加句子结束 token

        Returns:
            list[int]: token id 列表
        """
        # 调用底层 SentencePiece 编码器
        token_ids = self.processor.encode(text, out_type=int)

        # 按需在序列首尾插入特殊 token
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        将 token id 解码为文本

        Args:
            token_ids: token id 列表

        Returns:
            str: 解码后的文本
        """
        return self.processor.decode(token_ids)


def train_tokenizer() -> None:
    """训练 OPUS-100 英中共享 SentencePiece 分词器"""
    # 步骤1:从平行语料构建纯文本训练文件
    build_tokenizer_corpus(
        paths.INTERIM_TRAIN_DATASET_PATH,
        paths.TOKENIZER_CORPUS_PATH,
    )
    # 步骤2:使用 SentencePiece 训练 BPE 模型
    train_sentencepiece_tokenizer(
        paths.TOKENIZER_CORPUS_PATH,
        paths.TOKENIZER_MODEL_PREFIX_PATH,
    )

    # 步骤3:加载训练好的分词器,输出关键参数验证
    tokenizer = SentencePieceTokenizer(paths.TOKENIZER_MODEL_PATH)
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"pad_id: {tokenizer.pad_id}")
    print(f"unk_id: {tokenizer.unk_id}")
    print(f"bos_id: {tokenizer.bos_id}")
    print(f"eos_id: {tokenizer.eos_id}")


def main() -> None:
    """主函数:训练分词器"""
    train_tokenizer()


if __name__ == "__main__":
    main()
