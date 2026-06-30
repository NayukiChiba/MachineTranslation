"""
机器翻译 DataLoader 模块

功能：
1. 读取已经编码好的 processed JSONL
2. 在 batch 级别完成 padding
3. 导出 train / validation / test 三个 DataLoader

注意：
    本模块不生成 mask。
    create_dataloaders 默认会检查数据产物是否存在。
    如果缺失，会自动执行 raw -> interim -> tokenizer -> processed 管线。
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from configs import paths
from configs.defaults import DataLoaderConfig, TokenizerConfig
from src.data.utils import load_jsonl_item


def files_exist(file_paths: list[Path]) -> bool:
    """
    判断一组文件是否都存在且非空

    Args:
        file_paths: 文件路径列表

    Returns:
        bool: 所有文件存在且非空时返回 True
    """
    return all(
        file_path.exists() and file_path.stat().st_size > 0 for file_path in file_paths
    )


def prepare_data_pipeline(force: bool = False) -> None:
    """
    准备机器翻译训练所需的完整数据管线

    Args:
        force: 是否强制重新生成所有数据产物

    注意：
        第一次运行会比较慢，因为需要下载数据、清洗数据、训练分词器、
        生成 processed 数据。之后只要文件存在，就会直接跳过。
    """
    raw_paths = [
        paths.RAW_TRAIN_DATASET_PATH,
        paths.RAW_VAL_DATASET_PATH,
        paths.RAW_TEST_DATASET_PATH,
    ]
    interim_paths = [
        paths.INTERIM_TRAIN_DATASET_PATH,
        paths.INTERIM_VAL_DATASET_PATH,
        paths.INTERIM_TEST_DATASET_PATH,
    ]
    tokenizer_paths = [
        paths.TOKENIZER_MODEL_PATH,
        paths.TOKENIZER_VOCAB_PATH,
    ]
    processed_paths = [
        paths.PROCESSED_TRAIN_DATASET_PATH,
        paths.PROCESSED_VAL_DATASET_PATH,
        paths.PROCESSED_TEST_DATASET_PATH,
    ]

    if force or not files_exist(raw_paths):
        print("准备 raw 数据集")
        from src.data.download import download_raw_dataset

        download_raw_dataset()

    if force or not files_exist(interim_paths):
        print("准备 interim 数据集")
        from src.data.interim import create_interim_splits

        create_interim_splits()

    if force or not files_exist(tokenizer_paths):
        print("准备 SentencePiece 分词器")
        from src.data.tokenizer import train_tokenizer

        train_tokenizer()

    if force or not files_exist(processed_paths):
        print("准备 processed 数据集")
        from src.data.process import process_all_splits
        from src.data.tokenizer import SentencePieceTokenizer

        tokenizer = SentencePieceTokenizer(paths.TOKENIZER_MODEL_PATH)
        process_all_splits(tokenizer)


def load_samples(file_path: Path) -> list[dict[str, list[int]]]:
    """
    加载 processed JSONL 样本

    Args:
        file_path: processed JSONL 文件路径

    Returns:
        list[dict[str, list[int]]]: 已经编码好的样本列表
    """
    samples: list[dict[str, list[int]]] = []

    with file_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            item = load_jsonl_item(line)

            if item is None:
                continue

            samples.append(
                {
                    "source_ids": list(item["source_ids"]),
                    "target_input_ids": list(item["target_input_ids"]),
                    "target_output_ids": list(item["target_output_ids"]),
                }
            )

    return samples


def pad_token_ids(token_ids: list[int], max_length: int, pad_id: int) -> list[int]:
    """
    将 token id 序列 padding 到 batch 内最大长度

    Args:
        token_ids: token id 列表
        max_length: batch 内最大长度
        pad_id: padding token id

    Returns:
        list[int]: padding 后的 token id 列表
    """
    pad_count = max_length - len(token_ids)
    return token_ids + [pad_id] * pad_count


class EncodedTranslationDataset(Dataset):
    """已经编码好的英中翻译数据集"""

    def __init__(self, file_path: Path) -> None:
        """
        初始化数据集

        Args:
            file_path: processed JSONL 文件路径
        """
        self.samples = load_samples(file_path)

    def __len__(self) -> int:
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """
        获取单条样本

        Args:
            index: 样本索引

        Returns:
            dict[str, list[int]]: 已经编码好的 source 和 target
        """
        return self.samples[index]


def collate_batch(
    batch: list[dict[str, list[int]]],
    pad_id: int,
) -> dict[str, Tensor]:
    """
    整理一个 batch

    Args:
        batch: 样本列表
        pad_id: padding token id

    Returns:
        dict[str, Tensor]: padding 后的 batch
    """
    max_source_length = max(len(item["source_ids"]) for item in batch)
    max_target_length = max(len(item["target_input_ids"]) for item in batch)

    source_ids = torch.tensor(
        [
            pad_token_ids(item["source_ids"], max_source_length, pad_id)
            for item in batch
        ],
        dtype=torch.long,
    )
    target_input_ids = torch.tensor(
        [
            pad_token_ids(item["target_input_ids"], max_target_length, pad_id)
            for item in batch
        ],
        dtype=torch.long,
    )
    target_output_ids = torch.tensor(
        [
            pad_token_ids(item["target_output_ids"], max_target_length, pad_id)
            for item in batch
        ],
        dtype=torch.long,
    )

    return {
        "source_ids": source_ids,
        "target_input_ids": target_input_ids,
        "target_output_ids": target_output_ids,
    }


def create_dataloader(
    file_path: Path,
    batch_size: int = DataLoaderConfig.BATCH_SIZE,
    pad_id: int = TokenizerConfig.PAD_ID,
    shuffle: bool = False,
    num_workers: int = DataLoaderConfig.NUM_WORKERS,
) -> DataLoader:
    """
    创建单个 DataLoader

    Args:
        file_path: processed JSONL 文件路径
        batch_size: batch 大小
        pad_id: padding token id
        shuffle: 是否打乱数据
        num_workers: DataLoader worker 数量

    Returns:
        DataLoader: 单个数据切分的 DataLoader
    """
    dataset = EncodedTranslationDataset(file_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )


def create_dataloaders(
    batch_size: int = DataLoaderConfig.BATCH_SIZE,
    pad_id: int = TokenizerConfig.PAD_ID,
    num_workers: int = DataLoaderConfig.NUM_WORKERS,
    auto_prepare: bool = DataLoaderConfig.AUTO_PREPARE,
    force_prepare: bool = DataLoaderConfig.FORCE_PREPARE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 train / validation / test 三个 DataLoader

    Args:
        batch_size: batch 大小
        pad_id: padding token id
        num_workers: DataLoader worker 数量
        auto_prepare: 是否自动准备缺失的数据产物
        force_prepare: 是否强制重新生成数据产物

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: 训练、验证、测试 DataLoader
    """
    if auto_prepare:
        prepare_data_pipeline(force=force_prepare)

    train_dataloader = create_dataloader(
        paths.PROCESSED_TRAIN_DATASET_PATH,
        batch_size=batch_size,
        pad_id=pad_id,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_dataloader = create_dataloader(
        paths.PROCESSED_VAL_DATASET_PATH,
        batch_size=batch_size,
        pad_id=pad_id,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataloader = create_dataloader(
        paths.PROCESSED_TEST_DATASET_PATH,
        batch_size=batch_size,
        pad_id=pad_id,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, validation_dataloader, test_dataloader


def main() -> None:
    """主函数：检查 DataLoader 输出形状"""
    _, validation_dataloader, _ = create_dataloaders()
    batch = next(iter(validation_dataloader))

    for name, value in batch.items():
        print(f"{name}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
