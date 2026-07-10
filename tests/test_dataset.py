"""
数据集与 batch 整理测试.

测试 EncodedTranslationDataset 的 JSONL 加载能力及 collate_batch 的
填充(padding)行为.验证:
1. 数据集能正确解析 JSONL 格式的样本文件
2. collate_batch 能将不等长序列填充至统一长度
3. 填充值(pad_id)正确应用于源序列和目标序列
"""

import json

import torch

from src.data.dataloader import EncodedTranslationDataset, collate_batch


def test_dataset_loads_jsonl_and_collates_padding(tmp_path) -> None:
    """
    测试从 JSONL 文件加载数据集并对不等长样本进行 padding 整理.

    验证点:
    - 数据集长度与样本数一致
    - batch 张量形状为 (batchSize, maxSeqLen)
    - 较短序列末尾用 pad_id 填充
    """
    # 使用临时目录创建测试用 JSONL 文件
    dataset_path = tmp_path / "samples.jsonl"
    samples = [
        {
            "source_ids": [4, 3],  # 长度为 2 的源序列
            "target_input_ids": [2, 5],  # 长度为 2 的目标输入序列
            "target_output_ids": [5, 3],  # 长度为 2 的目标输出序列
        },
        {
            "source_ids": [6, 7, 3],  # 长度为 3 的源序列(比第一条长)
            "target_input_ids": [2, 8, 9],  # 长度为 3 的目标输入序列
            "target_output_ids": [8, 9, 3],  # 长度为 3 的目标输出序列
        },
    ]
    # 将样本写入 JSONL 文件,每行一个 JSON 对象
    dataset_path.write_text(
        "\n".join(json.dumps(sample) for sample in samples), encoding="utf-8"
    )

    # 从 JSONL 文件创建数据集实例
    dataset = EncodedTranslationDataset(dataset_path)
    # 将前两个样本整理为一个 batch,pad_id=0 表示用 0 填充
    batch = collate_batch([dataset[0], dataset[1]], pad_id=0)

    # 验证数据集包含 2 个样本
    assert len(dataset) == 2
    # 验证 batch 形状:2 个样本,最大序列长度 3
    assert batch["source_ids"].shape == (2, 3)
    # 验证第一条短样本的源序列末尾被填充了 0
    assert torch.equal(batch["source_ids"][0], torch.tensor([4, 3, 0]))
    # 验证第一条短样本的目标输出序列末尾被填充了 0
    assert torch.equal(batch["target_output_ids"][0], torch.tensor([5, 3, 0]))
