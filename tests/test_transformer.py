"""
Transformer 模型单元测试

功能:
1. 验证 Transformer 模型前向传播的形状正确性
2. 验证编码器-解码器分离调用的形状正确性
3. 验证模型配置的序列化与反序列化一致性
4. 验证输出张量数值稳定性(无 NaN/Inf)

设计思路:
- 使用极小的词汇表(32)和模型维度(16)以加速测试
- encoder_num_layers/decoder_num_layers 设为 1,减少计算开销
- dropout 设为 0.0,确保前向传播结果确定性
- 源序列和目标序列中包含填充 token(pad_id=0),验证填充处理
"""

import torch

from src.model.transformer import Transformer


def test_transformer_forward_and_config_round_trip() -> None:
    """
    验证 Transformer 前向传播与配置往返一致性

    测试内容:
    1. 前向传播输出形状:(batch, target_len, vocab_size)
    2. 分离的编码-解码调用与合并调用输出形状一致
    3. from_config 恢复的模型配置与原模型完全一致
    4. 输出 logits 中不含 NaN 或 Inf
    """
    # 构建小型 Transformer 模型,用于快速验证
    model = Transformer(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        d_feedforward=32,
        encoder_num_layers=1,
        decoder_num_layers=1,
        dropout=0.0,
        max_seq_length=16,
        pad_id=0,
    )
    # 构造含填充 token 的批次数据(batch_size=2)
    source_ids = torch.tensor([[4, 5, 3, 0], [6, 3, 0, 0]])
    target_ids = torch.tensor([[2, 7, 8], [2, 9, 0]])

    # 完整前向传播:编码器 + 解码器
    logits = model(source_ids, target_ids)
    # 分离调用:手动编码后再解码
    encoder_output = model.encode(source_ids)
    decoded_logits = model.decode(target_ids, encoder_output)
    # 通过配置重建模型,验证往返一致性
    restored_model = Transformer.from_config(model.config)

    # 验证输出形状:batch=2, target_len=3, vocab_size=32
    assert logits.shape == (2, 3, 32)
    assert decoded_logits.shape == (2, 3, 32)
    # 验证配置往返:重建后的模型配置应与原配置相同
    assert restored_model.config == model.config
    # 验证数值稳定性:所有 logits 必须是有限值
    assert torch.isfinite(logits).all()
