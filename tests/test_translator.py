"""
翻译器贪心解码流程测试.

测试目标:
1. 验证 Translator 的端到端翻译流程(编码 -> 自回归解码 -> 停 EOS)
2. 使用 FakeTokenizer 和 FakeAutoregressiveModel 作为轻量级替身,
   避免依赖真实模型权重,确保单元测试快速可靠
"""

import torch
import torch.nn as nn

from src.inference.translator import Translator


class FakeTokenizer:
    """满足推理协议的最小分词器"""

    pad_id = 0
    bos_id = 2
    eos_id = 3

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        # 固定编码:忽略输入文本,始终返回 token [5] 作为占位 ID
        token_ids = [5]
        if add_bos:
            token_ids.insert(0, self.bos_id)  # 可选添加 BOS 前缀
        if add_eos:
            token_ids.append(self.eos_id)  # 可选添加 EOS 后缀
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        # token 4 映射为"你",其余 token 映射为"?",构成简化的中文解码逻辑
        return "".join({4: "你"}.get(token_id, "?") for token_id in token_ids)


class FakeAutoregressiveModel(nn.Module):
    """第一步生成 token 4,第二步生成 eos"""

    def encode(
        self, source_ids: torch.Tensor, source_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # 返回全零编码器输出,形状为 (batch, seq_len, d_model=4)
        return torch.zeros(source_ids.size(0), source_ids.size(1), 4)

    def decode(
        self,
        target_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        # 构造全零 logits,形状为 (batch, seq_len, vocab_size=8)
        logits = torch.zeros(target_ids.size(0), target_ids.size(1), 8)
        # 第 1 步(BOS 之后)预测 token 4,第 2 步预测 EOS(token 3)
        next_token = 4 if target_ids.size(1) == 1 else 3
        # 将对应位置 logit 设为高值,确保贪婪解码选中正确 token
        logits[:, -1, next_token] = 10.0
        return logits


def test_translator_stops_at_eos() -> None:
    """验证 Translator 在遇到 EOS token 时正确停止解码并返回译文"""
    # 构造 Translator,使用伪造模型和分词器,避免加载真实权重
    translator = Translator(
        model=FakeAutoregressiveModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        max_generation_length=5,
        show_progress=False,
    )

    # 预期流程:BOS -> token 4("你") -> EOS(停止) -> decode 返回 "你"
    assert translator.translate("hello") == "你"
