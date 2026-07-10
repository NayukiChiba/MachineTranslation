"""
Transformer 贪心解码与 checkpoint 加载接口

翻译过程先执行一次 Encoder,再从 <bos> 开始逐 token 调用 Decoder.每一步只
选择 logits 最大的 token,因此结果是确定性的贪心解码,不包含随机采样.
"""

from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from configs import paths
from configs.defaults import InferenceConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.model.mask import create_causal_mask
from src.model.transformer import Transformer


class TokenizerProtocol(Protocol):
    """推理所需的最小分词器接口,便于替换实现和编写测试."""

    @property
    def pad_id(self) -> int:
        """填充 token 对应的 id,用于构建 padding mask,不参与实际翻译."""
        ...

    @property
    def bos_id(self) -> int:
        """句子起始 token 对应的 id,每条目标序列都以该 id 开头."""
        ...

    @property
    def eos_id(self) -> int:
        """句子结束 token 对应的 id,指示解码终止,本身不写入输出文本."""
        ...

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """将文本编码为 token id 列表,可选是否追加 bos/eos 特殊标记."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """将 token id 列表还原为可读文本字符串."""
        ...


class Translator:
    """使用已加载模型执行单句或批量贪心翻译."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerProtocol,
        device: torch.device | str = "cpu",
        max_generation_length: int = InferenceConfig.max_generation_length,
        show_progress: bool = InferenceConfig.show_progress,
    ) -> None:
        """
        初始化翻译器,配置模型、分词器与解码策略参数.

        Args:
            model: 已训练的 Transformer 模型实例.
            tokenizer: 分词器,负责文本与 token id 之间的双向转换.
            device: 推理设备,支持 "cpu"、"cuda"、"cuda:0" 等形式.
            max_generation_length: 单次翻译生成的最大 token 数(不含 bos),
                                   超过该上限仍未遇到 eos 则强制终止.
            show_progress: 是否在 tqdm 中显示解码进度条.

        Raises:
            ValueError: 当 max_generation_length 非正时抛出.
        """
        # 最大长度必须为正,否则循环无法生成任何目标 token.
        if max_generation_length <= 0:
            raise ValueError("max_generation_length 必须大于 0")
        # 统一转换设备表示,支持 cpu、cuda 和 cuda:0 等形式.
        self.device = torch.device(device)
        # 模型权重与后续创建的 source/target Tensor 必须在同一设备.
        self.model = model.to(self.device)
        # eval 关闭 Dropout,保证同一句输入得到稳定输出.
        self.model.eval()
        # tokenizer 负责文本和共享 SentencePiece token id 之间的转换.
        self.tokenizer = tokenizer
        # 该上限防止模型未生成 eos 时无限循环.
        self.max_generation_length = max_generation_length
        # 测试、批处理或日志重定向时可关闭动态进度条.
        self.show_progress = show_progress

    def translate(self, text: str) -> str:
        """把单句英文翻译为中文字符串."""
        # 去除首尾空白,避免把只有空格的文本送入 SentencePiece.
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("待翻译文本不能为空")

        # source 末尾添加 <eos>,让 Encoder 能识别输入边界.
        source_tokens = self.tokenizer.encode(normalized_text, add_eos=True)
        # 增加 batch 维后 shape=(1, source_length),dtype 必须是 long.
        source_ids = torch.tensor(
            [source_tokens],
            dtype=torch.long,
            device=self.device,
        )
        # shape=(1, source_length),True 位置会被注意力忽略.
        source_padding_mask = source_ids.eq(self.tokenizer.pad_id)

        # 推理不需要梯度,no_grad 可减少显存占用并加快矩阵计算.
        with torch.no_grad():
            # Encoder 只执行一次,输出 shape=(1, source_length, d_model).
            encoder_output = self.model.encode(source_ids, source_padding_mask)
            # Decoder 输入从 <bos> 开始,后续每轮追加一个预测 token.
            generated_ids = [self.tokenizer.bos_id]

            # 每次循环最多生成一个 token,因此进度单位使用 token.
            generation_steps = tqdm(
                range(self.max_generation_length),
                total=self.max_generation_length,
                desc="translate",
                unit="token",
                dynamic_ncols=True,
                leave=False,
                disable=not self.show_progress,
            )
            for _ in generation_steps:
                # shape=(1, generated_length),长度随解码轮数逐步增加.
                target_ids = torch.tensor(
                    [generated_ids],
                    dtype=torch.long,
                    device=self.device,
                )
                # 当前序列通常不含 pad,但仍传入标准二维 padding mask.
                target_padding_mask = target_ids.eq(self.tokenizer.pad_id)
                # 原始 helper 返回 (1,1,L,L),MultiheadAttention 需要二维 (L,L).
                target_causal_mask = (
                    create_causal_mask(target_ids.size(1), device=self.device)
                    .squeeze(0)
                    .squeeze(0)
                )
                # logits shape=(1, generated_length, vocab_size).
                logits = self.model.decode(
                    target_ids=target_ids,
                    encoder_output=encoder_output,
                    target_padding_mask=target_padding_mask,
                    target_causal_mask=target_causal_mask,
                    source_padding_mask=source_padding_mask,
                )
                # 只读取最后一个位置,并在 vocab_size 维选择最大 logit.
                next_token_id = int(logits[0, -1].argmax(dim=-1).item())
                # eos 表示目标句子结束,不把 eos 本身加入待解码文本.
                if next_token_id == self.tokenizer.eos_id:
                    break
                # 追加预测后,下一轮 target_length 增加 1.
                generated_ids.append(next_token_id)
                # postfix 显示已生成的普通 token 数,不包含开头的 bos.
                generation_steps.set_postfix(length=len(generated_ids) - 1)

        # 去掉开头 bos,并防御性过滤 pad/eos,得到纯目标文本 token.
        output_ids = [
            token_id
            for token_id in generated_ids[1:]
            if token_id not in {self.tokenizer.pad_id, self.tokenizer.eos_id}
        ]
        # SentencePiece 根据 token id 恢复中文文本和子词边界.
        return self.tokenizer.decode(output_ids)

    def translate_batch(self, texts: list[str]) -> list[str]:
        """复用同一模型依次翻译多条文本."""
        return [self.translate(text) for text in texts]


def load_translator(
    checkpoint_path: Path = paths.BEST_MODEL_PATH,
    tokenizer_path: Path = paths.TOKENIZER_MODEL_PATH,
    device: torch.device | str = "cpu",
    max_generation_length: int = InferenceConfig.max_generation_length,
    show_progress: bool = InferenceConfig.show_progress,
) -> Translator:
    """从统一路径加载模型、分词器并创建 Translator."""
    # 统一路径类型,并在 torch.load 前给出明确的缺失文件错误.
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    # map_location 支持把 GPU checkpoint 加载到 CPU;weights_only 限制反序列化范围.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    # tokenizer.model 定义实际词表、特殊 token id 和子词规则.
    tokenizer = SentencePieceTokenizer(Path(tokenizer_path))
    # 新 checkpoint 在 config.model 中保存完整 Transformer 构造参数.
    model_config = checkpoint.get("config", {}).get("model", {})
    if model_config:
        # 使用训练时结构重建模型,确保 state_dict 中每个参数形状匹配.
        model = Transformer.from_config(model_config)
    else:
        # 兼容旧 checkpoint,至少使用 tokenizer 的真实词表大小.
        model = Transformer(vocab_size=tokenizer.vocab_size)
    # 将 checkpoint 参数复制到刚创建的模型模块中.
    model.load_state_dict(checkpoint["model_state_dict"])
    # 返回配置完整的翻译器,调用者无需再处理设备或 eval 模式.
    return Translator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_generation_length=max_generation_length,
        show_progress=show_progress,
    )
