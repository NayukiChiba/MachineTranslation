"""
独立模型评估流程

Evaluator 不导入 Trainer、优化器、调度器或 checkpoint.它只接收一个模型和
一个可迭代 batch 数据源,在 no_grad 模式下计算 loss、困惑度和 token accuracy.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm

from configs.defaults import TokenizerConfig


class Evaluator:
    """在 DataLoader 上计算 loss、困惑度和 token accuracy."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        pad_id: int = TokenizerConfig.pad_id,
        criterion: nn.Module | None = None,
    ) -> None:
        """
        初始化评估器

        Args:
            model: 待评估的翻译模型(通常是 Transformer 实例)
            device: 评估所用设备,支持字符串或 torch.device 对象
            pad_id: padding token 的 ID,用于排除补齐位置对指标的影响
            criterion: 自定义损失函数;为 None 时默认使用忽略 pad 的交叉熵
        """
        # 统一转换为 torch.device,后续 Tensor.to 可以直接复用.
        self.device = torch.device(device)
        # 模型参数和输入 Tensor 必须位于同一设备.
        self.model = model.to(self.device)
        # pad_id 用于从准确率分母中排除补齐位置.
        self.pad_id = pad_id
        # 外部可注入自定义损失;默认使用忽略 <pad> 的交叉熵.
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=pad_id)

    def evaluate(
        self,
        dataloader: Iterable[dict[str, Tensor]],
        max_batches: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, float]:
        """
        遍历指定数据源并返回结构化评估指标

        在 no_grad 模式下对 dataloader 逐批前向计算,收集 loss、
        困惑度(perplexity)和 token 级准确率.评估结束后自动恢复模型原有的
        train/eval 模式,不会干扰外部训练流程.

        Args:
            dataloader: 提供批量数据的可迭代对象,每个 batch 为 dict,
                        期望包含 source_ids、target_input_ids、target_output_ids
            max_batches: 最多评估的 batch 数量,None 表示遍历全部数据
            show_progress: 是否显示 tqdm 进度条

        Returns:
            dict,包含三个固定键:loss(平均交叉熵)、perplexity(困惑度)、
            token_accuracy(排除 padding 后的 token 级准确率)
        """
        # 保存调用前模式,评估完成后恢复,避免影响外部训练流程.
        was_training = self.model.training
        # eval 关闭 Dropout,使同一 checkpoint 的评估结果稳定.
        self.model.eval()
        # total_loss 累加 batch 平均 loss * batch_size.
        total_loss = 0.0
        # total_samples 是平均 loss 的分母.
        total_samples = 0
        # total_correct 与 total_tokens 共同计算有效 token 准确率.
        total_correct = 0
        total_tokens = 0

        try:
            # 评估不反向传播;no_grad 可减少显存占用和计算开销.
            with torch.no_grad():
                # DataLoader 有 __len__,普通 Iterable 可能没有,因此允许 total=None.
                available_batches = (
                    len(dataloader) if hasattr(dataloader, "__len__") else None
                )
                # max_batches 同时限制进度条总数和真实循环次数.
                progress_total = available_batches
                if max_batches is not None and progress_total is not None:
                    progress_total = min(progress_total, max_batches)
                # 关闭 tqdm 时该对象仍像原 iterable 一样逐批产出数据.
                progress_bar = tqdm(
                    dataloader,
                    total=progress_total,
                    desc="eval",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                    disable=not show_progress,
                )

                for batch_index, batch in enumerate(progress_bar):
                    # 达到用户指定上限后停止,避免额外执行一次模型前向.
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    # batch 的三个值都是二维 long Tensor,统一迁移到评估设备.
                    batch = {
                        name: value.to(self.device) for name, value in batch.items()
                    }
                    # source_ids shape=(batch_size, source_length),送入 Encoder.
                    source_ids = batch["source_ids"]
                    # target_input_ids shape=(batch_size, target_length),以 <bos> 开头.
                    target_input_ids = batch["target_input_ids"]
                    # target_output_ids shape=(batch_size, target_length),作为监督标签.
                    target_output_ids = batch["target_output_ids"]
                    # logits shape=(batch_size, target_length, vocab_size).
                    logits = self.model(source_ids, target_input_ids)
                    # CrossEntropyLoss 需要类别维位于 dim=1,所以交换后两维.
                    loss = self.criterion(
                        logits.transpose(1, 2),
                        target_output_ids,
                    )

                    # 最后一个 batch 可能不完整,因此从真实输入读取 batch_size.
                    batch_size = source_ids.size(0)
                    # batch loss 默认已经平均,乘样本数后用于全局加权平均.
                    total_loss += float(loss.item()) * batch_size
                    total_samples += batch_size
                    # valid_mask shape=(batch_size, target_length),False 代表 <pad>.
                    valid_mask = target_output_ids != self.pad_id
                    # 在 vocab_size 维取 argmax 后得到二维 token id 预测.
                    predictions = logits.argmax(dim=-1)
                    # 同时满足预测正确和非 padding 才计入正确 token.
                    total_correct += int(
                        ((predictions == target_output_ids) & valid_mask).sum().item()
                    )
                    # 仅非 padding token 进入准确率分母.
                    total_tokens += int(valid_mask.sum().item())
                    # tqdm 显示累计指标,比展示单 batch 指标更平滑.
                    progress_bar.set_postfix(
                        loss=f"{total_loss / max(total_samples, 1):.4f}",
                        accuracy=f"{total_correct / max(total_tokens, 1):.4f}",
                    )
        finally:
            # 只有调用前处于 train 模式时才恢复,尊重外部调用者原状态.
            if was_training:
                self.model.train()

        # 空数据通常意味着路径或预处理有问题,应显式报错而不是返回 0.
        if total_samples == 0:
            raise ValueError("评估 DataLoader 没有可用样本")

        # 按实际样本数计算平均验证交叉熵.
        average_loss = total_loss / total_samples
        # exp(loss) 可能溢出;截断到 20 仍能表达模型表现很差.
        perplexity = math.exp(min(average_loss, 20.0))
        # 如果所有目标都是 padding,则使用分母 1 并返回 0 accuracy.
        token_accuracy = total_correct / max(total_tokens, 1)
        # 使用固定键名返回,便于 CLI、测试和可视化模块消费.
        return {
            "loss": average_loss,
            "perplexity": perplexity,
            "token_accuracy": token_accuracy,
        }
