"""
评估结果可视化与导出模块

功能:
1. 将训练过程中的 loss 曲线保存为 PNG 图片
2. 将模型预测结果导出为 JSONL 格式文件

使用方法:
    from src.evaluate.visualize import save_training_curves, save_predictions

    # 保存训练曲线
    history = {"train_loss": [2.3, 1.8, 1.5], "val_loss": [2.5, 2.0, 1.7]}
    imagePath = save_training_curves(history)

    # 保存预测结果
    predictions = [{"source": "hello", "target": "你好", "prediction": "你好"}]
    outputPath = save_predictions(predictions)
"""

import json
from pathlib import Path

from configs import paths


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path = paths.FIGURES_DIR / "training_curves.png",
) -> Path:
    """
    将训练和验证 loss 曲线绘制并保存为 PNG 图片

    Args:
        history (dict[str, list[float]]): 训练历史字典,
            支持 "train_loss" 和 "val_loss" 两个键,
            每个键对应的值为各 epoch 的 loss 列表
        output_path (Path): 输出图片路径,
            默认为 FIGURES_DIR/training_curves.png

    Returns:
        Path: 保存后的图片文件路径
    """
    # 延迟导入,避免在没有绘图需求时强制依赖 matplotlib
    import matplotlib.pyplot as plt

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建图形并设置尺寸
    figure, axis = plt.subplots(figsize=(8, 5))

    # 根据 history 中实际含有的键绘制对应曲线
    if history.get("train_loss"):
        axis.plot(history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        axis.plot(history["val_loss"], label="val_loss")

    # 设置坐标轴标签与网格
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.3)
    axis.legend()

    # 紧凑布局并保存为高分辨率图片
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_predictions(
    predictions: list[dict[str, str]],
    output_path: Path = paths.PREDICTIONS_DIR / "predictions.jsonl",
) -> Path:
    """
    将预测结果列表导出为 JSONL 格式文件,每行一条 JSON 记录

    Args:
        predictions (list[dict[str, str]]): 预测结果列表,
            每个字典通常包含 "source"、"target"、"prediction" 等字段
        output_path (Path): 输出文件路径,
            默认为 PREDICTIONS_DIR/predictions.jsonl

    Returns:
        Path: 保存后的文件路径
    """
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 逐行写入 JSON,确保中文不被转义为 unicode 编码
    with output_path.open("w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return output_path
