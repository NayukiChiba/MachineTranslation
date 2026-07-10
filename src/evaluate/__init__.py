"""
模型评估和指标统计模块。

功能：
1. 提供统一模型评估器（Evaluator），管理评估流程、数据加载、指标计算与结果输出
2. 提供评估指标函数：BLEU 分数计算、逐 Token 准确率计算
3. 通过 evaluate 子模块管理评估入口，支持批量评估与单独指标调用

使用方法：
    from src.evaluate import Evaluator, calculate_bleu, calculate_token_accuracy
"""

# 导入评估器主类，对外提供统一的评估入口
from src.evaluate.evaluator import Evaluator

# 导入独立指标函数，支持在评估器外部直接调用
from src.evaluate.metrics import calculate_bleu, calculate_token_accuracy

# 声明模块对外暴露的公共接口，避免内部实现细节泄露
__all__ = ["Evaluator", "calculate_bleu", "calculate_token_accuracy"]
