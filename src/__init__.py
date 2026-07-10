"""
机器翻译项目源码包

功能:
1. 数据处理:数据集下载、预处理、分词与词汇表构建
2. 模型定义:Transformer 机器翻译模型及其嵌入层
3. 训练管理:训练循环、检查点、早停、学习率调度与日志
4. 推理翻译:模型加载与文本翻译
5. 评估指标:BLEU 等翻译质量评估指标
6. CLI 交互:命令行菜单与参数解析

子包结构:
- src.data      数据处理与词汇表
- src.model     模型架构定义
- src.train     训练基础设施
- src.inference 推理与翻译
- src.evaluate  评估与可视化
- src.cli       命令行交互界面
"""

# 模型核心入口
# CLI 核心入口
from src.cli.menu import show_menu
from src.cli.parser import create_parser

# 评估核心入口
from src.evaluate.evaluator import Evaluator

# 推理核心入口
from src.inference.translator import Translator, load_translator
from src.model.transformer import Transformer

# 组件工厂函数
from src.train.optimizer import create_optimizer
from src.train.scheduler import create_scheduler

# 训练核心入口
from src.train.trainer import Trainer

__all__ = [
    "Transformer",
    "Trainer",
    "Translator",
    "load_translator",
    "Evaluator",
    "show_menu",
    "create_parser",
    "create_optimizer",
    "create_scheduler",
]
