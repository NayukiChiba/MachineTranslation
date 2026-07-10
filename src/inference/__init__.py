"""
模型推理和翻译接口模块。

功能：
1. 提供 Translator 类，封装训练好的翻译模型，支持批量翻译
2. 提供 load_translator 工厂函数，从检查点快速加载翻译器
3. 通过 __init__.py 统一对外导出，简化调用方导入路径

使用方法：
    from src.inference import Translator, load_translator

    # 方式一：从检查点加载
    translator = load_translator("checkpoints/best.pt")

    # 方式二：直接实例化
    translator = Translator(model, tokenizer, config)
    result = translator.translate("Hello, world!")
"""

# 从 translator 模块导入核心接口
from src.inference.translator import Translator, load_translator

# 控制 from src.inference import * 的导出内容
__all__ = ["Translator", "load_translator"]
