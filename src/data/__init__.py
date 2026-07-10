"""
机器翻译数据管线模块.

功能:
1. 对外暴露数据加载器(create_dataloaders)创建接口
2. 对外暴露完整数据管线(prepare_data_pipeline)准备接口

使用方法:
    from src.data import create_dataloaders, prepare_data_pipeline
"""

# 从 dataloader 子模块导入核心接口
from src.data.dataloader import create_dataloaders, prepare_data_pipeline

# 明确模块的公开 API,控制 import * 的行为
__all__ = [
    "create_dataloaders",
    "prepare_data_pipeline",
]
