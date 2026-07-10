"""
项目配置模块

集中管理项目的所有配置项,包括:
- defaults:默认训练参数与模型超参数配置
- paths:项目路径常量定义
"""

# 导入配置子模块
from configs import defaults, paths

# 对外暴露的子模块列表
__all__ = ["defaults", "paths"]
