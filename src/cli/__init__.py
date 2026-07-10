"""
CLI 包入口模块

提供命令行接口的统一入口，包含两个核心功能：
1. 命令行参数解析器（create_parser），用于解析用户输入的参数
2. 交互式菜单（show_menu），用于引导用户完成翻译及训练等操作

使用方法：
    from src.cli import create_parser, show_menu
"""

# 导入交互式菜单入口函数
from src.cli.menu import show_menu

# 导入命令行参数解析器构造函数
from src.cli.parser import create_parser

# 声明本包对外暴露的公共接口
__all__ = ["create_parser", "show_menu"]
