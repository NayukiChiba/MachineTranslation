"""
交互菜单辅助函数测试模块

测试范围:
1. _parseConfigValue:菜单文本输入的配置值解析与类型转换
2. _selectCheckpoint:交互式 checkpoint 文件选择逻辑
3. showMenu:主菜单循环的入口与退出流程
"""

import os

from configs import paths
from src.cli.menu import _parse_config_value, _select_checkpoint, show_menu


def test_parse_config_value_preserves_field_types() -> None:
    """菜单文本输入应转换为配置字段的真实类型."""
    # 整数输入应返回 int
    assert _parse_config_value("64", 32) == 64
    # 浮点数输入应返回 float
    assert _parse_config_value("0.001", 0.1) == 0.001
    # 布尔值输入应返回 bool
    assert _parse_config_value("false", True) is False
    # 普通字符串保持不变
    assert _parse_config_value("cuda:0", "cpu") == "cuda:0"
    # "none" 字符串应解析为 None
    assert _parse_config_value("none", None) is None


def test_select_checkpoint_defaults_to_newest(tmp_path, monkeypatch) -> None:
    """直接回车时应选择按修改时间排序后的最新 checkpoint."""
    # 创建新旧两个 checkpoint 文件
    older_checkpoint = tmp_path / "older.pth"
    newer_checkpoint = tmp_path / "newer.pth"
    older_checkpoint.write_bytes(b"old")
    newer_checkpoint.write_bytes(b"new")
    # 设置不同的修改时间(较新的时间戳更大)
    os.utime(older_checkpoint, (1, 1))
    os.utime(newer_checkpoint, (2, 2))

    # 模拟 checkpoint 目录为用户输入(直接回车时返回空字符串)
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "")

    # 应返回修改时间最新的文件
    assert _select_checkpoint() == newer_checkpoint


def test_show_menu_can_exit_immediately(monkeypatch) -> None:
    """输入 0 应正常结束菜单循环."""
    # 模拟用户输入 "0" 选择退出
    monkeypatch.setattr("builtins.input", lambda _: "0")

    # 菜单应正常退出,返回状态码 0
    assert show_menu() == 0
