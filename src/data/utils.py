"""
data 模块通用工具函数

功能:
1. JSONL 行解析
2. 其他 data 管线中复用的辅助函数

注意:
    所有 data 子模块中重复出现的工具函数统一放到这里,
    避免各文件各自复制一份相同逻辑.
"""

import json
from typing import Any


def load_jsonl_item(line: str) -> dict[str, Any] | None:
    """
    解析一行 JSONL

    Args:
        line: JSONL 单行文本

    Returns:
        dict[str, Any] | None: 解析后的字典,解析失败返回 None
    """
    # 去除首尾空白字符
    line = line.strip()

    # 跳过空行
    if not line:
        return None

    try:
        # 解析 JSON 字符串
        return json.loads(line)
    except json.JSONDecodeError:
        # 解析失败时静默返回 None
        return None
