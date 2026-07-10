"""
分词器模块的单元测试.

测试内容:
1. iter_parallel_texts 函数对空行、无效 JSON 行的跳过行为
2. 验证仅保留 src 和 tgt 均非空的合法行
"""

import json

from src.data.tokenizer import iter_parallel_texts


def test_iter_parallel_texts_skips_empty_and_invalid_lines(tmp_path) -> None:
    """
    验证 iter_parallel_texts 跳过无效行并正确提取有效行.

    场景:
    - 正常行(src 和 tgt 均非空):应被提取
    - tgt 为空的行:应被跳过
    - 非 JSON 行:应被跳过
    """
    # 构造测试用 JSONL 文件,包含正常行、译文字段为空的行、非 JSON 行
    input_path = tmp_path / "parallel.jsonl"
    # 写入测试 JSONL:第 1 行有效,第 2 行 tgt 为空,第 3 行非 JSON
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"en": "hello", "zh": "你好"}, ensure_ascii=False),
                json.dumps({"en": "world", "zh": ""}, ensure_ascii=False),
                "not-json",
            ]
        ),
        encoding="utf-8",
    )

    # 期望结果:仅第 1 行的两个字段和第 2 行的 src 字段被提取(tgt 为空跳过,非 JSON 跳过)
    assert list(iter_parallel_texts(input_path)) == ["hello", "你好", "world"]
