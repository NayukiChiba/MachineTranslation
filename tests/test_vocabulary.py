"""
词表模块测试

功能:
1. 测试 load_sentencepiece_vocab 加载 SentencePiece 词表文件
2. 测试 build_vocab_json_data 构建词表映射字典
3. 验证词表数据结构的正确性(id、token、score 字段)
4. 验证 token_to_id 和 id_to_token 映射键名使用蛇形命名
"""

from src.data.vocabulary import build_vocab_json_data, load_sentencepiece_vocab


def test_build_vocab_json_uses_snake_case_keys(tmp_path) -> None:
    """
    验证 build_vocab_json_data 和 load_sentencepiece_vocab 的正确性

    测试要点:
    - load_sentencepiece_vocab 能正确解析词表文件中的 token 和 score
    - build_vocab_json_data 返回的映射字典键名使用蛇形命名
    - token_to_id 和 id_to_token 两个映射方向均正确

    Args:
        tmp_path: pytest 提供的临时目录 fixture

    验证项:
    - 第二条词表项包含正确的 id、token 和 score 字段
    - token_to_id 映射正确(token 到 id 的字符串映射)
    - id_to_token 映射正确(id 字符串到 token 的映射)
    """
    # 创建临时词表文件,包含 pad 符号和普通 token
    vocab_path = tmp_path / "tokenizer.vocab"
    vocab_path.write_text("<pad>\t0\nhello\t-1.5\n", encoding="utf-8")

    # 加载词表为结构化条目列表
    items = load_sentencepiece_vocab(vocab_path)
    # 构建 JSON 格式的映射字典(token_to_id / id_to_token)
    vocab_data = build_vocab_json_data(vocab_path)

    # 验证单条词表项的结构:id、token、score 字段
    assert items[1] == {"id": 1, "token": "hello", "score": -1.5}
    # 验证 token 到 id 的映射,键名使用蛇形命名
    assert vocab_data["token_to_id"] == {"<pad>": 0, "hello": 1}
    # 验证 id 到 token 的映射,键名使用蛇形命名
    assert vocab_data["id_to_token"] == {"0": "<pad>", "1": "hello"}
