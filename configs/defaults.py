"""
configs/defaults.py
项目默认配置模块

"""


class DataConfig:
    """数据集配置"""

    # 限制英文句子最多 128 个单词，中文句子最多 160 个字符
    MAX_ENGLISH_WORDS = 128
    MAX_CHINESE_CHARS = 160
    # 限制英文句子长度与中文句子长度的比例，避免过长的句子对齐
    MAX_LENGTH_RATIO = 4.0
