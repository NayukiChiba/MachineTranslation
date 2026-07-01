# MachineTranslation

一个用于学习和实现神经机器翻译的 Python 项目。项目重点不是直接调用
成熟框架中的完整 Transformer，而是从基础组件开始手写 Attention，
逐步搭建可训练、可评估、可解释的机器翻译模型。

## 项目目标

- 手写并理解 Seq2Seq、Attention、Transformer 等核心结构
- 构建清晰可维护的机器翻译实验代码
- 支持数据预处理、模型训练、推理翻译、指标评估和注意力可视化
- 保留实验灵活性，便于替换模型结构和训练配置

## 技术栈

- Python 3.11+
- PyTorch
- NumPy / Pandas
- Matplotlib
- Ruff
- uv

## 推荐项目架构

```text
MachineTranslation/
├── configs/                     # 训练、数据、模型配置
│   ├── base_config.py
│   ├── data_config.py
│   └── train_config.py
├── data/                        # 小样例数据，可提交到仓库
│   └── samples/
├── datasets/                    # 原始和处理后的完整数据集，不提交
│   ├── raw/
│   └── processed/
├── docs/                        # 设计说明和实验记录
│   ├── attention.md
│   └── transformer.md
├── notebooks/                   # 探索性实验和可视化分析
├── outputs/                     # 模型权重、预测结果和图表，不提交
│   ├── checkpoints/
│   ├── logs/
│   ├── tensorboard/
│   ├── figures/
│   └── predictions/
├── src/
│   ├── cli/                     # 命令行入口
│   │   ├── __init__.py
│   │   └── main.py
│   ├── data/                    # 数据下载、清洗、分词、词表构建、DataLoader
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── download.py
│   │   ├── interim.py
│   │   ├── process.py
│   │   ├── tokenizer.py
│   │   ├── utils.py
│   │   └── vocabulary.py
│   ├── evaluate/                # 评估流程和指标统计
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── inference/               # 模型加载和翻译接口
│   │   ├── __init__.py
│   │   └── translator.py
│   ├── model/                   # 模型结构
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── seq2seq.py
│   │   └── transformer.py
│   ├── train/                   # 训练循环、损失和优化器
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/                   # 通用工具，保持克制使用
│   │   ├── __init__.py
│   │   ├── seed_manager.py
│   │   └── path_manager.py
├── tests/                       # 单元测试和轻量集成测试
│   ├── test_attention.py
│   ├── test_vocabulary.py
│   └── test_translator.py
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## 模块职责

### `cli`

负责命令行入口和任务分发。

- 启动训练任务
- 启动评估任务
- 启动单句或批量翻译
- 读取配置路径和运行参数

### `data`

负责完整的机器翻译数据管线：从原始语料下载到模型可用的张量输入。

- `download.py` — 从 Hugging Face 缓存导出 OPUS-100 中英数据集到 raw JSONL
- `interim.py` — 清洗中英文文本、过滤无效样本、去重，生成 interim JSONL
- `tokenizer.py` — 训练 SentencePiece BPE 分词器，提供 encode / decode 接口
- `vocabulary.py` — 词表查询与检查（token2id / id2token / 特殊 token）
- `process.py` — 使用分词器将 interim 文本转为 token id，生成 processed JSONL
- `dataloader.py` — 读取 processed JSONL，batch 级 padding，导出 DataLoader
- `utils.py` — data 管线通用工具（JSONL 解析等）

### `evaluate`

负责独立评估流程和指标统计。

- 加载 checkpoint 和测试集
- 计算 loss、accuracy、BLEU 等指标
- 保存评估报告
- 输出错误样例，辅助分析模型问题

### `inference`

负责把训练好的模型用于真实翻译。

- 加载模型权重和词表
- 支持贪心解码
- 后续可扩展 beam search
- 输出翻译文本和可选 Attention 权重

### `model`

负责模型结构实现，是本项目的核心模块。

- `attention.py`：手写点积注意力、缩放点积注意力、多头注意力
- `encoder.py`：编码器结构
- `decoder.py`：解码器结构
- `seq2seq.py`：带 Attention 的 Seq2Seq 模型
- `transformer.py`：完整 Transformer 翻译模型

### `train`

负责训练流程。

- 训练循环
- 验证集 loss 统计
- checkpoint 保存和加载
- 损失函数和优化器构建
- 随机种子和设备选择

### `utils`

负责少量跨模块复用的基础工具。

- 路径管理
- 随机种子设置
- 配置读取
- 日志初始化

## 建议实现顺序

1. 完成词表、分词和 Dataset 构建
2. 实现基础 Encoder-Decoder 模型
3. 实现手写 Attention，并接入 Decoder
4. 增加训练循环和 checkpoint
5. 增加推理翻译接口
6. 增加 Attention 可视化
7. 扩展到 Transformer
8. 补充测试和实验文档

## 开发环境

安装依赖：

```bash
uv sync
```

安装 pre-commit：

```bash
uv run pre-commit install
```

格式化代码：

```bash
uv run ruff format .
```

检查代码：

```bash
uv run ruff check .
```

运行测试：

```bash
uv run pytest
```

## 数据约定

建议平行语料使用 TSV 格式保存：

```text
source<TAB>target
hello<TAB>你好
how are you<TAB>你好吗
```

完整数据集放在 `datasets/` 下，不提交到 Git。可以把极小样例数据放入
`data/samples/`，用于测试和演示。

## 输出约定

- 模型权重保存到 `outputs/checkpoints/`
- 文本日志保存到 `outputs/logs/`
- TensorBoard 日志保存到 `outputs/tensorboard/`
- 图表保存到 `outputs/figures/`
- 翻译结果保存到 `outputs/predictions/`

这些目录默认不提交到 Git。

## 代码风格

本项目强调清晰、显式、易读：

- 路径处理统一使用 `os.path` 或 `pathlib`
- Python 文件、函数和变量使用 snake_case
- 类名使用 PascalCase
- 常量使用全大写加下划线
- 注释和文档字符串使用中文
- 不静默吞掉异常，错误信息应明确说明上下文

## 当前状态

项目处于初始化阶段。当前 README 提供推荐架构和实现路线，后续代码应围绕
“手写 Attention 的机器翻译实验项目”逐步落地。
