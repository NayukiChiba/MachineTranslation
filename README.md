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
├── main.py
├── configs/
│   ├── __init__.py
│   ├── paths.py                 # 路径常量
│   └── defaults.py              # 默认超参数
├── notebooks/                   # 探索性实验和可视化分析
├── outputs/                     # 模型权重、预测结果和图表，不提交
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── predictions/
├── src/
│   ├── __init__.py
│   ├── cli/                     # 命令行入口
│   │   ├── __init__.py
│   │   ├── actions.py           # CLI 和菜单共用任务管线
│   │   ├── parser.py
│   │   └── menu.py
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
│   │   ├── metrics.py
│   │   ├── visualize.py
│   │   └── evaluator.py
│   ├── inference/               # 模型加载和翻译接口
│   │   ├── __init__.py
│   │   └── translator.py
│   ├── model/                   # Transformer 模型
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── mask.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── transformer.py
│   ├── train/                   # 训练循环、损失和优化器
│   │   ├── __init__.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── early_stopping.py
│   │   ├── checkpoint.py
│   │   ├── logger.py
│   │   ├── utils.py
│   │   └── trainer.py
├── tests/                       # 单元测试和轻量集成测试
│   ├── test_tokenizer.py
│   ├── test_vocabulary.py
│   ├── test_dataset.py
│   ├── test_mask.py
│   ├── test_transformer.py
│   ├── test_train_components.py
│   ├── test_metrics.py
│   ├── test_translator.py
│   └── test_entrypoint.py
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
- 无参数启动时提供交互菜单
- CLI 和菜单复用 `actions.py`，避免业务流程重复

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

- `metrics.py` — 计算 loss、token accuracy、轻量 BLEU
- `evaluator.py` — 统一评估入口，返回结构化指标
- `visualize.py` — 保存 loss 曲线和预测样例

### `inference`

负责把训练好的模型用于真实翻译。

- 加载模型权重和词表
- 支持贪心解码
- 后续可扩展 beam search
- 输出翻译文本和可选 Attention 权重

### `model`

负责模型结构实现，是本项目的核心模块。

- `embedding.py`：TokenEmbedding 与 PositionalEncoding
- `mask.py`：padding mask 与 causal mask
- `encoder.py`：编码器结构
- `decoder.py`：解码器结构
- `transformer.py`：完整 Transformer 翻译模型

### `train`

负责训练流程。

- `optimizer.py` — 创建优化器
- `scheduler.py` — 创建学习率调度器
- `early_stopping.py` — 验证集无提升时提前停止
- `checkpoint.py` — 保存和加载 checkpoint
- `logger.py` — 控制台和文件日志
- `utils.py` — 随机种子、设备、参数统计、梯度裁剪
- `trainer.py` — 训练主循环，作为唯一调度者

训练模块遵循单向依赖：`optimizer`、`scheduler`、`early_stopping`、
`checkpoint`、`logger` 和 `utils` 彼此不导入，只有 `trainer.py` 负责组合。
`Trainer` 支持注入这些组件，便于独立测试和替换实现。

训练、验证、独立评估、逐 token 翻译和各数据处理阶段均使用 tqdm 展示进度。
CLI 可通过 `--no-progress` 关闭动态进度条。

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

## 运行方式

`main.py` 是项目唯一入口。入口结构参考同级 `Attention` 项目：
`src/cli/parser.py` 只声明参数，`main.py` 提供 `train_main`、`eval_main`、
`translate_main`，不携带参数时进入 `src/cli/menu.py` 的交互菜单。

启动交互菜单：

```bash
uv run python main.py
```

准备数据：

```bash
uv run python main.py prepare
```

训练模型：

```bash
uv run python main.py train --epochs 10 --batch-size 32
```

覆盖模型、优化器、调度器和运行配置：

```bash
uv run python main.py train --d-model 256 --num-heads 8 --encoder-layers 4 --decoder-layers 4 --optimizer adamw --lr 3e-4 --scheduler cosine_warmup --warmup-steps 500 --device cuda
```

从检查点恢复训练：

```bash
uv run python main.py train --resume outputs/checkpoints/last_model.pth
```

评估与翻译：

```bash
uv run python main.py eval --checkpoint outputs/checkpoints/best_model.pth --split test
uv run python main.py translate "hello world" --checkpoint outputs/checkpoints/best_model.pth --max-length 128
```

`translate` 不提供文本时会进入持续交互翻译模式。`eval` 也接受兼容别名
`evaluate`，但文档和测试统一使用 `eval`。

查看完整参数：

```bash
uv run python main.py --help
```

## 数据约定

数据管线使用 JSONL 保存平行语料。raw 和 interim 阶段的单条数据格式为：

```json
{"en": "hello", "zh": "你好"}
```

完整数据集默认放在项目的 `datasets/` 下，不提交到 Git。可通过环境变量
`MACHINE_TRANSLATION_DATASETS_DIR` 覆盖数据根目录。所有具体文件路径统一由
`configs/paths.py` 管理。

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

项目已具备从数据准备、Transformer 训练、早停和检查点恢复，到独立评估、
贪心翻译、日志记录和训练曲线保存的完整基础闭环。CLI、交互菜单和测试均已
接入，后续可以继续扩展 beam search、标准 BLEU 和 Attention 可视化。
