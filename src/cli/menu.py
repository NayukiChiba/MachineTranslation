"""
交互式菜单入口

无命令行参数启动 main.py 时进入本模块.整体写法参考 Attention 项目:每个
菜单任务拥有独立的 `_menu_*` 函数,训练前展示和修改配置,评估与翻译前选择
checkpoint,主菜单只负责循环和分发.
"""

from pathlib import Path
from typing import Any

from configs import paths
from configs.defaults import (
    DataLoaderConfig,
    InferenceConfig,
    ModelConfig,
    StaticConfig,
    TrainConfig,
)
from src.cli.actions import evaluate_model, prepare_data, train_model
from src.inference.translator import load_translator


def _parse_config_value(raw_value: str, current_value: Any) -> Any:
    """根据当前字段类型把用户输入转换为配置值."""
    # bool 是 int 的子类,因此必须在 int 判断之前单独处理.
    if isinstance(current_value, bool):
        normalized_value = raw_value.strip().lower()
        if normalized_value in {"true", "1", "yes", "y"}:
            return True
        if normalized_value in {"false", "0", "no", "n"}:
            return False
        raise ValueError("布尔值请输入 true/false、yes/no 或 1/0")
    # int 字段用于 batch、层数、步数和随机种子.
    if isinstance(current_value, int):
        return int(raw_value)
    # float 字段用于学习率、dropout、阈值和衰减系数.
    if isinstance(current_value, float):
        return float(raw_value)
    # 当前值为 None 的可选字段在本项目中只有浮点阈值.
    if current_value is None:
        if raw_value.strip().lower() in {"none", "null"}:
            return None
        return float(raw_value)
    # 设备、优化器类型、调度器类型等字段保持字符串.
    return raw_value


def _edit_config(config: type[StaticConfig]) -> type[StaticConfig]:
    """逐项询问静态配置字段,空输入表示保留当前值."""
    # updates 只记录用户实际输入的字段,未输入字段继续使用原对象值.
    updates: dict[str, Any] = {}
    print(f"\n修改 {config.__name__},直接回车保留括号内默认值")
    for fieldName in config.getFieldNames():
        current_value = getattr(config, fieldName)
        raw_value = input(f"{fieldName} ({current_value!r}): ").strip()
        if not raw_value:
            continue
        updates[fieldName] = _parse_config_value(raw_value, current_value)
    return config.withOverrides(**updates)


def _ask_to_edit(config: type[StaticConfig]) -> type[StaticConfig]:
    """展示配置,并在用户确认后进入逐字段编辑."""
    config.printSummary()
    answer = input("是否修改以上配置? [y/N]: ").strip().lower()
    if answer != "y":
        return config
    return _edit_config(config)


def _select_checkpoint() -> Path | None:
    """按修改时间列出统一 checkpoint 目录中的所有权重文件."""
    # 最近写入的 checkpoint 排在第一位,直接回车即可选择最新文件.
    checkpoints = sorted(
        paths.CHECKPOINTS_DIR.glob("*.pth"),
        key=lambda checkpoint_path: checkpoint_path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        print("没有找到 checkpoint,请先训练模型")
        return None

    print("\n可用 checkpoint:")
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"  {checkpoint_index}. {checkpoint_path.name}")

    raw_choice = input(f"请选择 [1-{len(checkpoints)}],默认 1: ").strip() or "1"
    try:
        selected_index = int(raw_choice) - 1
    except ValueError:
        print("输入不是整数,已选择最新 checkpoint")
        return checkpoints[0]
    if 0 <= selected_index < len(checkpoints):
        return checkpoints[selected_index]
    print("序号超出范围,已选择最新 checkpoint")
    return checkpoints[0]


def _menu_prepare() -> None:
    """交互式数据准备入口."""
    # 默认复用已有产物,只有明确输入 y 时才重新执行全部阶段.
    force_answer = input("是否强制重新生成全部数据? [y/N]: ").strip().lower()
    prepare_data(force=force_answer == "y")


def _menu_train() -> None:
    """交互式训练入口,可分别修改模型、DataLoader 和训练配置."""
    print("\n准备训练配置")
    # 模型配置控制所有 Transformer 张量的隐藏维度和层数.
    model_config = _ask_to_edit(ModelConfig)
    # DataLoader 配置控制 batch 第一维、worker 数和数据准备行为.
    loader_config = _ask_to_edit(DataLoaderConfig)
    # 训练配置控制优化器、调度器、早停、AMP、日志和 tqdm.
    train_config = _ask_to_edit(TrainConfig)

    # 如果存在 last_model.pth,则允许用户从上次 optimizer/scheduler 状态继续.
    resume_from: Path | None = None
    if paths.LAST_MODEL_PATH.is_file():
        resume_answer = input("发现 last_model.pth,是否恢复训练? [y/N]: ").strip()
        if resume_answer.lower() == "y":
            resume_from = paths.LAST_MODEL_PATH

    # 共享 action 负责创建数据、模型与 Trainer,menu 不复制训练实现.
    train_model(
        model_config=model_config,
        train_config=train_config,
        loader_config=loader_config,
        resume_from=resume_from,
    )


def _menu_eval() -> None:
    """交互式评估入口."""
    # 评估必须先选择一个包含模型权重的 checkpoint.
    checkpoint_path = _select_checkpoint()
    if checkpoint_path is None:
        return

    print("\n评估数据集:")
    print("  1. test")
    print("  2. val")
    print("  3. train")
    split_choice = input("请选择 [1-3],默认 1: ").strip()
    # 未知输入安全回退到 test,避免意外在完整训练集上耗时评估.
    split = {"1": "test", "2": "val", "3": "train", "": "test"}.get(
        split_choice,
        "test",
    )
    # 用户可以修改 batch_size、worker_count 和数据准备策略.
    loader_config = _ask_to_edit(DataLoaderConfig)
    # 设备默认与 TrainConfig 保持一致,CUDA 可用时自动选择 CUDA.
    device = input(f"device ({TrainConfig.device}): ").strip() or TrainConfig.device
    # 空输入表示遍历完整切分,否则只评估指定数量的 batch.
    raw_max_batches = input("max_batches (None): ").strip()
    max_batches = int(raw_max_batches) if raw_max_batches else None
    evaluate_model(
        checkpoint_path=checkpoint_path,
        split=split,
        loader_config=loader_config,
        device=device,
        max_batches=max_batches,
        show_progress=True,
    )


def _menu_translate() -> None:
    """交互式翻译入口,模型只加载一次并持续处理多条文本."""
    # 与评估相同,翻译模型结构必须与 checkpoint 中的参数完全一致.
    checkpoint_path = _select_checkpoint()
    if checkpoint_path is None:
        return
    # 允许修改最大生成长度和是否显示 token 级进度条.
    inference_config = _ask_to_edit(InferenceConfig)
    # 默认设备从训练配置读取,也允许临时改为 cpu 或指定 cuda 编号.
    device = input(f"device ({TrainConfig.device}): ").strip() or TrainConfig.device
    # 在循环外加载 tokenizer 和模型,避免每翻译一句都重复读取 checkpoint.
    translator = load_translator(
        checkpoint_path=checkpoint_path,
        tokenizer_path=paths.TOKENIZER_MODEL_PATH,
        device=device,
        max_generation_length=inference_config.max_generation_length,
        show_progress=inference_config.show_progress,
    )

    print("进入交互翻译模式,输入 quit、exit 或 q 返回主菜单")
    while True:
        source_text = input("\n请输入英文文本: ").strip()
        if source_text.lower() in {"quit", "exit", "q"}:
            return
        if not source_text:
            print("输入文本不能为空")
            continue
        # translate 返回目标语言字符串;模型输出 shape 在内部逐步增长.
        translation = translator.translate(source_text)
        print(f"翻译结果: {translation}")


def print_banner() -> None:
    """打印简洁的项目标题."""
    print("=" * 60)
    print("MachineTranslation - Encoder-Decoder Transformer")
    print("=" * 60)


def show_menu() -> int:
    """显示主菜单并循环分发任务,直到用户选择退出."""
    print_banner()
    while True:
        print("\n主菜单:")
        print("  1. 准备数据")
        print("  2. 训练模型")
        print("  3. 评估模型")
        print("  4. 翻译文本")
        print("  0. 退出")
        choice = input("\n请选择操作 [0-4]: ").strip()

        try:
            if choice == "0":
                print("退出程序")
                return 0
            if choice == "1":
                _menu_prepare()
            elif choice == "2":
                _menu_train()
            elif choice == "3":
                _menu_eval()
            elif choice == "4":
                _menu_translate()
            else:
                print("无效选择,请重新输入")
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            # 交互菜单捕获可恢复错误,打印上下文后允许用户继续选择任务.
            print(f"执行失败: {error}")


# 兼容上一版入口名称;新代码和 README 统一使用 show_menu.
run_menu = show_menu


if __name__ == "__main__":
    raise SystemExit(show_menu())
