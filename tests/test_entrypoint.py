"""
唯一入口和任务主函数分流测试.

测试 main.py 的命令路由逻辑:
1. 无参数时进入交互式菜单
2. train 子命令分发给 train_main
3. eval 子命令分发给 eval_main
4. translate 子命令分发给 translate_main
"""

import main


def test_main_without_arguments_routes_to_menu(monkeypatch) -> None:
    """无参数时必须进入 show_menu,而不是创建 CLI parser."""
    # 替换 show_menu 为返回哨兵值的 lambda,验证路由逻辑
    monkeypatch.setattr(main, "show_menu", lambda: 9)

    # 无参数调用 main,应返回 0(正常退出)
    assert main.main([]) == 0


def test_main_routes_train_command_to_train_main(monkeypatch) -> None:
    """train 子命令必须分发给独立 train_main."""
    # 用于捕获被调用的子命令名
    received_commands: list[str] = []

    def fake_train_main(args) -> None:
        received_commands.append(args.command)

    # 用假函数替换真实 train_main,验证路由分发
    monkeypatch.setattr(main, "train_main", fake_train_main)

    # 验证 main 正确解析 train 子命令并调用 train_main
    assert main.main(["train", "--epochs", "1", "--warmup-steps", "0"]) == 0
    assert received_commands == ["train"]


def test_main_routes_eval_alias_to_eval_main(monkeypatch) -> None:
    """eval 主命令必须分发给独立 eval_main."""
    # 用于捕获被调用的子命令名
    received_commands: list[str] = []

    def fake_eval_main(args) -> None:
        received_commands.append(args.command)

    # 用假函数替换真实 eval_main,验证路由分发
    monkeypatch.setattr(main, "eval_main", fake_eval_main)

    # 验证 main 正确解析 eval 子命令并调用 eval_main
    assert main.main(["eval", "--checkpoint", "model.pth"]) == 0
    assert received_commands == ["eval"]


def test_main_routes_translate_command_to_translate_main(monkeypatch) -> None:
    """translate 子命令必须保留输入文本并进入 translate_main."""
    # 用于捕获传入待翻译文本
    received_texts: list[str] = []

    def fake_translate_main(args) -> None:
        received_texts.append(args.text)

    # 用假函数替换真实 translate_main,验证路由分发
    monkeypatch.setattr(main, "translate_main", fake_translate_main)

    # 验证 main 正确解析 translate 子命令并将文本传入 translate_main
    assert main.main(["translate", "hello"]) == 0
    assert received_texts == ["hello"]
