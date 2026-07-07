"""
训练模块

子模块:
    - utils.py            : set_seed / get_device / count_parameters /
                            clip_gradients / move_batch_to_device
    - optimizer.py        : create_optimizer
    - scheduler.py        : create_scheduler
    - early_stopping.py   : EarlyStopping
    - checkpoint.py       : save_checkpoint / load_checkpoint
    - logger.py           : Logger
    - trainer.py          : Trainer (唯一调度者)

依赖规则:
    trainer.py 可以 import 所有 train/ 下模块。
    optimizer / scheduler / early_stopping / checkpoint / logger / utils
    彼此之间不互相 import。
"""
