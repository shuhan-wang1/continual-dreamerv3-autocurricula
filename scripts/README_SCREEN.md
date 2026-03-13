# 使用 screen 跑 Craftax 实验

## 1. 创建 screen 会话

```bash
screen -S craftax
```

（`craftax` 可换成任意名字）

## 2. 进入项目目录并运行

```bash
cd /root/autodl-tmp/continual-dreamerv3-autocurricula

# 单次实验（推荐过夜跑）
STEPS=500000 bash scripts/run_single_experiment.sh soft

# 或跑三个对比实验（baseline / soft / hard）
STEPS=500000 bash scripts/run_action_mask_experiments.sh
```

## 3. 断开会话（实验继续在后台跑）

按：`Ctrl+A`，松开，再按 `d`

## 4. 明天重新连上 screen 看结果

```bash
screen -r craftax
```

## 5. 查看结果摘要

```bash
cd /root/autodl-tmp/continual-dreamerv3-autocurricula
python scripts/summarize_action_mask_results.py logs/action_mask_*
```

## 常用 screen 命令

| 操作       | 命令                    |
|------------|-------------------------|
| 新建会话   | `screen -S 名字`        |
| 列出会话   | `screen -ls`            |
| 恢复会话   | `screen -r 名字`         |
| 断开会话   | `Ctrl+A` 然后 `d`       |
| 结束会话   | 在 screen 里 `exit`    |
