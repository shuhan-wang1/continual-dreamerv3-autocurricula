# 明天要做的事 (Action-Mask 项目)

## 已保存内容

- **代码**：`Action-Mask` 分支已本地 commit（commit 78ea0b2）
- **备份**：`/root/autodl-tmp/continual-dreamerv3-action-mask-backup-*.tar.gz`（项目完整打包）
- **AutoDL 数据盘**：实例关机后 `/root/autodl-tmp` 数据会保留

---

## 明天步骤

### 1. Push 到 GitHub（需 Shuhan 先加你为 collaborator）

```bash
cd /root/autodl-tmp/continual-dreamerv3-autocurricula
git push -u origin Action-Mask
```

若 Shuhan 还没加你，可先 fork 仓库，push 到自己的 fork，再提 PR。

### 2. 换到组员的 dreamer 镜像跑训练

- 新建实例时选组员的 **dreamer 镜像**（CUDA 12 环境）
- 克隆并运行：
  ```bash
  git clone https://github.com/shuhan-wang1/continual-dreamerv3-autocurricula.git
  cd continual-dreamerv3-autocurricula
  git checkout Action-Mask
  conda activate dreamer
  STEPS=5000 bash scripts/run_single_experiment.sh soft
  ```

### 3. 若本机继续用（当前 CUDA 11.8）

先尝试修复 JAX：
```bash
pip uninstall jax jaxlib -y
pip install jax[cuda11_local]==0.4.33
```

---

## 项目结构速览

| 路径 | 说明 |
|------|------|
| `craftax_mask/` | Action mask 规则、提取器、mask 计算 |
| `scripts/run_single_experiment.sh` | 单次实验脚本 |
| `scripts/run_action_mask_experiments.sh` | 批量实验 |
| `input_args.py` | `--action_mask_enabled`, `--action_mask_mode` 等参数 |

---

*生成于 2026-03-13*
