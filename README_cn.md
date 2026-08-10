# 中文 | [English](README.md)

# tron1-rl-isaaclab

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 LimX **TRON1** 双足机器人强化学习训练栈，使用 PPO 训练 locomotion 与 manipulation 策略。本仓库扩展了 Isaac Lab 模板，支持 TRON1 机器人变体的 Sim-to-Real 训练。

## 仓库结构

```
.
|-- exts/bipedal_locomotion/   # Isaac Lab extension：env/asset/MDP/robot cfg
|-- rsl_rl/                    # 项目内 vendored 的 rsl_rl fork（PPO + on-policy runner）
|-- scripts/rsl_rl/            # 训练 / play 入口（train.py / play.py）
|-- pyproject.toml             # Python 项目配置
|-- LICENSE                    # Apache 2.0
```

## 环境要求

- Isaac Sim + Isaac Lab，且 `isaaclab` / `isaaclab_tasks` / `isaaclab_rl` 可被 import
- Python 3.10
- GPU（推荐 >= 12 GB 显存，用于多环境训练）

## 安装

```bash
# 1. clone 仓库连同子模块
git clone --recurse-submodules https://github.com/limxdynamics/tron1-rl-isaaclab.git
cd tron1-rl-isaaclab
# 如果已 clone 但未拉子模块：
git submodule update --init --recursive

# 2. editable install extension 与 vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

## 训练

任务 ID 在 `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py` 中注册。

```bash
# Solefoot (SF)
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON1A-Blind-Flat-v0 --num_envs 4096 --headless

# Wheelfoot (WF)
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON1A-Blind-Flat-v0 --num_envs 4096 --headless
```

常用选项：
- `--checkpoint_path <path>` -- 从指定的 .pt checkpoint 恢复训练
- `--video` -- 启用视频录制
- `--max_iterations N` -- 覆盖最大迭代次数

日志路径：`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`

## 机器人形态

| 形态 | 末端 | Task ID 前缀 |
|---|---|---|
| SF_TRON1A | sole foot (ankle pitch) | `Isaac-Limx-SF-TRON1A-...` |
| WF_TRON1A | wheel | `Isaac-Limx-WF-TRON1A-...` |

## 架构概览

三个顶层包：

1. **`exts/bipedal_locomotion/`** -- Isaac Lab extension。env/asset/MDP/robot cfg 全部在这里
2. **`rsl_rl/`** -- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 的 vendored fork，提供 PPO 训练器与 on-policy runner
3. **`scripts/rsl_rl/`** -- 训练与推理入口脚本

## 相关仓库

| 仓库 | 描述 |
|---|---|
| [tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | TRON1 机器人模型文件 |
| [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym) | TRON1 Isaac Gym RL 训练 |
| [tron1-rl-deploy-ros](https://github.com/limxdynamics/tron1-rl-deploy-ros) | TRON1 RL 部署（ROS） |
| [tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python) | TRON1 RL 部署（Python） |

## 许可证

[Apache 2.0](LICENSE)。