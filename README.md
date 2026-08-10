# English | [中文](README_cn.md)

# tron1-rl-isaaclab

Reinforcement learning training stack for the LimX **TRON1** bipedal robot, built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and using PPO to train locomotion and manipulation policies. This repository extends the Isaac Lab template to support sim-to-real training for TRON1 robot variants.

## Repository Structure

```
.
|-- exts/bipedal_locomotion/   # Isaac Lab extension: env/asset/MDP/robot cfg
|-- rsl_rl/                    # Vendored rsl_rl fork (PPO + on-policy runner)
|-- scripts/rsl_rl/            # Training/play entry points (train.py / play.py)
|-- pyproject.toml             # Python project configuration
|-- LICENSE                    # Apache 2.0
```

## Requirements

- Isaac Sim + Isaac Lab, with `isaaclab` / `isaaclab_tasks` / `isaaclab_rl` importable
- Python 3.10
- GPU (>= 12 GB VRAM recommended for multi-env training)

## Installation

```bash
# 1. Clone the repository with submodules
git clone --recurse-submodules https://github.com/limxdynamics/tron1-rl-isaaclab.git
cd tron1-rl-isaaclab
# If already cloned without submodules:
git submodule update --init --recursive

# 2. Editable install of the extension and vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

## Training

Task IDs are registered in `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py`.

```bash
# Solefoot (SF)
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON1A-Blind-Flat-v0 --num_envs 4096 --headless

# Wheelfoot (WF)
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON1A-Blind-Flat-v0 --num_envs 4096 --headless
```

Common options:
- `--checkpoint_path <path>` -- resume from a specific `.pt` checkpoint
- `--video` -- enable video recording
- `--max_iterations N` -- override the maximum iteration count

Log path: `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`

## Robot Morphologies

| Morphology | End-effector | Task ID Prefix |
|---|---|---|
| SF_TRON1A | sole foot (ankle pitch) | `Isaac-Limx-SF-TRON1A-...` |
| WF_TRON1A | wheel | `Isaac-Limx-WF-TRON1A-...` |

## Architecture Overview

Three top-level packages:

1. **`exts/bipedal_locomotion/`** -- Isaac Lab extension. All env/asset/MDP/robot configs live here.
2. **`rsl_rl/`** -- Vendored fork of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) providing the PPO trainer and on-policy runner.
3. **`scripts/rsl_rl/`** -- Entry-point scripts for training and inference.

## Related Repositories

| Repository | Description |
|---|---|
| [tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | TRON1 robot model files |
| [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym) | TRON1 RL training with Isaac Gym |
| [tron1-rl-deploy-ros](https://github.com/limxdynamics/tron1-rl-deploy-ros) | TRON1 RL deployment (ROS) |
| [tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python) | TRON1 RL deployment (Python) |

## License

[Apache 2.0](LICENSE).