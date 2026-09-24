# DDT_Mjlab — NP3O Locomotion for Wheel-Legged Robots

Locomotion training for the **D1** (quadruped with wheels) robot, using **NP3O** (BarlowTwins-augmented constrained PPO) built on [mjlab](https://github.com/mujocolab/mjlab).

This is an **mjlab-native** migration from the original IsaacLab-based `DDT_Lab-np3o` codebase, ported to the MuJoCo-backed mjlab framework with behavior equivalence.

---

## Prerequisites

| Dependency | Version |
| ---------- | ------- |
| Python     | 3.11    |
| CUDA       | > 12.4  |
| MJLab      | 1.4.0   |

---

## Installation

### 1. Create conda environment

```bash
conda create -n <your_env_name> python=3.11
conda activate your_env_name
```

### 2. Install PyTorch with CUDA support

mjlab 1.4.0 recommends CUDA version > 12.4. Lower versions may work but training will be significantly slower.

```bash
# Example: install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install mjlab

```bash
pip install mjlab==1.4.0
```

### 4. Clone this repo

```bash
git clone <repo-url> ddt_mjlab-np3o
cd ddt_mjlab-np3o
```

### 5. Verify installation

```bash
python scripts/train.py --list-tasks
```

Expected output:

```
Available tasks:
  Mjlab-Cartpole-Balance
  Mjlab-Cartpole-Swingup
  Mjlab-Lift-Cube-Yam
  Mjlab-Lift-Cube-Yam-Depth
  Mjlab-Lift-Cube-Yam-Rgb
  Mjlab-Multi-Cube-Seg-Yam
  Mjlab-Tracking-Flat-Unitree-G1
  Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation
  Mjlab-Velocity-Flat-D1
  Mjlab-Velocity-Flat-D1H
  Mjlab-Velocity-Flat-Dynawheel
  Mjlab-Velocity-Flat-TITA
  Mjlab-Velocity-Flat-TITATIT
  Mjlab-Velocity-Flat-Unitree-G1
  Mjlab-Velocity-Flat-Unitree-Go1
  Mjlab-Velocity-Rough-D1
  Mjlab-Velocity-Rough-D1H
  Mjlab-Velocity-Rough-Dynawheel
  Mjlab-Velocity-Rough-TITA
  Mjlab-Velocity-Rough-TITATIT
  Mjlab-Velocity-Rough-Unitree-G1
  Mjlab-Velocity-Rough-Unitree-Go1
```

---

## Training

```bash
conda activate <your_env_name>

# D1 — flat ground
python scripts/train.py Mjlab-Velocity-Flat-Dynawheel
```

### Common flags

| Flag                 | Default  | Description                             |
| -------------------- | -------- | --------------------------------------- |
| `--num_envs`       | 4096     | Number of parallel environments         |
| `--max_iterations` | 5000     | Override total training iterations      |
| `--device`         | `auto` | Training device (`cuda:0` or `cpu`) |
| `--resume`         | False    | Resume training from a checkpoint       |
| `--load_run`       | None     | Run directory to load checkpoint from   |

### Logs

Checkpoints and TensorBoard events are written to:

```
logs/np3o/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
├── model_<iter>.pt      # policy checkpoint
├── events.out.tfevents…  # TensorBoard
├── config.json          # effective environment and NP3O configuration
└── cmd.txt              # exact training command and Python interpreter
```

Resuming into an existing run preserves the original files and creates
`config_resume_<timestamp>.json` and `cmd_resume_<timestamp>.txt`.

### TITATIT configuration

Edit `np3o/tasks/locomotion/config/titatit/titatit_flat_cfg.py` for environment settings:

- `UniformVelocityCommandCfg.Ranges`: velocity command ranges.
- `max_curriculum`: maximum curriculum forward speed.
- `rewards`: reward weights such as `action_rate_l2`.
- `_TITATIT_DEFAULT_JOINT_POS`: default joint pose.
- Actuator definitions near the top of the file: stiffness, damping and limits.

Edit `np3o/tasks/locomotion/config/titatit/__init__.py` for TITATIT-only training settings:

- `entropy_coef`: exploration pressure.
- `init_noise_std`: initial policy action noise.
- `max_iterations`: total training iterations.

After changing configuration, start a new run so the new values take effect:

```bash
python scripts/train.py Mjlab-Velocity-Flat-TITATIT
```

### TITATIT 复杂地形（Rough）

训练与回放：

```bash
python scripts/train.py Mjlab-Velocity-Rough-TITATIT
python scripts/play.py Mjlab-Velocity-Rough-TITATIT --checkpoint-file logs/np3o/titatit_rough/YYYY-MM-DD_HH-MM-SS/model_15000.pt --viewer=native --num_envs=1
```

独立环境配置：`np3o/tasks/locomotion/config/titatit/titatit_rough_cfg.py`。
调整 `_ROUGH_TERRAIN_CFG` 即可修改地形，Flat 配置不受影响。
沿用 D1 Rough 的 8×8 m 地块、10 行×20 列难度地图和 20 m 边界：

| 地形 | 比例 | 参数范围 |
| --- | --- | --- |
| 平地 | 20% | 平面 |
| 金字塔台阶 | 20% | 每阶高度 0–0.2 m，踏面 0.3 m，平台 3 m |
| 倒金字塔台阶 | 20% | 每阶高度 0–0.2 m，踏面 0.3 m，平台 3 m |
| 正坡 / 反坡 | 各 10% | 坡度 0–1.0，平台 2 m |
| 随机起伏 | 10% | noise_range=(0.02, 0.1)，noise_step=0.02 |
| 波浪 | 10% | amplitude_range=(0, 0.2)，num_waves=4 |

训练初始最高等级为 2，开启距离驱动的地形课程；play 分布在所有等级上。
重置姿态参考 D1 Rough：z 在默认高度与地形原点之上额外偏移 0.2–0.5 m，roll/pitch 为 ±0.5 rad。
机器人模型、关节顺序、PD、观测、奖励及约束采用 TITATIT 参数；纵向命令课程从 ±1.0 扩展至 ±1.5 m/s，横向为 0。
Rough 算法配置位于 `titatit/__init__.py` 的 `_TITATIT_ROUGH_RL_CFG`：默认 15000 轮，entropy_coef=0.003，init_noise_std=0.8。
日志保存到 `logs/np3o/titatit_rough/`，自动包含训练配置和启动命令。台阶参数范围不代表策略已经具备对应通行能力，需训练与回放验证。

## Resume training

```bash
python scripts/train.py Mjlab-Velocity-Flat-D1 \
    --num_envs=4096 \
    --resume \
    --load_run logs/np3o/d1_flat/YYYY-MM-DD_HH-MM-SS/model_xx.pt
```

---

## Play / Evaluate

```bash
# Play with a specific checkpoint
python scripts/play.py Mjlab-Velocity-Flat-TITATIT \
    --checkpoint-file logs/np3o/titatit_flat/YYYY-MM-DD_HH-MM-SS/model_6000.pt
```

---

## Update GitHub

Training logs are ignored by `.gitignore`. Review, commit and upload code changes with:

```bash
git status
git add -A
git commit -m "Describe this update"
git push
```

---

## Available robots & tasks

| Robot        | Description           | Flat task                  | Rough task                  |
| ------------ | --------------------- | -------------------------- | --------------------------- |
| **D1** | Quadruped with wheels | `Mjlab-Velocity-Flat-D1` | `Mjlab-Velocity-Rough-D1` |
| **TITATIT** | Quadruped with wheels | `Mjlab-Velocity-Flat-TITATIT` | `Mjlab-Velocity-Rough-TITATIT` |

---

## Project Structure

Key source files:

```
np3o/
├── algorithms/np3o/              # NP3O algorithm, BarlowTwins actor-critic, runner, wrapper
│   ├── actor_critic.py
│   ├── np3o.py
│   ├── runner.py
│   └── wrapper.py
└── tasks/locomotion/
    ├── cmdp/                     # Commands, costs, rewards, observations, terminations, curriculums
    │   ├── commands.py
    │   ├── costs.py
    │   ├── rewards.py
    │   ├── observations.py
    │   ├── terminations.py
    │   └── curriculums.py
    ├── config/                   # Per-robot, per-terrain env configs
    │   └── d1/
    └── rl/                       # Shared NP3O runner hyper-parameters
        └── rl_cfg.py
```

---

## Task Registration

Tasks are registered automatically via `mjlab.tasks.registry.register_mjlab_task`.

When you run a script, the side-effect import

```python
import np3o.tasks.locomotion  # noqa: F401
```

triggers `np3o/tasks/locomotion/__init__.py` to auto-discover every package under `config/` (e.g. `d1/`) and import them. Each package then registers its flat / rough tasks.

### Adding your own tasks

Suppose you want a custom D1 task with different rewards. Create a new package under `config/` (e.g. `my_tasks/`) and reuse the existing D1 assets:

```python
# np3o/tasks/locomotion/config/my_tasks/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from np3o.algorithms.np3o.runner import MjlabNP3ORunner

# Reuse existing D1 env configs, or import your own variants
from ..d1.d1_flat_cfg import d1_flat_env_cfg, d1_flat_play_env_cfg
from ...rl.rl_cfg import d1_np3o_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Mytask-D1",
    env_cfg=d1_flat_env_cfg(play=False),
    play_env_cfg=d1_flat_play_env_cfg(),
    rl_cfg=d1_np3o_runner_cfg(experiment_name="mytask_d1", max_iterations=10000),
    runner_cls=MjlabNP3ORunner,
)
```

No manual per-package import is needed in `train.py` or `play.py` — `np3o/tasks/locomotion/__init__.py` auto-discovers `my_tasks/` at runtime.

Verify that your task is registered:

```bash
python scripts/train.py --list-tasks
```
