# rl_learning

Reinforcement learning experiments with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

**Focus:** SB3 examples under `gymnasium/nov28_2024/` (start with `ppo_lander/`). NEAT experiments in `cartpole/` and `bipedal/` are archived learning material.

## Local Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd rl_learning
uv sync
```

Optional extras:

```bash
uv sync --extra snake   # OpenCV for custom_snake/
```

## Running examples

Always use `uv run` from the repo root (or `cd` into the example folder first):

```bash
# Lunar Lander (SB3 + A2C)
uv run python gymnasium/nov28_2024/ppo_lander/main.py

# Load a trained model
cd gymnasium/nov28_2024/ppo_lander
uv run python load.py

# TensorBoard (from an example dir that has logs/)
uv run tensorboard --logdir=logs
```

```bash
# Custom Snake (needs snake extra)
uv sync --extra snake
cd gymnasium/nov28_2024/custom_snake
uv run python sneklearn.py
```

```bash
# MuJoCo Ant
cd mujoco/dec2_2024
uv run python train.py
uv run python evaluate.py
```

## Dependencies

| Package | Used for |
|---------|----------|
| `gymnasium` | Environments |
| `stable-baselines3` | PPO, A2C, etc. |
| `tensorboard` | Training logs |
| `box2d-py`, `pygame` | Lunar Lander, BipedalWalker |
| `mujoco` | Ant (`mujoco/dec2_2024/`) |
| `opencv-python` (optional) | Custom Snake env |
