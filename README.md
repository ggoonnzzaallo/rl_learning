# rl_learning

Reinforcement learning experiments with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Project layout

```
gymnasium/may23_2026/          # SB3 learning track (CartPole, Mountain Car)
gymnasium/nov28_2024/          # Earlier experiments (Lunar Lander, NEAT, Snake, …)
mujoco/may23_2026/             # MuJoCo custom envs (balance bot)
mujoco/dec2_2024/              # MuJoCo Ant + PPO
```

NEAT experiments under `nov28_2024/cartpole/` and `bipedal/` are archived learning material.

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

Always use `uv run` from the repo root (or `cd` into the example folder first).

### CartPole (PPO) — start here

```bash
cd gymnasium/may23_2026/cartpole
# See ENV.md for environment reference
uv run python explore.py          # random agent + re-run UI
uv run python train.py            # opens TensorBoard; resumes if checkpoint exists
uv run python train.py --fresh    # new random policy
uv run python play.py
```

### Mountain Car — discrete (PPO) vs continuous (SAC)

Harder than CartPole (sparse reward). Hyperparameters live in `variants.py` (`PPOVariant` / `SACVariant`).

```bash
cd gymnasium/may23_2026/mountain_car
uv run python explore.py --env discrete      # random actions
uv run python explore.py --env continuous

uv run python train.py --env discrete          # PPO
uv run python train.py --env continuous        # SAC
uv run python train.py --env continuous --fresh

uv run python play.py --env discrete           # loads ppo_*_final
uv run python play.py --env continuous         # loads sac_*_final

# TensorBoard (both variants under logs/)
uv run tensorboard --logdir=logs
```

| Variant | Env | Algorithm | Default model |
|---------|-----|-----------|---------------|
| `discrete` | `MountainCar-v0` | PPO | `models/discrete/ppo_mountain_car_discrete_final.zip` |
| `continuous` | `MountainCarContinuous-v0` | SAC | `models/continuous/sac_mountain_car_continuous_final.zip` |

Training **resumes from checkpoint by default**; use `--fresh` for a new run. Tune SAC exploration via `ent_coef` in `variants.py` (e.g. `"auto"`, `"auto_0.1"`, or a float).

### Lunar Lander (older script)

```bash
uv run python gymnasium/nov28_2024/ppo_lander/main.py
cd gymnasium/nov28_2024/ppo_lander && uv run python load.py
```

### Custom Snake (optional extra)

```bash
uv sync --extra snake
cd gymnasium/nov28_2024/custom_snake
uv run python sneklearn.py
```

### MuJoCo Ant

```bash
cd mujoco/dec2_2024
uv run python train.py
uv run python evaluate.py
```

### Balance bot (MuJoCo + SAC)

Custom two-wheeled inverted pendulum. **Validate sim with PID before training.**

```bash
cd mujoco/may23_2026/balance_bot
uv run python explore.py
uv run python watch.py                 # passive fall, zero motor input
uv run python pid_baseline.py          # compare sim to real robot behaviour
uv run python train.py                 # SAC; resumes by default
uv run python train.py --fresh
uv run python play.py
```

See [`mujoco/may23_2026/balance_bot/README.md`](mujoco/may23_2026/balance_bot/README.md) for physical params and future work (friction, contact, etc.).

## Dependencies

| Package | Used for |
|---------|----------|
| `gymnasium` | Environments |
| `stable-baselines3` | PPO, SAC, A2C |
| `tensorboard` | Training logs |
| `box2d-py`, `pygame` | Lunar Lander, Mountain Car rendering |
| `mujoco` | Ant, balance bot (`mujoco/may23_2026/balance_bot/`) |
| `opencv-python` (optional) | Custom Snake env |

## Gitignored artifacts

`models/`, `logs/`, and `.venv/` are not committed. Reproduce by running `train.py` in each example folder.
