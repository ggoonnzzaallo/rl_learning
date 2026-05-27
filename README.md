# rl_learning

Reinforcement learning experiments with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Python 3.12+, managed with [uv](https://docs.astral.sh/uv/).

## Project layout

```
gymnasium/
  may23_2026/              # SB3 learning track (CartPole, Mountain Car) + shared training_utils.py
  nov28_2024/              # Earlier experiments (Lunar Lander, NEAT, Snake, …)
  first_attempt/           # Archived NEAT + BipedalWalker scratch work
  tutorials.md             # Curated YouTube / reading links

mujoco/
  may23_2026/balance_bot/  # Custom two-wheeled inverted pendulum (MuJoCo + SAC)
  dec2_2024/               # MuJoCo Ant + PPO
  Archive/first_attempt/   # Archived early MuJoCo scripts

serialcom/dec1_2024/       # Arduino ↔ Python serial experiments (servo, basic I/O)
pimoroni/                  # Pimoroni Servo 2040 (MicroPython on board; see pimoroni/README.md)
```

NEAT experiments under `nov28_2024/cartpole/` and `bipedal/` are archived learning material. `first_attempt/` and `mujoco/Archive/` are older scratch code, not part of the main SB3 workflow.

## Local Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd rl_learning
uv sync
```

Optional extras:

```bash
uv sync --extra snake   # OpenCV for custom_snake/
uv sync --extra pico    # mpremote for Pimoroni Servo 2040
```

Always use `uv run` from the repo root (or `cd` into an example folder first).

## Running examples

### CartPole (PPO) — start here

```bash
cd gymnasium/may23_2026/cartpole
# See ENV.md for environment reference
uv run python explore.py          # random agent + re-run UI
uv run python train.py            # opens TensorBoard; resumes if checkpoint exists
uv run python train.py --fresh    # new random policy
uv run python play.py
```

CartPole and Mountain Car share checkpoint helpers in [`gymnasium/may23_2026/training_utils.py`](gymnasium/may23_2026/training_utils.py).

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
uv run python poke.py                  # MuJoCo interactive viewer (click to push)
uv run python train.py                 # SAC; resumes by default
uv run python train.py --fresh
uv run python play.py
```

| Doc | Contents |
|-----|----------|
| [`balance_bot/README.md`](mujoco/may23_2026/balance_bot/README.md) | Coordinates, camera, `robot_params.py`, PID gate, training |
| [`balance_bot/ENV.md`](mujoco/may23_2026/balance_bot/ENV.md) | Observation, action, reward, termination |
| [`balance_bot/DEPLOY_NOTES.md`](mujoco/may23_2026/balance_bot/DEPLOY_NOTES.md) | Sim-to-real checklist (future) |

Hardware-side PID / IMU code for the Raspberry Pi lives in `actual_robot.py` and `actual_robot_drive_mode.py` (run on the Pi, not via `uv run` in this repo).

Default SAC model: `models/default/sac_balance_bot_final.zip`

### Serial communication (Arduino)

Standalone experiments under `serialcom/dec1_2024/` — basic Python ↔ Arduino messaging and servo sketches. Not wired into the main `pyproject.toml` dependencies; install `pyserial` locally if you revisit these scripts.

### Pimoroni Servo 2040

MicroPython runs **on the board**; use Cursor to edit and `uv sync --extra pico` + `uv run mpremote` to run scripts. See [`pimoroni/README.md`](pimoroni/README.md).

## Dependencies

| Package | Used for |
|---------|----------|
| `gymnasium[mujoco]` | Environments (Box2D + MuJoCo extras) |
| `stable-baselines3` | PPO, SAC, A2C |
| `tensorboard` | Training logs |
| `box2d-py`, `pygame` | Lunar Lander, Mountain Car rendering |
| `mujoco` | Ant, balance bot |
| `opencv-python` (optional `[snake]` extra) | Custom Snake env |

## Gitignored artifacts

`models/`, `logs/`, `.venv/`, and `.mujoco_cache/` are not committed. Reproduce trained policies by running `train.py` in each example folder.
