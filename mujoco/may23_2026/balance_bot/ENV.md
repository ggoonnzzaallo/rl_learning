# BalanceBot-v0 environment reference

> **Validate sim with PID before SAC training.** Run `uv run python pid_baseline.py` and
> confirm MuJoCo behaviour matches your real robot. See [README.md](README.md).

Custom MuJoCo env: simplified two-wheeled inverted pendulum (box body + two wheels).

## Description

A rectangular chassis balances on a wheel axle. The agent issues **normalized drive
commands** τ_l, τ_r ∈ [-1, 1]. Dynamics use a **slide + hinge** abstraction (v1): the
**average** of τ_l and τ_r sets a **target cart speed** on the slide joint (ideal velocity
drive — no motor torque / RPM model). Wheels are rendered but not contact-driven yet.

## `gym.make` / `BalanceBotEnv` options

| Argument | Default | Description |
|----------|---------|-------------|
| `initial_hinge_rad` | `0.0` | Hinge angle θ (rad) applied on **every** `reset()` and on **Backspace** in the human viewer (hinge reset noise is added on top). Matches `pid_baseline.py` `--tilt-deg` / `--tilt-rad` and `watch.py` tilt flags. |

## Observation space

Shape `(4,)`, float64:

| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0 | x | m | Position along track |
| 1 | θ | rad | Body pitch (tilt) |
| 2 | ẋ | m/s | Horizontal velocity |
| 3 | θ̇ | rad/s | Pitch rate |

With `render_mode="human"`, the env can print this **θ** / **θ̇** (same indices as the table) together with **normalized drive** and **commanded cart speed (m/s)** in the terminal; see [README.md](README.md) (`motor_log_hz`, `log_motor_speeds`).

## Action space

Shape `(2,)`, float32, range `[-1, 1]`:

| Index | Name | Description |
|-------|------|-------------|
| 0 | τ_l | Left drive command (normalized, [-1, 1]) |
| 1 | τ_r | Right drive command (normalized, [-1, 1]) |

Mapped to MuJoCo controls before simulation (1D slide — only the **mean** of τ_l and τ_r matters):

```
u = clip((τ_l + τ_r) / 2, -1, 1)
ẋ_cmd = u * max_cart_speed_m_s
ctrl = ẋ_cmd   →  velocity actuator on `slide` (high kv ≈ unlimited torque)
```

## Reward

Survival + shaping (motor-friendly):

```
reward = 1
         - k_theta * θ²
         - k_vel   * (ẋ² + θ̇²)
         - k_act   * (τ_l² + τ_r²)
         - k_dact  * ((τ_l - τ_l_prev)² + (τ_r - τ_r_prev)²)
```

Coefficients in [`robot_params.py`](robot_params.py). `k_dact` penalizes jerk (helps avoid
motor overheating from rapid reversals).

`info` dict each step:

| Key | Meaning |
|-----|---------|
| `reward_survive` | 1.0 if upright |
| `control_effort` | ‖action‖ |
| `action_jerk` | ‖action − prev_action‖ |
| `saturation_fraction` | 1.0 if any \|action_i\| > 0.9 |

## Termination

Episode ends when:

1. \|θ\| > `max_tilt_rad` (default 0.3 rad), or
2. Any observation is non-finite.

## Truncation

1000 steps (`max_episode_steps` at registration).

## Scripts

| Command | Description |
|---------|-------------|
| `uv run python watch.py` | Zero motor input — continuous until Ctrl+C, no fall cutoff |
| `uv run python explore.py` | Random actions |
| `uv run python pid_baseline.py` | PID validation |
| `uv run python train.py` | SAC training |
| `uv run python play.py` | Trained policy |

## Files

| File | Role |
|------|------|
| `robot_params.py` | Physical + reward parameters |
| `assets/balance_bot.xml.template` | MJCF template |
| `assets/balance_bot.xml` | Generated model (gitignored if desired) |
| `env.py` | `BalanceBotEnv` + registration |
| `variants.py` | SAC hyperparameters |
| `pid_controller.py` | PID gains (replace with your tuned values) |

## SAC hyperparameters

See [`variants.py`](variants.py) (`SACVariant`). Defaults: 1M total timesteps
(20 × 50k), `net_arch=[256, 256]`, `ent_coef="auto_0.1"`.
