# Mountain Car environment reference

This folder trains **both** Gymnasium variants for comparison:

| Variant | `gymnasium.make(...)` | Action space | Algorithm |
|---------|------------------------|--------------|-----------|
| **Discrete** | `MountainCar-v0` | `Discrete(3)` — left / no push / right | **PPO** |
| **Continuous** | `MountainCarContinuous-v0` | `Box(-1, 1)` — throttle | **SAC** |

Docs: [Mountain Car (discrete)](https://gymnasium.farama.org/environments/classic_control/mountain_car/) · [Mountain Car Continuous](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)

## Description

A car sits in a sinusoidal valley with not enough power to drive straight up the right hill. The agent must **rock back and forth** to build momentum and reach the flag on the right.

Harder than CartPole: reward is **sparse** (penalty every step until success).

## Observation space (both variants)

Shape `(2,)`:

| Num | Observation | Min | Max |
|-----|-------------|-----|-----|
| 0 | Car position (x) | -1.2 | 0.6 |
| 1 | Car velocity | -0.07 | 0.07 |

## Action space — discrete (`MountainCar-v0`)

| Num | Action |
|-----|--------|
| 0 | Accelerate left |
| 1 | No acceleration |
| 2 | Accelerate right |

**Episode limit:** 200 steps (truncation).

## Action space — continuous (`MountainCarContinuous-v0`)

One float in `[-1, 1]`: negative = push left, positive = push right (scaled force).

**Episode limit:** 999 steps (truncation).

## Reward (both)

**-1** for every timestep until the car reaches the goal. Fewer steps ⇒ higher (less negative) total reward.

- Random agent: often **-200** (discrete, hits step limit) or **-999** (continuous).
- Good agent: much closer to **0** (reaches flag quickly).
- Discrete “solved” benchmark: average reward **≥ -110** over 100 episodes ([Gymnasium](https://gymnasium.farama.org/environments/classic_control/mountain_car/)).

## Episode end

1. **Termination:** position ≥ **0.5** (flag on right hill)
2. **Truncation:** max episode length (200 discrete / 999 continuous)

## Starting state

Position uniform in **[-0.6, -0.4]**, velocity **0**.

## Comparing discrete vs continuous in this repo

```bash
# Train (discrete = PPO, continuous = SAC)
uv run python train.py --env discrete
uv run python train.py --env continuous

# Training resumes from ..._final.zip by default (if it exists)
uv run python train.py --env continuous --fresh   # new SAC run (ignores old PPO zips)
uv run python train.py --env continuous --resume-from models/continuous/sac_mountain_car_continuous_500000

# TensorBoard: overlay learning curves
uv run tensorboard --logdir=logs

# Play each trained policy
uv run python play.py --env discrete
uv run python play.py --env continuous
```

Artifacts:

```
models/discrete/ppo_mountain_car_discrete_final.zip
models/continuous/sac_mountain_car_continuous_final.zip
logs/discrete/
logs/continuous/
```

Older `ppo_mountain_car_continuous_*.zip` checkpoints are from PPO runs; SAC uses the `sac_` prefix.

## Scripts

| Script | Purpose |
|--------|---------|
| `explore.py` | Random agent + re-run UI (`--env discrete\|continuous`) |
| `train.py` | PPO training (`--env` required) |
| `play.py` | Watch trained agent (`--env` must match training) |
| `variants.py` | `PPOVariant` / `SACVariant` hyperparameters (all passed to SB3) |
| `agent_factory.py` | Builds `PPO` or `SAC` from variant config |

## Hyperparameters (`variants.py`)

### Shared (both variants)

| Field | Continuous default | Meaning |
|-------|-------------------|---------|
| `timesteps_per_iter` | 50_000 | Steps per `learn()` chunk |
| `n_iters` | 20 | Chunks per `train.py` run → **1M** total |

### `SACVariant` (continuous only)

All fields are passed to `stable_baselines3.SAC`. SB3 defaults match these unless you override.

| Field | Default | What it controls |
|-------|---------|------------------|
| `learning_rate` | `3e-4` | Adam LR (actor + critics) |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `learning_starts` | `1000` | Random steps before gradient updates |
| `batch_size` | `256` | Minibatch size per update |
| `tau` | `0.005` | Soft target network update rate |
| `gamma` | `0.99` | Discount factor |
| `train_freq` | `1` | Train every N env steps |
| `gradient_steps` | `1` | Gradient steps per train call (`-1` = match rollout length) |
| `ent_coef` | `"auto"` | Entropy coefficient (`"auto"`, `"auto_0.1"`, or float) |
| `target_entropy` | `"auto"` | Target entropy when `ent_coef="auto"` |
| `target_update_interval` | `1` | How often target nets update |
| `n_steps` | `1` | N-step returns in replay buffer |
| `net_arch` | `(256, 256)` | Hidden layer sizes (actor + critic MLP) |

### `PPOVariant` (discrete only)

| Field | Default | What it controls |
|-------|---------|------------------|
| `ent_coef` | `0.0` | Exploration bonus |
| `learning_rate` | `3e-4` | Policy / value LR |
| `n_steps` | `2048` | Steps per rollout before update |
| `batch_size` | `64` | Minibatch size |
| `n_epochs` | `10` | Epochs over rollout per update |
| `gamma` | `0.99` | Discount |
| `gae_lambda` | `0.95` | GAE factor |
| `clip_range` | `0.2` | PPO clip range |
