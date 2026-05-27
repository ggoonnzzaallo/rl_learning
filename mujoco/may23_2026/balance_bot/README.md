# Balance bot (MuJoCo + SAC)

Two-wheeled inverted pendulum in MuJoCo. Train a SAC policy in simulation to keep the robot upright.

> **Before SAC training:** run your PID in sim and confirm behaviour matches the real robot.
> See [Validate sim with PID](#validate-sim-with-pid) below.

## Local setup

From the repo root:

```bash
cd rl_learning
uv sync
cd mujoco/may23_2026/balance_bot
```

## World coordinates (simulator)

MuJoCo uses a **right-handed** frame fixed to the world (see the coloured capsules at the floor origin when you open the viewer). The **yellow sphere** on the chassis marks the **measured body center of mass** (`body_com_z` in [`robot_params.py`](robot_params.py), MJCF `site body_cog` on the `chassis` body — it pitches with the robot).

| Axis | Colour | Meaning in this model |
|------|--------|------------------------|
| **+X** | Red | Cart **slide** direction (`joint slide` axis `1 0 0`). Forward/back along the track. |
| **+Y** | Green | Lateral direction; **wheel axle** lies along **Y** (left wheel at negative **Y**, right at positive **Y**). |
| **+Z** | Blue | **Up**. Gravity is **`(0, 0, -9.81)`** in [`assets/balance_bot.xml.template`](assets/balance_bot.xml.template) (`<option gravity="0 0 -9.81" .../>`). |

The axis capsules are **visual only** (`contype="0" conaffinity="0"`). The model is regenerated from the template when the env starts (`robot_params.render_xml()`).

### Contact / collision visualization

When **`show_contact_viz=True`** in [`robot_params.py`](robot_params.py) (default), the human viewer draws MuJoCo’s built-in contact debug geometry:

| Visual | Colour | Meaning |
|--------|--------|---------|
| **Contact points** | Red disks | Where two geoms touch (e.g. wheel–floor, body–floor if it hits) |
| **Contact force arrows** | Orange | Normal contact force direction and relative magnitude |

Styling is in the MJCF `<visual>` block ([`assets/balance_bot.xml.template`](assets/balance_bot.xml.template)). Press **`C`** in the Gymnasium viewer to toggle. `poke.py` enables contacts on launch.

## Viewer camera controls

The Gymnasium viewer **tracks the chassis** as the robot moves along the track (MuJoCo `mjCAMERA_TRACKING` on the `chassis` body). You can still **orbit, pan, and zoom** with the mouse; distance / angle defaults are in `DEFAULT_CAMERA_CONFIG` in [`env.py`](env.py).

Click the **MuJoCo window** first so it has keyboard/mouse focus.

| Input | Action |
|-------|--------|
| **Left drag** | Orbit / spin camera (vertical drag = elevation, horizontal = azimuth) |
| **Shift + left drag** | Orbit the other axis (swap horizontal vs vertical) |
| **Right drag** | Pan (move look-at point) |
| **Shift + right drag** | Pan the other axis |
| **Scroll wheel** | Zoom in / out |
| **Tab** | Cycle fixed cameras (returns to free camera after the last one) |
| **Backspace** | **Reset simulation** (upright pose + noise, re-applies `initial_hinge_rad` / CLI `--tilt-deg`; works while paused) |
| **C** | Toggle **contact** visualization (red points + force arrows; on by default via `show_contact_viz`) |
| **H** | Hide/show the on-screen help overlay |

On macOS, **right-drag** may need a two-finger click, or **Control + left-click**, depending on your trackpad/mouse settings.

The sim keeps running while you move the camera — you can orbit anytime during `watch.py`, `explore.py`, or `play.py`.

## Applying forces by clicking (poke / push)

If you want to **click the robot and push it with a force**, use:

```bash
uv run python poke.py
```

This launches MuJoCo's built-in interactive viewer (different from the Gymnasium viewer).
Press **`H`** in the window to show the on-screen help overlay with the exact mouse
controls for *perturb / apply force* (the modifier keys can vary by platform).

## Viewer window size

All scripts that use **`gym.make("BalanceBot-v0", render_mode="human")`** (`explore.py`, `play.py`, `pid_baseline.py`, `train.py`, `watch.py`) read **`render_width` and `render_height`** from [`robot_params.py`](robot_params.py) (default **1280 × 720**).

`poke.py` uses MuJoCo’s `launch_passive` viewer; its window size is **not** controlled by `robot_params.py` (MuJoCo sets that internally — resize once or use full-screen if your build supports it).

With `render_mode="human"`, the env prints **hinge angle θ** (same as `obs[1]` in rad and °), **θ̇** (`obs[3]`), **normalized drive** L/R/avg in `[-1, 1]`, and **commanded cart speed** in m/s (~`motor_log_hz` in `robot_params.py`, default 10 Hz). Pass `log_motor_speeds=False` to `BalanceBotEnv` to disable.

## Physical parameters — edit this file

**File to modify:** [`robot_params.py`](robot_params.py)

Change the numbers in the `RobotParams` dataclass (or pass a custom instance to `BalanceBotEnv(params=...)`). After saving, the next `explore.py` / `watch.py` / `train.py` run regenerates `assets/balance_bot.xml` automatically.

| Parameter | Unit | What it is |
|-----------|------|------------|
| `body_half_width` | m | Half of chassis length (x, direction of travel) |
| `body_half_depth` | m | Half of chassis thickness (y, between the wheels) |
| `body_half_height` | m | Half of chassis height (z, vertical) |
| `body_mass` | kg | Chassis mass |
| `wheel_radius` | m | Wheel radius |
| `wheel_half_width` | m | Half of wheel thickness (along y) |
| `wheel_mass` | kg | Mass per wheel |
| `wheelbase` | m | Centre-to-centre distance between left and right wheels |
| `max_cart_speed_m_s` | m/s | Cart speed when mean drive command = **+1** |
| `slide_velocity_kv` | — | Velocity servo stiffness on `slide` (default **200**; raise if tracking feels soft) |

Hardware motor specs (gear ratio, stall torque, RPM) are **not** used in sim — see [`actual_robot.py`](actual_robot.py) for the Pi.

### Actuation vs wheel visuals

**mean(τ_l, τ_r)** in `[-1, 1]` sets **target cart speed** `ẋ_cmd = mean × max_cart_speed_m_s` on the slide joint (ideal drive, no torque limit). The **wheel cylinders are non‑jointed geoms** on the cart. The **chassis** tilts on the **`hinge`** as the cart accelerates.

Example — measure your robot, then set:

```python
DEFAULT_PARAMS = RobotParams(
    body_half_width=0.10,
    body_half_depth=0.04,
    body_half_height=0.15,
    body_mass=1.2,
    wheel_radius=0.032,
    wheel_half_width=0.012,
    wheel_mass=0.08,
    wheelbase=0.14,
)
```

Do **not** edit `assets/balance_bot.xml` by hand; it is generated from `assets/balance_bot.xml.template` + `robot_params.py`.

### Lighter rendering (MacBook Air)

Also in `robot_params.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `low_gpu_render` | `True` | Disables reflections, shadows, fog in the viewer |
| `show_contact_viz` | `True` | Red contact points + orange force arrows in the human viewer |
| `render_width` / `render_height` | 1280 × 720 | MuJoCo viewer window (Gymnasium scripts) |

Set `low_gpu_render=False` if you want full-quality visuals on a faster machine.

## Validate sim with PID

1. Port your tuned PID gains into [`pid_controller.py`](pid_controller.py).
2. Run in the MuJoCo viewer and compare to real-robot behaviour (balance, wobble, recovery, motor effort):

```bash
uv run python pid_baseline.py
```

3. Batch metrics:

```bash
uv run python pid_baseline.py --compare --episodes 10
```

4. If sim and reality don't match, tune `robot_params.py` first — **do not train SAC until PID looks right.**

| Check | Pass if… |
|-------|----------|
| Balance | Holds upright with your real-world gains |
| Recovery | Similar response to a small push |
| Oscillation | Similar wobble — not wildly different |
| Motor effort | Not pegged constantly |

## Train SAC

```bash
uv run python explore.py          # random actions + viewer
uv run python watch.py            # zero motors, runs until Ctrl+C (no tilt cutoff)
uv run python pid_baseline.py       # PID validation gate
uv run python train.py              # SAC; opens TensorBoard; resumes by default
uv run python train.py --fresh      # new random policy
uv run python play.py               # watch trained agent
```

Default model: `models/default/sac_balance_bot_final.zip`

TensorBoard:

```bash
uv run tensorboard --logdir=logs
```

Watch `rollout/ep_rew_mean` and `rollout/ep_len_mean` (longer = better balance).

## Docs

- [`ENV.md`](ENV.md) — observation, action, reward, termination
- [`DEPLOY_NOTES.md`](DEPLOY_NOTES.md) — robot deploy checklist (future)

## Future work — model realism

The v1 model is **simplified** (slide + hinge dynamics; wheels are visual only). Next steps after SAC trains:

1. **Friction and contact** — wheel–ground contact; contact-driven wheel torques instead of differential-force abstraction
2. **Motor dynamics** — armature, damping, frictionloss, gear/backlash
3. **6-DOF free-joint body** — full floating chassis
4. **Domain randomization** — randomize friction, mass, motor gain during training
5. **Sensor noise** — inject IMU noise in `_get_obs()` before sim-to-real
6. **System identification** — tune sim params against hardware logs
