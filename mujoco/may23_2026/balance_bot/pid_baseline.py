"""
Run your PID controller in the MuJoCo sim — validate before SAC training.

  uv run python pid_baseline.py              # viewer, interactive loop
  uv run python pid_baseline.py --compare    # batch metrics, no viewer

Compare sim behaviour to what you see on the real robot. Tune robot_params.py
and pid_controller.py until they qualitatively match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import env  # noqa: F401
import gymnasium as gym
import numpy as np

from pid_controller import BalanceBotPID


@dataclass
class EpisodeMetrics:
    steps: int
    total_reward: float
    max_abs_theta: float
    mean_control_effort: float
    mean_action_jerk: float
    saturation_fraction: float


def run_episode(
    gym_env,
    controller: BalanceBotPID,
    *,
    render: bool,
) -> EpisodeMetrics:
    controller.reset()
    obs, _ = gym_env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    max_theta = 0.0
    efforts: list[float] = []
    jerks: list[float] = []
    saturations: list[float] = []

    while not done:
        action = controller.compute(obs)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        total_reward += reward
        steps += 1
        max_theta = max(max_theta, abs(float(obs[1])))
        efforts.append(info.get("control_effort", 0.0))
        jerks.append(info.get("action_jerk", 0.0))
        saturations.append(info.get("saturation_fraction", 0.0))
        done = terminated or truncated
        if render:
            gym_env.render()

    return EpisodeMetrics(
        steps=steps,
        total_reward=total_reward,
        max_abs_theta=max_theta,
        mean_control_effort=float(np.mean(efforts)) if efforts else 0.0,
        mean_action_jerk=float(np.mean(jerks)) if jerks else 0.0,
        saturation_fraction=float(np.mean(saturations)) if saturations else 0.0,
    )


def print_metrics(ep: int, m: EpisodeMetrics) -> None:
    print(
        f"Episode {ep}: steps={m.steps}, reward={m.total_reward:.1f}, "
        f"max|θ|={m.max_abs_theta:.3f} rad, "
        f"effort={m.mean_control_effort:.3f}, jerk={m.mean_action_jerk:.3f}, "
        f"sat={m.saturation_fraction:.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PID baseline in BalanceBot sim")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run batch episodes and print aggregate metrics (no viewer)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=0.0,
        help="Initial forward tilt in degrees (e.g. 1.0)",
    )
    parser.add_argument(
        "--tilt-rad",
        type=float,
        default=0.0,
        help="Initial forward tilt in radians (overrides --tilt-deg)",
    )
    args = parser.parse_args()

    initial_theta_rad = float(args.tilt_rad) if args.tilt_rad != 0.0 else float(
        np.deg2rad(args.tilt_deg)
    )

    render_mode = None if args.compare else "human"
    gym_env = gym.make(
        "BalanceBot-v0",
        render_mode=render_mode,
        initial_hinge_rad=initial_theta_rad,
    )
    controller = BalanceBotPID()

    print("PID baseline — confirm sim matches real-world behaviour before SAC training.")
    if initial_theta_rad != 0.0:
        print(f"Initial tilt: {initial_theta_rad:.5f} rad")
    print("Tune robot_params.py and pid_controller.py as needed.\n")

    if args.compare:
        metrics: list[EpisodeMetrics] = []
        for ep in range(1, args.episodes + 1):
            m = run_episode(
                gym_env,
                controller,
                render=False,
            )
            metrics.append(m)
            print_metrics(ep, m)

        print("\n--- Aggregate ---")
        print(f"Mean steps: {np.mean([m.steps for m in metrics]):.0f}")
        print(f"Mean reward: {np.mean([m.total_reward for m in metrics]):.1f}")
        print(f"Mean max|θ|: {np.mean([m.max_abs_theta for m in metrics]):.3f} rad")
        print(f"Mean effort: {np.mean([m.mean_control_effort for m in metrics]):.3f}")
        print(f"Mean jerk: {np.mean([m.mean_action_jerk for m in metrics]):.3f}")
    else:
        try:
            ep = 1
            while True:
                m = run_episode(
                    gym_env,
                    controller,
                    render=True,
                )
                print_metrics(ep, m)
                ep += 1
        except KeyboardInterrupt:
            print("\nDone.")

    gym_env.close()


if __name__ == "__main__":
    main()
