"""
Passive sim: zero motor input, watch the robot fall under gravity.

Runs continuously until Ctrl+C — no tilt cutoff, no episode restart.

  uv run python watch.py
  uv run python watch.py --tilt 0.05   # start slightly off-vertical (radians)
"""

from __future__ import annotations

import argparse

import numpy as np

from env import BalanceBotEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch balance bot fall with zero motor input"
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=0.0,
        help="Initial pitch offset in radians (e.g. 0.05)",
    )
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=0.0,
        help="Initial pitch offset in degrees (e.g. 5)",
    )
    args = parser.parse_args()

    tilt = float(args.tilt) if args.tilt != 0.0 else float(np.deg2rad(args.tilt_deg))

    gym_env = BalanceBotEnv(
        render_mode="human",
        terminate_on_fall=False,
        initial_hinge_rad=tilt,
    )
    zero = np.zeros(2, dtype=np.float32)

    obs, _ = gym_env.reset()

    print("Passive mode: motor commands = [0, 0]")
    print("No tilt cutoff — sim runs until you press Ctrl+C.")
    print(f"Observation [x, θ, ẋ, θ̇]: {obs}")
    if tilt:
        print(f"Initial tilt: {tilt:.3f} rad")
    print()

    try:
        step = 0
        while True:
            obs, _, _, _, _ = gym_env.step(zero)
            gym_env.render()
            step += 1
            if step % 100 == 0:
                print(f"step {step:5d}  θ={obs[1]:+.3f} rad  x={obs[0]:+.3f} m")
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        gym_env.close()


if __name__ == "__main__":
    main()
