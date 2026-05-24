"""
Smoke test: random agent on Mountain Car (discrete or continuous).

  uv run python explore.py --env discrete
  uv run python explore.py --env continuous

Environment reference: ENV.md
"""

from __future__ import annotations

import argparse

import gymnasium as gym

from ui import wait_after_episode
from variants import get_variant


def run_episode(env) -> float:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Mountain Car with random actions")
    parser.add_argument(
        "--env",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Which Mountain Car variant to run",
    )
    args = parser.parse_args()

    variant = get_variant(args.env)
    env = gym.make(variant.env_id, render_mode="human")

    obs, _ = env.reset()
    print(f"Variant: {variant.name} ({variant.env_id})")
    print("Observation:", obs)
    print("Action space:", env.action_space)

    while True:
        total_reward = run_episode(env)
        print(f"Episode finished. Total reward: {total_reward:.0f}")

        if wait_after_episode(env) == "quit":
            break

    env.close()


if __name__ == "__main__":
    main()
