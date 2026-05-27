"""
Smoke test: random agent on BalanceBot-v0.

  uv run python explore.py

Environment reference: ENV.md
"""

from __future__ import annotations

import env  # noqa: F401
import gymnasium as gym


def run_episode(env) -> tuple[float, int]:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return total_reward, steps


def main() -> None:
    gym_env = gym.make("BalanceBot-v0", render_mode="human")
    obs, _ = gym_env.reset()
    print("Env: BalanceBot-v0")
    print("Observation [x, θ, ẋ, θ̇]:", obs)
    print("Action space (left/right motor):", gym_env.action_space)
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            total_reward, steps = run_episode(gym_env)
            print(f"Episode finished. reward={total_reward:.1f}, steps={steps}")
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        gym_env.close()


if __name__ == "__main__":
    main()
