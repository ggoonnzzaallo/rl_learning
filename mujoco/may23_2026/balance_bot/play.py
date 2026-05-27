"""
Watch a trained SAC balance bot agent.

  uv run python play.py
  uv run python play.py --model models/default/sac_balance_bot_final
"""

from __future__ import annotations

import argparse
import os

import env  # noqa: F401
import gymnasium as gym
from stable_baselines3 import SAC

from variants import DEFAULT_VARIANT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODES = 10


def default_model_path() -> str:
    v = DEFAULT_VARIANT
    return os.path.join(SCRIPT_DIR, "models", v.name, f"{v.model_prefix}_final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play trained SAC balance bot")
    parser.add_argument("--model", default=None, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    if not model_path.endswith(".zip") and os.path.exists(f"{model_path}.zip"):
        model_path = f"{model_path}.zip"

    env_id = DEFAULT_VARIANT.env_id
    gym_env = gym.make(env_id, render_mode="human")
    model = SAC.load(model_path, env=gym_env)

    print(f"Env: {env_id}")
    print(f"Model: {model_path}\n")

    for ep in range(1, args.episodes + 1):
        obs, _ = gym_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_theta = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            total_reward += reward
            steps += 1
            max_theta = max(max_theta, abs(float(obs[1])))
            done = terminated or truncated

        print(
            f"Episode {ep}: reward={total_reward:.1f}, steps={steps}, "
            f"max|θ|={max_theta:.3f} rad"
        )

    gym_env.close()


if __name__ == "__main__":
    main()
