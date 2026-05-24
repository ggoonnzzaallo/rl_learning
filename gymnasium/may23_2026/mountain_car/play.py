"""
Watch a trained Mountain Car agent (PPO discrete or SAC continuous).

  uv run python play.py --env discrete
  uv run python play.py --env continuous
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC

from variants import SACVariant, get_variant

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODES = 10


def default_model_path(variant_name: str) -> str:
    variant = get_variant(variant_name)
    return os.path.join(
        SCRIPT_DIR,
        "models",
        variant.name,
        f"{variant.model_prefix}_final",
    )


def load_model(variant, model_path: str, env):
    if isinstance(variant, SACVariant):
        return SAC.load(model_path, env=env)
    return PPO.load(model_path, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Mountain Car with a trained agent")
    parser.add_argument(
        "--env",
        choices=["discrete", "continuous"],
        required=True,
        help="Must match the variant used during training",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to saved model (default: models/<variant>/..._final)",
    )
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()

    variant = get_variant(args.env)
    model_path = args.model or default_model_path(args.env)
    if not model_path.endswith(".zip") and os.path.exists(f"{model_path}.zip"):
        model_path = f"{model_path}.zip"

    env = gym.make(variant.env_id, render_mode="human")
    model = load_model(variant, model_path, env)

    print(f"Variant: {variant.name} ({variant.env_id})")
    print(f"Algorithm: {variant.algorithm.upper()}")
    print(f"Model: {model_path}\n")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep}: reward = {total_reward:.0f}, steps = {steps}")

    env.close()


if __name__ == "__main__":
    main()
