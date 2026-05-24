"""
Watch a trained PPO agent play CartPole-v1.

Environment reference: ENV.md

Run train.py first, then:
  uv run python play.py
  uv run python play.py --model models/ppo_cartpole_30000
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "CartPole-v1"
EPISODES = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "models", "ppo_cartpole_final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play CartPole with a trained PPO model")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to saved model (with or without .zip extension)",
    )
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()

    model_path = args.model
    if not model_path.endswith(".zip") and os.path.exists(f"{model_path}.zip"):
        model_path = f"{model_path}.zip"

    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(model_path, env=env)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep}: reward = {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    main()
