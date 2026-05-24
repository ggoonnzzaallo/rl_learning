"""
Train PPO on CartPole-v1.

Env reference: ENV.md
Opens TensorBoard in the browser by default. Use --no-tensorboard to skip.

  uv run python train.py
  uv run python train.py --fresh
"""

import argparse
import os
import subprocess
import sys
import webbrowser

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_utils import create_or_resume_ppo  # noqa: E402

ENV_ID = "CartPole-v1"
MODEL_PREFIX = "ppo_cartpole"
TIMESTEPS_PER_ITER = 10_000
N_ITERS = 5  # 50_000 per run (fresh or resume adds this many more)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")


def launch_tensorboard(logdir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            logdir,
            "--reload_interval",
            "1",
            "--bind_all",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = "http://localhost:6006/"
    print(f"\nTensorBoard: {url}")
    print("  Scalars to watch: rollout/ep_rew_mean, rollout/ep_len_mean")
    print("  (ep_len_mean → 500 means the pole is balanced for a full episode)\n")
    webbrowser.open(url)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from a new random policy (do not load checkpoints)",
    )
    parser.add_argument(
        "--resume-from",
        metavar="PATH",
        default=None,
        help="Load this checkpoint (implies resume)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not launch TensorBoard in the browser",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if args.resume_from:
        print("Mode: resume (explicit checkpoint)")
    elif not args.fresh:
        print("Mode: resume if checkpoint exists, else fresh")
    else:
        print("Mode: fresh (--fresh)")
    print(f"This run: {N_ITERS * TIMESTEPS_PER_ITER:,} timesteps\n")

    tb_proc = launch_tensorboard(LOGS_DIR) if not args.no_tensorboard else None

    env = gym.make(ENV_ID)
    model = create_or_resume_ppo(
        env,
        LOGS_DIR,
        MODEL_PREFIX,
        MODELS_DIR,
        resume=not args.fresh,
        resume_from=args.resume_from,
    )

    try:
        for _ in range(1, N_ITERS + 1):
            model.learn(
                total_timesteps=TIMESTEPS_PER_ITER,
                reset_num_timesteps=False,
                tb_log_name="PPO",
            )
            checkpoint = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}_{model.num_timesteps}")
            model.save(checkpoint)
            print(f"Saved checkpoint: {checkpoint}")

        final_path = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}_final")
        model.save(final_path)
        print(f"Saved final model: {final_path} ({model.num_timesteps:,} total timesteps)")
    finally:
        env.close()
        if tb_proc is not None:
            print("TensorBoard left running at http://localhost:6006/ — close that process when done.")


if __name__ == "__main__":
    main()
