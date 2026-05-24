"""
Train Mountain Car: PPO (discrete) or SAC (continuous).

  uv run python train.py --env discrete      # PPO
  uv run python train.py --env continuous    # SAC
  uv run python train.py --env continuous --fresh

Resumes from saved model by default. Opens TensorBoard in the browser.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser

import gymnasium as gym

from agent_factory import create_or_resume
from variants import Variant, format_variant_hyperparams, get_variant

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def paths_for(variant: Variant) -> tuple[str, str]:
    models_dir = os.path.join(SCRIPT_DIR, "models", variant.name)
    logs_dir = os.path.join(SCRIPT_DIR, "logs", variant.name)
    return models_dir, logs_dir


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
    print("  Watch: rollout/ep_rew_mean (higher is better, closer to 0 = faster to flag)")
    print("  Compare discrete vs continuous: train both, then --logdir=logs\n")
    webbrowser.open(url)
    return proc


def train_variant(
    variant: Variant,
    *,
    launch_tb: bool,
    resume: bool,
    resume_from: str | None,
) -> None:
    models_dir, logs_dir = paths_for(variant)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Variant: {variant.name}")
    print(f"  Env: {variant.env_id}")
    print(f"  Algorithm: {variant.algorithm.upper()}")
    print(format_variant_hyperparams(variant))
    if resume_from:
        print("  Mode: resume (explicit checkpoint)")
    elif resume:
        print("  Mode: resume if checkpoint exists, else fresh")
    else:
        print("  Mode: fresh (--fresh)")
    print()

    tb_proc = launch_tensorboard(os.path.join(SCRIPT_DIR, "logs")) if launch_tb else None

    env = gym.make(variant.env_id)
    model = create_or_resume(
        variant,
        env,
        logs_dir,
        models_dir,
        resume=resume,
        resume_from=resume_from,
    )

    try:
        for _ in range(1, variant.n_iters + 1):
            model.learn(
                total_timesteps=variant.timesteps_per_iter,
                reset_num_timesteps=False,
                tb_log_name=variant.tb_log_name,
            )
            checkpoint = os.path.join(
                models_dir, f"{variant.model_prefix}_{model.num_timesteps}"
            )
            model.save(checkpoint)
            print(f"Saved checkpoint: {checkpoint}")

        final_path = os.path.join(models_dir, f"{variant.model_prefix}_final")
        model.save(final_path)
        print(f"Saved final model: {final_path} ({model.num_timesteps:,} total timesteps)")
    finally:
        env.close()
        if tb_proc is not None:
            print("TensorBoard left running at http://localhost:6006/ — close that process when done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mountain Car (PPO or SAC)")
    parser.add_argument(
        "--env",
        choices=["discrete", "continuous"],
        required=True,
        help="MountainCar-v0 (3 actions) or MountainCarContinuous-v0 (throttle)",
    )
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

    train_variant(
        get_variant(args.env),
        launch_tb=not args.no_tensorboard,
        resume=not args.fresh,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
