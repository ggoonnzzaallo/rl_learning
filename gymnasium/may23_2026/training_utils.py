"""Shared helpers for cartpole / mountain_car train.py scripts."""

from __future__ import annotations

import os
import re
from typing import Literal, Union

from stable_baselines3 import PPO, SAC

AlgorithmName = Literal["ppo", "sac"]
Model = Union[PPO, SAC]


def resolve_checkpoint_path(
    models_dir: str,
    model_prefix: str,
    *,
    resume: bool,
    resume_from: str | None,
) -> str:
    if resume_from:
        path = resume_from
        if not path.endswith(".zip"):
            path = f"{path}.zip"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    if not resume:
        raise RuntimeError("resolve_checkpoint_path called without resume")

    final = os.path.join(models_dir, f"{model_prefix}_final.zip")
    if os.path.isfile(final):
        return final

    pattern = re.compile(rf"^{re.escape(model_prefix)}_(\d+)\.zip$")
    best_steps = -1
    best_path = None
    if os.path.isdir(models_dir):
        for name in os.listdir(models_dir):
            match = pattern.match(name)
            if match:
                steps = int(match.group(1))
                if steps > best_steps:
                    best_steps = steps
                    best_path = os.path.join(models_dir, name)

    if best_path is None:
        raise FileNotFoundError(
            f"No checkpoint to resume in {models_dir}. "
            f"Train once without --resume, or pass --resume-from PATH."
        )
    return best_path


def create_or_resume_ppo(
    env,
    logs_dir: str,
    model_prefix: str,
    models_dir: str,
    *,
    resume: bool,
    resume_from: str | None,
    ent_coef: float = 0.0,
) -> PPO:
    def new_model() -> PPO:
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=logs_dir,
            ent_coef=ent_coef,
        )

    def load_model(path: str) -> PPO:
        model = PPO.load(path, env=env, tensorboard_log=logs_dir, verbose=1)
        model.ent_coef = ent_coef
        return model
    if resume_from:
        path = resolve_checkpoint_path(
            models_dir,
            model_prefix,
            resume=True,
            resume_from=resume_from,
        )
        model = load_model(path)
        print(f"Resuming from: {path}")
        print(f"  Timesteps so far: {model.num_timesteps:,}")
        print(f"  ent_coef: {ent_coef}\n")
        return model

    if resume:
        try:
            path = resolve_checkpoint_path(
                models_dir,
                model_prefix,
                resume=True,
                resume_from=None,
            )
        except FileNotFoundError:
            print("No saved checkpoint found — starting fresh.\n")
            return new_model()

        model = load_model(path)
        print(f"Resuming from: {path}")
        print(f"  Timesteps so far: {model.num_timesteps:,}")
        print(f"  ent_coef: {ent_coef}\n")
        return model

    print("Mode: fresh (new random policy)\n")
    return new_model()


def _create_or_resume_with_handlers(
    env,
    logs_dir: str,
    model_prefix: str,
    models_dir: str,
    *,
    resume: bool,
    resume_from: str | None,
    label: str,
    new_model,
    load_model,
) -> Model:
    """Shared resume logic for mountain car PPO / SAC."""

    if resume_from:
        path = resolve_checkpoint_path(
            models_dir,
            model_prefix,
            resume=True,
            resume_from=resume_from,
        )
        model = load_model(path)
        print(f"Resuming {label} from: {path}")
        print(f"  Timesteps so far: {model.num_timesteps:,}\n")
        return model

    if resume:
        try:
            path = resolve_checkpoint_path(
                models_dir,
                model_prefix,
                resume=True,
                resume_from=None,
            )
        except FileNotFoundError:
            print(f"No saved {label} checkpoint found — starting fresh.\n")
            return new_model()

        model = load_model(path)
        print(f"Resuming {label} from: {path}")
        print(f"  Timesteps so far: {model.num_timesteps:,}\n")
        return model

    print(f"Mode: fresh ({label}, new random policy)\n")
    return new_model()
