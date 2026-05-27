"""Build or resume SAC models from variant config."""

from __future__ import annotations

import os
import sys

from stable_baselines3 import SAC

from variants import SACVariant

_GYMNASIUM_MAY23 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "gymnasium",
    "may23_2026",
)
sys.path.insert(0, _GYMNASIUM_MAY23)
from training_utils import _create_or_resume_with_handlers  # noqa: E402


def create_or_resume(
    variant: SACVariant,
    env,
    logs_dir: str,
    models_dir: str,
    *,
    resume: bool,
    resume_from: str | None,
):
    return _create_or_resume_with_handlers(
        env,
        logs_dir,
        variant.model_prefix,
        models_dir,
        resume=resume,
        resume_from=resume_from,
        label="SAC",
        new_model=lambda: SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=variant.learning_rate,
            buffer_size=variant.buffer_size,
            learning_starts=variant.learning_starts,
            batch_size=variant.batch_size,
            tau=variant.tau,
            gamma=variant.gamma,
            train_freq=variant.train_freq,
            gradient_steps=variant.gradient_steps,
            ent_coef=variant.ent_coef,
            target_entropy=variant.target_entropy,
            target_update_interval=variant.target_update_interval,
            n_steps=variant.n_steps,
            policy_kwargs=variant.policy_kwargs(),
        ),
        load_model=lambda path: SAC.load(path, env=env, tensorboard_log=logs_dir, verbose=1),
    )
