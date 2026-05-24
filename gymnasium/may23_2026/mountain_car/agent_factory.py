"""Build or resume PPO / SAC models from variant configs."""

from __future__ import annotations

import os
import sys

from stable_baselines3 import PPO, SAC

from variants import PPOVariant, SACVariant, Variant

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_utils import _create_or_resume_with_handlers  # noqa: E402


def create_or_resume(variant: Variant, env, logs_dir: str, models_dir: str, *, resume: bool, resume_from: str | None):
    if isinstance(variant, PPOVariant):
        return _create_or_resume_with_handlers(
            env,
            logs_dir,
            variant.model_prefix,
            models_dir,
            resume=resume,
            resume_from=resume_from,
            label="PPO",
            new_model=lambda: PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=logs_dir,
                ent_coef=variant.ent_coef,
                learning_rate=variant.learning_rate,
                n_steps=variant.n_steps,
                batch_size=variant.batch_size,
                n_epochs=variant.n_epochs,
                gamma=variant.gamma,
                gae_lambda=variant.gae_lambda,
                clip_range=variant.clip_range,
            ),
            load_model=lambda path: _load_ppo(path, env, logs_dir, variant.ent_coef),
        )

    if isinstance(variant, SACVariant):
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

    raise TypeError(f"Unsupported variant type: {type(variant)!r}")


def _load_ppo(path: str, env, logs_dir: str, ent_coef: float) -> PPO:
    model = PPO.load(path, env=env, tensorboard_log=logs_dir, verbose=1)
    model.ent_coef = ent_coef
    return model
