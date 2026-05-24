"""Mountain Car training configs — PPO (discrete) and SAC (continuous)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, Union


@dataclass(frozen=True)
class BaseVariant:
    name: str
    env_id: str
    timesteps_per_iter: int
    n_iters: int
    tb_log_name: str
    model_prefix: str

    @property
    def total_timesteps(self) -> int:
        return self.timesteps_per_iter * self.n_iters


@dataclass(frozen=True)
class PPOVariant(BaseVariant):
    """Hyperparameters passed to stable_baselines3.PPO."""

    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    @property
    def algorithm(self) -> Literal["ppo"]:
        return "ppo"


@dataclass(frozen=True)
class SACVariant(BaseVariant):
    """Hyperparameters passed to stable_baselines3.SAC."""

    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto_0.1"  # "auto", "auto_0.1", or fixed float e.g. 0.2
    target_entropy: str | float = "auto"
    target_update_interval: int = 1
    n_steps: int = 1  # n-step returns in replay buffer (SB3 default: 1)
    net_arch: tuple[int, ...] = (256, 256)  # MLP hidden layers (actor + critic)

    @property
    def algorithm(self) -> Literal["sac"]:
        return "sac"

    def policy_kwargs(self) -> dict:
        return {"net_arch": list(self.net_arch)}


Variant = Union[PPOVariant, SACVariant]


VARIANTS: dict[str, Variant] = {
    "discrete": PPOVariant(
        name="discrete",
        env_id="MountainCar-v0",
        timesteps_per_iter=50_000,
        n_iters=10,
        tb_log_name="PPO_discrete",
        model_prefix="ppo_mountain_car_discrete",
    ),
    "continuous": SACVariant(
        name="continuous",
        env_id="MountainCarContinuous-v0",
        timesteps_per_iter=1_000,
        n_iters=25,
        tb_log_name="SAC_continuous",
        model_prefix="sac_mountain_car_continuous",
    ),
}


def get_variant(name: str) -> Variant:
    key = name.lower()
    if key not in VARIANTS:
        choices = ", ".join(VARIANTS)
        raise ValueError(f"Unknown variant {name!r}. Choose: {choices}")
    return VARIANTS[key]


def format_variant_hyperparams(variant: Variant) -> str:
    """Human-readable hyperparameters for train.py startup log."""
    lines = [
        f"timesteps_per_iter: {variant.timesteps_per_iter}",
        f"n_iters: {variant.n_iters}",
        f"total_timesteps: {variant.total_timesteps:,}",
    ]
    skip = {"name", "env_id", "timesteps_per_iter", "n_iters", "tb_log_name", "model_prefix"}
    for f in fields(variant):
        if f.name in skip:
            continue
        lines.append(f"{f.name}: {getattr(variant, f.name)}")
    return "\n".join(f"  {line}" for line in lines)
