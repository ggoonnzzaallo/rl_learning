"""Balance bot SAC training config."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal


@dataclass(frozen=True)
class SACVariant:
    name: str = "default"
    env_id: str = "BalanceBot-v0"
    timesteps_per_iter: int = 10_000
    n_iters: int = 2
    tb_log_name: str = "SAC_balance_bot"
    model_prefix: str = "sac_balance_bot"

    learning_rate: float = 3e-4
    buffer_size: int = 300_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto_0.1"
    target_entropy: str | float = "auto"
    target_update_interval: int = 1
    n_steps: int = 1
    net_arch: tuple[int, ...] = (256, 256)

    @property
    def algorithm(self) -> Literal["sac"]:
        return "sac"

    @property
    def total_timesteps(self) -> int:
        return self.timesteps_per_iter * self.n_iters

    def policy_kwargs(self) -> dict:
        return {"net_arch": list(self.net_arch)}


DEFAULT_VARIANT = SACVariant()


def format_variant_hyperparams(variant: SACVariant) -> str:
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
