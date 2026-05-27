"""
PID controller for the balance bot.

Tuned hardware values: Kp=3.8, Ki=0, Kd=0.2 on pitch (θ).

Interface matches BalanceBotEnv: obs = [x, θ, ẋ, θ̇], action = [τ_l, τ_r] in [-1, 1]
(normalized drive; env uses mean(τ_l, τ_r) as target cart speed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_params import DEFAULT_PARAMS


@dataclass
class PIDGains:
    """Pitch-angle PID (your real-robot tuning)."""

    kp: float = 4.8 # 3.8 in the real robot
    ki: float = 0.0
    kd: float = 0.7 # 0.2 in the real robot
    # Optional position loop — leave at 0 if you only balance in place
    kp_x: float = 0.0
    kd_x: float = 0.0
    max_command: float = 1.0  # clip differential command before splitting
    motor_sign: float = 1.0  # set to -1.0 if the robot reacts the wrong way


class BalanceBotPID:
    def __init__(
        self,
        gains: PIDGains | None = None,
        *,
        dt: float | None = None,
    ) -> None:
        self.gains = gains or PIDGains()
        self.dt = dt if dt is not None else DEFAULT_PARAMS.control_dt
        self._integral_theta = 0.0

    def reset(self) -> None:
        self._integral_theta = 0.0

    def compute(self, obs: np.ndarray) -> np.ndarray:
        """
        PID on pitch + optional PD on cart position → differential wheel torques.

        Returns [τ_l, τ_r] clipped to [-1, 1].
        """
        x, theta, x_dot, theta_dot = obs
        g = self.gains

        self._integral_theta += theta * self.dt
        u = (
            g.kp * theta
            + g.ki * self._integral_theta
            + g.kd * theta_dot
            + g.kp_x * x
            + g.kd_x * x_dot
        )
        u = float(np.clip(g.motor_sign * u, -g.max_command, g.max_command))

        # Balance bots drive BOTH wheels the same direction to accelerate under the CoM.
        # (Equal-and-opposite commands would be an in-place yaw command on a real robot,
        # and in this simplified sim it cancels the slide force term.)
        tau_l = u
        tau_r = u
        return np.array([tau_l, tau_r], dtype=np.float32)
