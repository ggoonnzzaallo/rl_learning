"""Custom Gymnasium MuJoCo env: two-wheeled inverted pendulum (simplified)."""

from __future__ import annotations

import time

import glfw
import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register

from robot_params import DEFAULT_PARAMS, RobotParams, render_xml

# Applied when the GLFW viewer is created; tracking body id is set in _configure_human_viewer.
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "elevation": -12.0,
    "azimuth": 90.0,
}


class BalanceBotEnv(MujocoEnv, utils.EzPickle):
    """Simplified balance bot: slide + hinge; mean(τ_l, τ_r) commands cart speed (ideal drive)."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        params: RobotParams = DEFAULT_PARAMS,
        render_mode: str | None = None,
        terminate_on_fall: bool = True,
        log_motor_speeds: bool | None = None,
        initial_hinge_rad: float = 0.0,
        **kwargs,
    ):
        self.params = params
        self.terminate_on_fall = terminate_on_fall
        self.initial_hinge_rad = float(initial_hinge_rad)
        if log_motor_speeds is None:
            log_motor_speeds = render_mode == "human"
        self.log_motor_speeds = log_motor_speeds
        self._last_motor_log_time = 0.0
        xml_path = render_xml(params)
        utils.EzPickle.__init__(
            self,
            params,
            render_mode,
            terminate_on_fall,
            log_motor_speeds,
            initial_hinge_rad,
            **kwargs,
        )

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self._prev_action = np.zeros(2, dtype=np.float64)
        self._viewer_tuned = False
        self._viewer_key_hooked = False
        self._viewer_just_reset = False
        self._camera_track_body_id: int | None = None

        MujocoEnv.__init__(
            self,
            xml_path,
            params.frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=params.render_width,
            height=params.render_height,
            **kwargs,
        )

        self._camera_track_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )

        # Two channels in [-1, 1]; only the mean drives the 1D slide (differential has no yaw DOF).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    def _drive_commands_norm(self, wheel_action: np.ndarray) -> tuple[float, float, float]:
        """Left / right / mean normalized drive commands in [-1, 1] (what the env uses)."""
        wheel_action = np.clip(wheel_action, -1.0, 1.0)
        left = float(wheel_action[0])
        right = float(wheel_action[1])
        return left, right, (left + right) / 2.0

    def _maybe_log_motor_speeds(
        self,
        left: float,
        right: float,
        avg: float,
        *,
        theta_rad: float,
        theta_dot_rad_s: float,
        cart_vel_cmd_m_s: float,
    ) -> None:
        if not self.log_motor_speeds:
            return
        now = time.monotonic()
        interval = 1.0 / float(self.params.motor_log_hz)
        if now - self._last_motor_log_time < interval:
            return
        self._last_motor_log_time = now
        theta_deg = float(np.rad2deg(theta_rad))
        theta_dot_deg_s = float(np.rad2deg(theta_dot_rad_s))
        print(
            f"\r  θ={theta_rad:+.4f} rad ({theta_deg:+.2f}°)  "
            f"θ̇={theta_dot_rad_s:+.3f} rad/s ({theta_dot_deg_s:+.1f}°/s)  |  "
            f"drive  L {left:+.3f}  R {right:+.3f}  avg {avg:+.3f}  "
            f"→ ẋ_cmd={cart_vel_cmd_m_s:+.3f} m/s",
            end="",
            flush=True,
        )

    def _end_motor_log_line(self) -> None:
        if self.log_motor_speeds:
            print()

    def _wheel_to_sim_action(self, wheel_action: np.ndarray) -> np.ndarray:
        """Mean of left/right commands → target cart speed on slide joint (m/s)."""
        wheel_action = np.clip(wheel_action, -1.0, 1.0)
        u_fwd = float((wheel_action[0] + wheel_action[1]) / 2.0)
        v_cmd = u_fwd * float(self.params.max_cart_speed_m_s)
        return np.array([v_cmd], dtype=np.float64)

    def _compute_reward(
        self, obs: np.ndarray, wheel_action: np.ndarray, terminated: bool
    ) -> float:
        if terminated:
            return 0.0
        p = self.params
        x, theta, x_dot, theta_dot = obs
        alive = 1.0
        r_theta = -p.k_theta * theta**2
        r_vel = -p.k_vel * (x_dot**2 + theta_dot**2)
        r_act = -p.k_act * float(np.sum(wheel_action**2))
        delta = wheel_action - self._prev_action
        r_dact = -p.k_dact * float(np.sum(delta**2))
        return alive + r_theta + r_vel + r_act + r_dact

    def _apply_viewer_reset(self) -> None:
        """Reset physics from the GLFW viewer (Backspace); next step skips one control tick."""
        self._end_motor_log_line()
        mujoco.mj_resetData(self.model, self.data)
        self.reset_model()
        self._viewer_just_reset = True
        print("\n[BalanceBot] Simulation reset (Backspace)")
        if self.initial_hinge_rad != 0.0:
            print(f"  Re-applied initial hinge θ = {self.initial_hinge_rad:.5f} rad")

    def step(self, action):
        if self._viewer_just_reset:
            self._viewer_just_reset = False
            obs = self._get_obs()
            info = {
                "reward_survive": 1.0,
                "control_effort": 0.0,
                "action_jerk": 0.0,
                "saturation_fraction": 0.0,
                "drive_cmd_left": 0.0,
                "drive_cmd_right": 0.0,
                "drive_cmd_avg": 0.0,
                "cart_vel_cmd_m_s": 0.0,
                "viewer_reset": True,
            }
            if self.render_mode == "human":
                self.render()
            return obs, 0.0, False, False, info

        wheel_action = np.asarray(action, dtype=np.float64).reshape(2)
        left, right, avg = self._drive_commands_norm(wheel_action)
        cart_vel_cmd_m_s = avg * float(self.params.max_cart_speed_m_s)
        sim_action = self._wheel_to_sim_action(wheel_action)
        self.do_simulation(sim_action, self.frame_skip)

        obs = self._get_obs()
        theta = float(obs[1])
        theta_dot = float(obs[3])
        unhealthy = not np.isfinite(obs).all()
        fallen = self.terminate_on_fall and abs(theta) > self.params.max_tilt_rad
        terminated = bool(unhealthy or fallen)
        reward = self._compute_reward(obs, wheel_action, terminated)

        control_effort = float(np.linalg.norm(wheel_action))
        action_jerk = float(np.linalg.norm(wheel_action - self._prev_action))
        saturation = float(np.any(np.abs(wheel_action) > 0.9))
        info = {
            "reward_survive": float(not terminated),
            "control_effort": control_effort,
            "action_jerk": action_jerk,
            "saturation_fraction": saturation,
            "drive_cmd_left": left,
            "drive_cmd_right": right,
            "drive_cmd_avg": avg,
            "cart_vel_cmd_m_s": cart_vel_cmd_m_s,
        }

        self._prev_action = wheel_action.copy()
        self._maybe_log_motor_speeds(
            left,
            right,
            avg,
            theta_rad=theta,
            theta_dot_rad_s=theta_dot,
            cart_vel_cmd_m_s=cart_vel_cmd_m_s,
        )

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def _hook_viewer_reset_key(self, viewer) -> None:
        """Backspace resets the sim (works while paused; Gymnasium uses R for transparent)."""
        if self._viewer_key_hooked or viewer.window is None:
            return

        original = viewer._key_callback

        def key_callback(window, key, scancode, action, mods):
            if action == glfw.RELEASE and key == glfw.KEY_BACKSPACE:
                self._apply_viewer_reset()
                return
            original(window, key, scancode, action, mods)

        glfw.set_key_callback(viewer.window, key_callback)
        self._viewer_key_hooked = True

    def _configure_human_viewer(self) -> None:
        """Track the chassis in the GLFW viewer; optional low-GPU flags (once per viewer)."""
        if self.render_mode != "human":
            return
        viewer = self.mujoco_renderer._viewers.get("human")
        if viewer is None:
            return

        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(self._camera_track_body_id)

        self._hook_viewer_reset_key(viewer)

        if self._viewer_tuned:
            return

        if self.params.low_gpu_render:
            off = (
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_FOG,
                mujoco.mjtRndFlag.mjRND_HAZE,
            )
            for flag in off:
                viewer.scn.flags[flag] = 0
        if self.params.show_contact_viz:
            viewer._contacts = True
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self._viewer_tuned = True

    def render(self):
        result = super().render()
        self._configure_human_viewer()
        return result

    def reset_model(self):
        self._end_motor_log_line()
        p = self.params
        noise = p.reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=-noise, high=noise, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-noise, high=noise, size=self.model.nv
        )
        hinge_delta = float(qpos[1] - self.init_qpos[1])
        qpos[1] = float(self.initial_hinge_rad) + hinge_delta
        self.set_state(qpos, qvel)
        self._prev_action = np.zeros(2, dtype=np.float64)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def close(self):
        self._end_motor_log_line()
        super().close()


register(
    id="BalanceBot-v0",
    entry_point="env:BalanceBotEnv",
    max_episode_steps=1000,
)
