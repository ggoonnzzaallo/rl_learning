"""Physical and reward parameters for the balance bot MuJoCo model.

Edit values here to match your real robot. `render_xml()` writes `assets/balance_bot.xml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEMPLATE_PATH = ASSETS_DIR / "balance_bot.xml.template"
OUTPUT_PATH = ASSETS_DIR / "balance_bot.xml"


@dataclass(frozen=True)
class RobotParams:
    # --- Chassis (box half-extents, metres): x=length, y=lateral, z=height ---
    body_half_width: float = 0.0375  # full 75 mm (x, travel direction)
    body_half_depth: float = 0.076  # full 152 mm (y, lateral)
    body_half_height: float = 0.055  # full height 110 mm (z)
    body_mass: float = 0.544  # 544 g or 0.544kg
    # Mass centre height above wheel axle (m). Lower than box midpoint — battery at bottom.
    body_com_z: float = 0.032  # measured CoM above axle (~24 mm)
    body_ground_clearance_m: float = 0.0165  # min gap, body bottom to floor (16.5 mm)

    # --- Wheels (geometry / mass only; no motor model in sim) ---
    wheel_radius: float = 0.032  # 64 mm diameter → 32 mm radius
    wheel_half_width: float = 0.0125  # full width 25 mm (along wheel axle / y)
    wheel_mass: float = 0.024  # 24 g
    wheelbase: float = 0.185  # 160 mm inner-edge-to-inner-edge + 25 mm wheel width

    # --- Drive (sim): ideal velocity on slide joint — enough torque to track command ---
    max_cart_speed_m_s: float = 1.0  # cart speed when mean(τ_l, τ_r) = +1
    slide_velocity_kv: float = 200.0  # MuJoCo velocity servo gain (higher = stiffer tracking)

    # --- Simulation ---
    timestep: float = 0.002
    frame_skip: int = 5  # control dt = timestep * frame_skip = 0.01 s (100 Hz)
    slide_range: float = 2.0  # m, track half-length

    # --- Termination ---
    max_tilt_rad: float = 0.3

    # --- Reset ---
    reset_noise_scale: float = 0.01

    # --- Reward shaping (motor-friendly: k_dact >= k_act) ---
    k_theta: float = 0.5
    k_vel: float = 0.05
    k_act: float = 0.02
    k_dact: float = 0.1

    # --- Rendering (lighter on GPU — good for MacBook Air) ---
    low_gpu_render: bool = True
    show_contact_viz: bool = True  # red contact points + orange force arrows in human viewer
    render_width: int = 1280
    render_height: int = 720
    motor_log_hz: float = 10.0

    @property
    def control_dt(self) -> float:
        """Env / PID step duration (s): MuJoCo timestep × frame_skip."""
        return float(self.timestep * self.frame_skip)

    def axle_height_m(self) -> float:
        """Wheel-axle height above ground (m). Hinge origin on the cart is at this z."""
        return (
            self.body_ground_clearance_m
            + self.body_half_height
            - self.body_com_z
        )

    def chassis_geom_z(self) -> float:
        """Body geom / mass centre height above the wheel axle (hinge origin)."""
        return self.body_com_z

    def wheel_z(self) -> float:
        """Wheel centre z relative to axle so wheel bottoms touch the floor at reset."""
        return self.wheel_radius - self.axle_height_m()


DEFAULT_PARAMS = RobotParams()


def render_xml(params: RobotParams = DEFAULT_PARAMS, output_path: Path = OUTPUT_PATH) -> str:
    """Fill the MJCF template and write `assets/balance_bot.xml`. Returns the path."""
    template = TEMPLATE_PATH.read_text()
    half_wb = params.wheelbase / 2.0
    v_max = params.max_cart_speed_m_s
    xml = template.format(
        timestep=params.timestep,
        wheel_radius=params.wheel_radius,
        axle_height=params.axle_height_m(),
        slide_range=params.slide_range,
        body_half_width=params.body_half_width,
        body_half_height=params.body_half_height,
        body_half_depth=params.body_half_depth,
        body_mass=params.body_mass,
        chassis_geom_z=params.chassis_geom_z(),
        wheel_half_width=params.wheel_half_width,
        wheel_mass=params.wheel_mass,
        wheel_z=params.wheel_z(),
        left_wheel_y=-half_wb,
        right_wheel_y=half_wb,
        max_cart_speed=v_max,
        slide_velocity_kv=params.slide_velocity_kv,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml)
    return str(output_path)
