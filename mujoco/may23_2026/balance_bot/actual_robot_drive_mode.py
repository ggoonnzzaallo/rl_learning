#!/usr/bin/env python3
"""
Self-balancing robot with keyboard drive mode (SSH arrow keys).

Balance PID runs continuously; arrow keys add capped forward/back and turn commands
on top via differential motor mixing.

Run over SSH with a TTY:
    ssh -t pi@raspberrypi.local 'python3 actual_robot_drive_mode.py'

Controls:
    ↑ / W  forward     ↓ / S  backward
    ← / A  turn left   → / D  turn right
    Space  stop drive immediately
    q      quit cleanly
    Ctrl+C quit
"""
from __future__ import annotations

import select
import smbus2
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from math import atan2, degrees

from gpiozero import PWMOutputDevice

# ===== IMU Configuration =====
bus = smbus2.SMBus(1)
IMU_ADDR = 0x6A

# ===== Motor Configuration =====
in1 = PWMOutputDevice(23)  # pin 16 - Left motor forward
in2 = PWMOutputDevice(24)  # pin 18 - Left motor backward
in3 = PWMOutputDevice(19)  # pin 35 - Right motor forward
in4 = PWMOutputDevice(26)  # pin 37 - Right motor backward

# ===== Balance PID Parameters =====
Kp = 3.8
Ki = 0
Kd = 0.2
TARGET_ANGLE = 2.5

# ===== Control Loop Parameters =====
LOOP_TIME = 0.02  # 50 Hz
MAX_MOTOR_SPEED = 1.0
ANGLE_DEADBAND = 2.0

# ===== Complementary / noise filters =====
FILTER_ALPHA = 0.96
ENABLE_RATE_FILTER = True
RATE_FILTER_ALPHA = 0.8
ENABLE_ANGLE_FILTER = False
ANGLE_FILTER_ALPHA = 0.9
MANUAL_ROLL_OFFSET = 0.0

# ===== Drive mode limits (start conservative, tune up on hardware) =====
MAX_DRIVE = 0.35
MAX_TURN = 0.25
MAX_YAW_RATE_DEG_S = 60.0
DRIVE_SLEW = 2.0  # full scale per second
KEY_IDLE_TIMEOUT = 0.15  # seconds without key repeat → stop that axis

# Turn rate PID: maps yaw-rate error → normalized differential command
TURN_RATE_KP = 0.015

# Lean into acceleration (degrees of target offset at full forward command)
LEAN_GAIN = 3.0

# Reduce user drive when already tilted (balance axis degrees)
TILT_DERATE_START = 12.0
TILT_DERATE_END = 30.0

# Further reduce user drive when balance PID is working hard
BALANCE_HEADROOM_START = 0.7

# ===== Initialize IMU =====
print("Initializing IMU...")
bus.write_byte_data(IMU_ADDR, 0x10, 0b01101100)
bus.write_byte_data(IMU_ADDR, 0x11, 0b01101100)
time.sleep(0.1)
print("IMU initialized")


def read_word(reg: int) -> int:
    hi = bus.read_byte_data(IMU_ADDR, reg + 1)
    lo = bus.read_byte_data(IMU_ADDR, reg)
    val = (hi << 8) | lo
    return val if val < 32768 else val - 65536


def normalize_angle(angle: float) -> float:
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def read_imu() -> dict:
    ax = read_word(0x28) * 0.000244
    ay = read_word(0x2A) * 0.000244
    az = read_word(0x2C) * 0.000244

    gx = read_word(0x22) * 0.070
    gy = read_word(0x24) * 0.070
    gz = read_word(0x26) * 0.070

    roll = normalize_angle(degrees(atan2(ay, az)))
    pitch = normalize_angle(degrees(atan2(-ax, (ay**2 + az**2) ** 0.5)))

    return {
        "roll": roll,
        "pitch": pitch,
        "roll_rate": gx,
        "pitch_rate": gy,
        "yaw_rate": gz,
        "accel": (ax, ay, az),
        "gyro": (gx, gy, gz),
    }


def set_motor_speed(left_speed: float, right_speed: float) -> None:
    left_speed = clamp(left_speed, -1.0, 1.0)
    right_speed = clamp(right_speed, -1.0, 1.0)

    if left_speed > 0:
        in1.value = abs(left_speed)
        in2.value = 0
    elif left_speed < 0:
        in1.value = 0
        in2.value = abs(left_speed)
    else:
        in1.value = 0
        in2.value = 0

    if right_speed > 0:
        in3.value = abs(right_speed)
        in4.value = 0
    elif right_speed < 0:
        in3.value = 0
        in4.value = abs(right_speed)
    else:
        in3.value = 0
        in4.value = 0


def stop() -> None:
    set_motor_speed(0, 0)


def slew_toward(current: float, target: float, max_delta: float) -> float:
    delta = clamp(target - current, -max_delta, max_delta)
    return current + delta


def tilt_derate_scale(tilt_deg: float) -> float:
    abs_tilt = abs(tilt_deg)
    if abs_tilt <= TILT_DERATE_START:
        return 1.0
    if abs_tilt >= TILT_DERATE_END:
        return 0.0
    return 1.0 - (abs_tilt - TILT_DERATE_START) / (TILT_DERATE_END - TILT_DERATE_START)


def balance_headroom_scale(u_bal: float) -> float:
    abs_bal = abs(u_bal)
    if abs_bal <= BALANCE_HEADROOM_START:
        return 1.0
    return clamp(1.0 - (abs_bal - BALANCE_HEADROOM_START) / 0.3, 0.3, 1.0)


def mix_motor_commands(
    u_bal: float, u_fwd: float, u_turn: float, max_speed: float = MAX_MOTOR_SPEED
) -> tuple[float, float]:
    left = clamp(u_bal + u_fwd - u_turn, -max_speed, max_speed)
    right = clamp(u_bal + u_fwd + u_turn, -max_speed, max_speed)
    return left, right


def motor_direction(speed: float) -> str:
    if speed > 0.001:
        return "F"
    if speed < -0.001:
        return "B"
    return "-"


@dataclass
class DriveCommand:
    """Thread-safe drive targets updated by the keyboard reader."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    fwd_target: float = 0.0
    turn_sign: float = 0.0  # +1 left, -1 right
    fwd: float = 0.0
    turn_sign_slewed: float = 0.0
    last_fwd_time: float = 0.0
    last_turn_time: float = 0.0
    quit_requested: bool = False
    enabled: bool = False

    def set_forward(self, direction: float) -> None:
        now = time.monotonic()
        with self.lock:
            self.fwd_target = direction * MAX_DRIVE
            self.last_fwd_time = now

    def set_turn(self, direction: float) -> None:
        """direction: +1 = left, -1 = right."""
        now = time.monotonic()
        with self.lock:
            self.turn_sign = direction
            self.last_turn_time = now

    def stop_immediate(self) -> None:
        with self.lock:
            self.fwd_target = 0.0
            self.turn_sign = 0.0
            self.fwd = 0.0
            self.turn_sign_slewed = 0.0
            self.last_fwd_time = 0.0
            self.last_turn_time = 0.0

    def request_quit(self) -> None:
        with self.lock:
            self.quit_requested = True

    def tick(self, dt: float) -> None:
        now = time.monotonic()
        with self.lock:
            if self.enabled and now - self.last_fwd_time > KEY_IDLE_TIMEOUT:
                self.fwd_target = 0.0
            if self.enabled and now - self.last_turn_time > KEY_IDLE_TIMEOUT:
                self.turn_sign = 0.0

            max_delta = DRIVE_SLEW * dt
            self.fwd = slew_toward(self.fwd, self.fwd_target, max_delta)
            self.turn_sign_slewed = slew_toward(
                self.turn_sign_slewed, self.turn_sign, max_delta * 2.0
            )

    def snapshot(self) -> tuple[float, float]:
        with self.lock:
            return self.fwd, self.turn_sign_slewed

    def should_quit(self) -> bool:
        with self.lock:
            return self.quit_requested


class StdinKeyboardReader:
    """Read arrow keys from SSH stdin (termios + select, no extra dependencies)."""

    def __init__(self, drive: DriveCommand) -> None:
        self.drive = drive
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_term: list | None = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            print("Warning: stdin is not a TTY — drive commands disabled (balance-only).")
            return False

        self._old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.drive.enabled = True
        self._running = True
        self._thread = threading.Thread(target=self._run, name="keyboard", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        self.drive.enabled = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
            self._old_term = None

    def _read_bytes(self, count: int, timeout: float = 0.02) -> str:
        out = []
        deadline = time.monotonic() + timeout
        while len(out) < count and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if not ready:
                break
            out.append(sys.stdin.read(1))
        return "".join(out)

    def _handle_key(self, key: str) -> None:
        if key in ("\x1b[A", "w", "W"):
            self.drive.set_forward(+1.0)
        elif key in ("\x1b[B", "s", "S"):
            self.drive.set_forward(-1.0)
        elif key in ("\x1b[D", "a", "A"):
            self.drive.set_turn(+1.0)
        elif key in ("\x1b[C", "d", "D"):
            self.drive.set_turn(-1.0)
        elif key == " ":
            self.drive.stop_immediate()
        elif key in ("q", "Q", "\x03"):
            self.drive.request_quit()

    def _run(self) -> None:
        while self._running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue

            ch = sys.stdin.read(1)
            if not ch:
                continue

            if ch == "\x1b":
                rest = self._read_bytes(2, timeout=0.02)
                self._handle_key("\x1b" + rest if rest else ch)
            else:
                self._handle_key(ch)


class ComplementaryFilter:
    def __init__(self, alpha: float = 0.96, initial_angle: float = 0.0) -> None:
        self.alpha = alpha
        self.filtered_angle = initial_angle
        self.last_time = time.time()

    def update(self, accel_angle: float, gyro_rate: float, dt: float | None = None) -> float:
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = LOOP_TIME

        gyro_angle = self.filtered_angle + gyro_rate * dt
        accel_normalized = normalize_angle(accel_angle)
        gyro_normalized = normalize_angle(gyro_angle)
        angle_diff = normalize_angle(accel_normalized - gyro_normalized)
        self.filtered_angle = normalize_angle(
            gyro_normalized + (1 - self.alpha) * angle_diff
        )
        self.last_time = current_time
        return self.filtered_angle

    def reset(self, angle: float = 0.0) -> None:
        self.filtered_angle = angle
        self.last_time = time.time()


class LowPassFilter:
    def __init__(self, alpha: float = 0.8, initial_value: float = 0.0) -> None:
        self.alpha = alpha
        self.filtered_value = initial_value

    def update(self, new_value: float) -> float:
        self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

    def reset(self, value: float = 0.0) -> None:
        self.filtered_value = value


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, target: float = 0.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, current_value: float, current_rate: float) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = LOOP_TIME

        error = self.target - current_value
        self.integral = clamp(self.integral + error * dt, -10.0, 10.0)
        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * (-current_rate)
        )
        self.last_time = current_time
        return output


class TurnRateController:
    """Map target yaw rate (deg/s) to differential drive command."""

    def __init__(self, kp: float, max_turn: float) -> None:
        self.kp = kp
        self.max_turn = max_turn

    def compute(self, omega_target_deg_s: float, yaw_rate_deg_s: float) -> float:
        error = omega_target_deg_s - yaw_rate_deg_s
        u_turn = self.kp * error
        return clamp(u_turn, -self.max_turn, self.max_turn)


def filter_imu_pipeline(
    imu_data: dict,
    roll_offset: float,
    angle_filter: ComplementaryFilter,
    rate_filter: LowPassFilter | None,
    angle_lp_filter: LowPassFilter | None,
    dt: float,
) -> tuple[float, float, float]:
    accel_roll_corrected = normalize_angle(imu_data["roll"] - roll_offset)
    roll_rate_raw = imu_data["roll_rate"]

    if rate_filter is not None:
        roll_rate = rate_filter.update(roll_rate_raw)
    else:
        roll_rate = roll_rate_raw

    roll_filtered = angle_filter.update(accel_roll_corrected, roll_rate, dt=dt)

    if angle_lp_filter is not None:
        roll_corrected = angle_lp_filter.update(roll_filtered)
    else:
        roll_corrected = roll_filtered

    return roll_corrected, roll_rate, roll_rate_raw


def main() -> None:
    print("\n" + "=" * 80)
    print("Self-Balancing Robot - Drive Mode")
    print("=" * 80)
    print(f"Balance PID: Kp={Kp}, Ki={Ki}, Kd={Kd}, target={TARGET_ANGLE}°")
    print(
        f"Drive limits: MAX_DRIVE={MAX_DRIVE}, MAX_TURN={MAX_TURN}, "
        f"MAX_YAW_RATE={MAX_YAW_RATE_DEG_S}°/s"
    )
    print(f"Control loop: {1 / LOOP_TIME:.1f} Hz")
    print("\nControls (SSH terminal must have focus):")
    print("  ↑/W forward   ↓/S backward   ←/A turn left   →/D turn right")
    print("  Space = stop drive   q = quit   Ctrl+C = quit")
    print("=" * 80 + "\n")

    roll_offset = -180.0 + MANUAL_ROLL_OFFSET

    pid = PIDController(Kp, Ki, Kd, TARGET_ANGLE)
    turn_pid = TurnRateController(TURN_RATE_KP, MAX_TURN)
    angle_filter = ComplementaryFilter(alpha=FILTER_ALPHA, initial_angle=0.0)
    rate_filter = (
        LowPassFilter(alpha=RATE_FILTER_ALPHA, initial_value=0.0)
        if ENABLE_RATE_FILTER
        else None
    )
    angle_lp_filter = (
        LowPassFilter(alpha=ANGLE_FILTER_ALPHA, initial_value=0.0)
        if ENABLE_ANGLE_FILTER
        else None
    )

    drive = DriveCommand()
    keyboard = StdinKeyboardReader(drive)
    keyboard_ok = keyboard.start()

    stop()
    print("Starting balance + drive control...")
    time.sleep(0.5)

    try:
        loop_count = 0
        start_time = time.time()

        while not drive.should_quit():
            loop_start = time.time()

            imu_data = read_imu()
            roll_corrected, roll_rate, roll_rate_raw = filter_imu_pipeline(
                imu_data,
                roll_offset,
                angle_filter,
                rate_filter,
                angle_lp_filter,
                LOOP_TIME,
            )

            if abs(roll_corrected) > 45.0:
                print(f"\nRobot fallen! Angle: {roll_corrected:.1f}°")
                stop()
                drive.stop_immediate()
                print("Waiting for robot to be upright again...")
                while abs(roll_corrected) > 20.0 and not drive.should_quit():
                    imu_data = read_imu()
                    roll_corrected, roll_rate, _ = filter_imu_pipeline(
                        imu_data,
                        roll_offset,
                        angle_filter,
                        rate_filter,
                        angle_lp_filter,
                        0.1,
                    )
                    time.sleep(0.1)
                if drive.should_quit():
                    break
                print("Resuming balance control...")
                pid.integral = 0.0
                angle_filter.reset(0.0)
                if rate_filter:
                    rate_filter.reset(0.0)
                if angle_lp_filter:
                    angle_lp_filter.reset(0.0)
                time.sleep(0.5)
                continue

            drive.tick(LOOP_TIME)
            u_fwd, turn_sign = drive.snapshot()

            derate = tilt_derate_scale(roll_corrected)
            u_fwd *= derate

            omega_target = turn_sign * MAX_YAW_RATE_DEG_S * derate
            u_turn = turn_pid.compute(omega_target, imu_data["yaw_rate"]) if keyboard_ok else 0.0

            pid.target = TARGET_ANGLE + LEAN_GAIN * u_fwd
            pid_output = pid.compute(roll_corrected, roll_rate)
            u_bal = pid_output / 100.0

            if abs(roll_corrected) < ANGLE_DEADBAND and abs(roll_rate) < 5.0:
                u_bal = 0.0

            u_bal = clamp(u_bal, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED)

            headroom = balance_headroom_scale(u_bal)
            u_fwd *= headroom
            u_turn *= headroom

            left_motor_speed, right_motor_speed = mix_motor_commands(
                u_bal, u_fwd, u_turn, MAX_MOTOR_SPEED
            )
            set_motor_speed(left_motor_speed, right_motor_speed)

            if loop_count % 25 == 0:
                elapsed = time.time() - start_time
                rate_display = f"{roll_rate:7.1f}°/s"
                if ENABLE_RATE_FILTER:
                    rate_display += f" (raw: {roll_rate_raw:6.1f})"
                drive_label = "ON" if keyboard_ok else "OFF"
                print(
                    f"Time: {elapsed:6.2f}s | "
                    f"Roll: {roll_corrected:7.2f}° | "
                    f"Rate: {rate_display} | "
                    f"Drive: {drive_label} fwd {u_fwd:+.3f} turn {u_turn:+.3f} | "
                    f"Bal: {u_bal:+.3f} | "
                    f"L {left_motor_speed:+.3f} ({motor_direction(left_motor_speed)}) | "
                    f"R {right_motor_speed:+.3f} ({motor_direction(right_motor_speed)})"
                )

            loop_count += 1

            elapsed = time.time() - loop_start
            sleep_time = LOOP_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_count % 100 == 0:
                print(
                    f"\nLoop time exceeded! ({elapsed * 1000:.1f}ms > "
                    f"{LOOP_TIME * 1000:.1f}ms)"
                )

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        keyboard.stop()
        stop()
        print("Motors stopped. Robot safe.")


if __name__ == "__main__":
    main()
