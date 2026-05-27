# Deploy notes (future — not implemented)

Checklist for when you move from sim to the real robot. **Nothing here runs yet.**

## Hardware inventory

- **IMU:** Adafruit LSM6DS3TR-C (6-DoF accel + gyro, I2C 0x6A/0x6B)
- **Compute:** Raspberry Pi 5
- **Motors:** two wheel motors (driver TBD)
- **Encoders:** TBD — needed for sim `x`, `ẋ` on real robot

## Obs / action contract

| Sim (train now) | Real robot (later) |
|-----------------|-------------------|
| θ, θ̇ | LSM6DS3TR-C: accel → θ, gyro → θ̇ |
| x, ẋ | Wheel encoders (not from IMU alone) |
| [τ_l, τ_r] ∈ [-1, 1] | Scale to motor driver units |

Balance-only deploy (`[θ, θ̇]` obs) possible if encoders aren't ready — requires
retraining a separate policy.

## Pre-deploy checklist

- [ ] Sim validated with PID (`pid_baseline.py`) matching real behaviour
- [ ] SAC policy trained and tested in sim (`play.py`)
- [ ] Match control `dt` to sim (`robot_params.control_dt`)
- [ ] IMU mounting: pitch axis, `theta_offset_rad`, sign convention
- [ ] IMU calibration on Pi (upright zero, tilt sign)
- [ ] Motor driver wired; map normalized actions to physical units
- [ ] **Current/torque limits** in driver (hard cap)
- [ ] **Slew rate limit** on Pi (mirrors `k_dact` in reward)
- [ ] **Thermal cutoff** — stop if over-temp or sustained high effort
- [ ] Export policy (SB3 → lightweight numpy `.npz`)
- [ ] Pi control loop at 50–100 Hz
- [ ] System ID: tune `max_cart_speed_m_s`, `slide_velocity_kv`, friction

## Pi dependencies (when implementing)

```
numpy
adafruit-blinka
adafruit-circuitpython-lsm6ds
```

[Adafruit LSM6DS3TR-C guide](https://learn.adafruit.com/adafruit-lsm6ds3tr-c-6-dof-accel-gyro-imu/python-circuitpython)

## Sim realism before serious sim-to-real

See README **Future work**: friction/contact, motor dynamics, domain randomization,
sensor noise injection.
