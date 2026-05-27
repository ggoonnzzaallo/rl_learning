# Pimoroni Servo 2040

Exploration notes and starter scripts for the [Servo 2040](https://shop.pimoroni.com/products/servo-2040) — an RP2040 board that drives up to **18 hobby servos** with Pimoroni’s `servo` MicroPython library.

## Python on this board (important)

Yes, it runs **Python**, but not the same Python as this repo’s RL code:

| Where code runs | What you use |
|-----------------|--------------|
| **On the Servo 2040** | **MicroPython** (Pimoroni build with `servo`, `pimoroni`, etc. baked in) |
| **On your Mac** | **Cursor** to edit scripts; **`mpremote`** (terminal) to upload, run, and REPL |

Scripts in `pimoroni/board/` are edited in Cursor and executed on the board via `mpremote run` (or copied and run as `main.py` on boot).

## Hardware you have wired

- **Servo port #1** → library constant `servo2040.SERVO_1` (GPIO 0)
- **Power**: Small servos may work from USB; larger loads need a separate 5–6 V supply on the board’s servo power input (see Pimoroni docs). Always common GND between supply and board.

## Board capabilities (what to explore next)

| Feature | Notes |
|---------|--------|
| **18 × servo headers** | `Servo` (PWM, up to 16 at once) or `ServoCluster` (PIO, many servos smoothly) |
| **6 × RGB LEDs** | Addressable (WS2812-style) on the edge |
| **6 × analog sensor headers** | Multiplexed ADC inputs |
| **Voltage / current sense** | Monitor servo rail health |
| **User button** | Same as BOOTSEL; `Button(servo2040.USER_SW)` |
| **Qwiic / Stemma QT I2C** | Extra sensors |
| **USB** | Power, REPL, file transfer |

Official example gallery: [pimoroni-pico/micropython/examples/servo2040](https://github.com/pimoroni/pimoroni-pico/tree/main/micropython/examples/servo2040)

API reference: [servo module README](https://github.com/pimoroni/pimoroni-pico/blob/main/micropython/modules/servo/README.md)

## One-time setup

### 1. Flash Pimoroni MicroPython

1. Unplug USB, hold **BOOTSEL**, plug in USB → board appears as **RPI-RP2**.
2. Download **`pico-v*-pimoroni-micropython.uf2`** from [pimoroni-pico releases](https://github.com/pimoroni/pimoroni-pico/releases/latest) (Servo 2040 uses the generic Pico image).
3. Drag the `.uf2` onto **RPI-RP2**. The board reboots; the drive disappears (normal).

Guide: [setting-up-micropython.md](https://github.com/pimoroni/pimoroni-pico/blob/main/setting-up-micropython.md)

### 2. Tooling on your Mac (Cursor workflow)

From the repo root:

```bash
uv sync --extra pico
```

This installs [`mpremote`](https://docs.micropython.org/en/latest/reference/mpremote.html), which talks to the board over USB. Use it from Cursor’s integrated terminal.

Plug in the board, then:

```bash
uv run mpremote connect list
```

Quick firmware check (interactive REPL — `Ctrl+]` or `Ctrl+X` to exit):

```bash
uv run mpremote repl
```

```python
>>> from servo import Servo, servo2040
>>> print(servo2040.NUM_SERVOS)
18
```

If `import servo` fails, re-flash the Pimoroni `.uf2`.

### 3. Run your first script

Edit `pimoroni/board/servo1_smoke_test.py` in **Cursor**, then from the repo root:

```bash
uv run mpremote run pimoroni/board/servo1_smoke_test.py
```

The servo on port **#1** should move: center → min → max → center, then stop.

**Common commands** (all via `uv run mpremote …`):

| Goal | Command |
|------|---------|
| Run a script once | `mpremote run pimoroni/board/servo1_smoke_test.py` |
| Copy to board filesystem | `mpremote cp pimoroni/board/servo1_smoke_test.py :servo1_smoke_test.py` |
| Run a file already on the board | `mpremote exec "import servo1_smoke_test"` or reset after saving as `:main.py` |
| Interactive REPL | `mpremote repl` |
| List files on board | `mpremote fs ls :` |

If you have multiple serial devices, pin the port:

```bash
uv run mpremote connect /dev/cu.usbmodem101 run pimoroni/board/servo1_smoke_test.py
```

## Servo API cheat sheet (port #1)

```python
from servo import Servo, servo2040

s = Servo(servo2040.SERVO_1)
s.enable()           # power on; first time → middle position
s.to_min()           # calibrated minimum (often ~0°)
s.to_mid()
s.to_max()
s.value(45)          # set angle in degrees (angular servo)
s.to_percent(0.5)    # 0.0 = min, 1.0 = max (library mapping)
s.pulse(1500)        # pulse width in µs (low-level)
s.disable()          # stop PWM (less holding torque / heat)
```

## Finding the USB port (macOS)

```bash
ls /dev/cu.usb*
```

Use the `cu.*` device with `mpremote connect /dev/cu.usbmodem…`. If nothing appears, check cable (data, not charge-only) and that firmware is flashed.

## Suggested exploration path

1. **`board/servo1_smoke_test.py`** — confirm wiring and power
2. Pimoroni’s **`single_servo.py`** — sine sweep and stepped motion ([source](https://github.com/pimoroni/pimoroni-pico/blob/main/micropython/examples/servo2040/single_servo.py))
3. **`led_rainbow.py`** — edge RGB LEDs
4. **`read_sensors.py`** / **`current_meter.py`** — ADC and power monitoring
5. **`servo_cluster.py`** — when you add more servos
6. **`calibration.py`** — if your servo doesn’t hit mechanical limits cleanly

## Talking to the board from your Mac (later)

Same idea as `serialcom/dec1_2024/`: run a small MicroPython loop on the board that reads UART commands, and a Python script on the laptop sends angles. Useful if you later want RL policies on the Mac and actuators on the Servo 2040. Not required for first bring-up.

## Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| No USB device | Data USB cable; try another port; re-flash with BOOTSEL |
| `ImportError: servo` | Install Pimoroni firmware, not generic MicroPython only |
| Servo twitches / no move | External servo power; signal on **#1**; brown = GND, red = +5 V, orange = signal |
| Jitter at rest | Normal under load; `disable()` when idle; better power supply |
| Only moves a few degrees | Wrong servo type in calibration; try Pimoroni `calibration.py` example |

## Links

- [Product page](https://shop.pimoroni.com/products/servo-2040)
- [MicroPython setup](https://github.com/pimoroni/pimoroni-pico/blob/main/setting-up-micropython.md)
- [Servo 2040 examples](https://github.com/pimoroni/pimoroni-pico/tree/main/micropython/examples/servo2040)
- [Servo library docs](https://github.com/pimoroni/pimoroni-pico/blob/main/micropython/modules/servo/README.md)
