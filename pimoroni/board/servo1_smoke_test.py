"""
First bring-up for a single angular servo on Servo 2040 port #1.

From repo root:
  uv run mpremote run pimoroni/board/servo1_smoke_test.py

Or copy to the board first:
  uv run mpremote cp pimoroni/board/servo1_smoke_test.py :servo1_smoke_test.py
"""

import time
from servo import Servo, servo2040

PAUSE_S = 1.0

servo = Servo(servo2040.SERVO_1)

print("Servo 1 smoke test — enable and sweep min / mid / max")
servo.enable()
time.sleep(PAUSE_S)

for label, move in (
    ("min", servo.to_min),
    ("mid", servo.to_mid),
    ("max", servo.to_max),
    ("mid (home)", servo.to_mid),
):
    print(label)
    move()
    time.sleep(PAUSE_S)

print("Done — disabling servo PWM")
servo.disable()
