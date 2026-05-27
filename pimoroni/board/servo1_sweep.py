"""
Gentle sine sweep on servo port #1 (after smoke test passes).

Adjust SWEEP_DEG if the horn hits mechanical stops.
"""

import math
import time
from servo import Servo, servo2040

SWEEP_DEG = 45.0
CYCLES = 2
STEP_DELAY_S = 0.02

servo = Servo(servo2040.SERVO_1)
servo.enable()
time.sleep(0.5)

print(f"Sweeping ±{SWEEP_DEG}° for {CYCLES} cycle(s)")
for _ in range(CYCLES):
    for deg in range(360):
        servo.value(math.sin(math.radians(deg)) * SWEEP_DEG)
        time.sleep(STEP_DELAY_S)

servo.to_mid()
time.sleep(0.5)
servo.disable()
print("Finished")
