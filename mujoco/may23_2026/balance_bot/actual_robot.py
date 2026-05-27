#!/usr/bin/env python3
"""
Self-balancing robot using PID control
Combines IMU readings with motor control to maintain balance
"""
import smbus2
import time
from math import atan2, degrees
from gpiozero import PWMOutputDevice

# ===== IMU Configuration =====
bus = smbus2.SMBus(1)
IMU_ADDR = 0x6A

# ===== Motor Configuration =====
# Motor control pins using PWM (BCM numbers)
in1 = PWMOutputDevice(23)  # pin 16 - Left motor forward
in2 = PWMOutputDevice(24)  # pin 18 - Left motor backward
in3 = PWMOutputDevice(19)  # pin 35 - Right motor forward
in4 = PWMOutputDevice(26)  # pin 37 - Right motor backward

# ===== PID Parameters =====
# These values need to be tuned for your specific robot
Kp = 3.8  # Proportional gain (adjust based on response)
Ki = 0    # Integral gain (helps eliminate steady-state error)
Kd = 0.2   # Derivative gain (damping, reduces oscillations)

# Target angle (0° = upright)
TARGET_ANGLE = 2.5

# ===== Control Loop Parameters =====
LOOP_TIME = 0.02  # 50 Hz control loop (20ms)
MAX_MOTOR_SPEED = 1  # Limit motor speed to prevent overshoot
ANGLE_DEADBAND = 2.0   # Stop motors if angle is within this range (degrees)

# ===== Complementary Filter Parameters =====
# Alpha determines how much to trust gyro (0.0-1.0)
# Higher alpha = more trust in gyro (better for fast movements, but drifts)
# Lower alpha = more trust in accel (better for stability, but noisier)
# Typical range: 0.95-0.98 for balance robots
FILTER_ALPHA = 0.96  # 96% gyro, 4% accel

# ===== Additional Noise Filtering Parameters =====
# These filters help reduce high-frequency noise that causes twitching with high Kd
# Enable/disable and tune these based on your needs

# Low-pass filter for gyroscope rate (CRITICAL for reducing derivative noise)
# This filters the rate signal before it goes into the PID derivative term
# Higher value (closer to 1.0) = less filtering, faster response, more noise
# Lower value (closer to 0.0) = more filtering, slower response, less noise
# Recommended: 0.7-0.9 for balance robots (0.8 is a good starting point)
ENABLE_RATE_FILTER = True
RATE_FILTER_ALPHA = 0.8  # Exponential moving average coefficient for rate

# Low-pass filter for final angle (optional additional smoothing)
# This adds a second stage of filtering on the complementary filter output
# Use this if you still see noise after filtering the rate
# Higher value = less filtering, faster response
# Lower value = more filtering, slower response (may add lag)
# Recommended: 0.85-0.95 (0.9 is a good starting point, or disable if not needed)
ENABLE_ANGLE_FILTER = False  # Set to True to enable
ANGLE_FILTER_ALPHA = 0.9  # Exponential moving average coefficient for angle

# ===== IMU Roll Offset =====
# IMU roll -180° is treated as 0° (robot lying flat)
# Add manual offset here to fine-tune if needed
# Positive values rotate the reference clockwise, negative counter-clockwise
MANUAL_ROLL_OFFSET = 0.0  # Adjust this to fine-tune the zero reference (degrees)

# ===== Initialize IMU =====
print("Initializing IMU...")
bus.write_byte_data(IMU_ADDR, 0x10, 0b01101100)  # Accel ±8 g, 416 Hz
bus.write_byte_data(IMU_ADDR, 0x11, 0b01101100)  # Gyro ±2000 dps, 416 Hz
time.sleep(0.1)
print("IMU initialized")

def read_word(reg):
    """Read 16-bit signed value from IMU register"""
    hi = bus.read_byte_data(IMU_ADDR, reg + 1)
    lo = bus.read_byte_data(IMU_ADDR, reg)
    val = (hi << 8) | lo
    return val if val < 32768 else val - 65536

def normalize_angle(angle):
    """
    Normalize angle to [-180, 180] degrees to handle wrapping
    Example: 350° -> -10°, -190° -> 170°
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def read_imu():
    """Read accelerometer and gyroscope data, return all angles and rates"""
    # Accelerometer readings (g)
    ax = read_word(0x28) * 0.000244  # ±8 g → 0.244 mg/LSB
    ay = read_word(0x2A) * 0.000244
    az = read_word(0x2C) * 0.000244
    
    # Gyroscope readings (°/s)
    gx = read_word(0x22) * 0.070  # ±2000 dps → 70 mdps/LSB
    gy = read_word(0x24) * 0.070
    gz = read_word(0x26) * 0.070
    
    # Calculate all angles
    roll = normalize_angle(degrees(atan2(ay, az)))      # Left/right tilt
    pitch = normalize_angle(degrees(atan2(-ax, (ay**2 + az**2)**0.5)))  # Forward/backward tilt
    # Note: Yaw requires magnetometer, not available from accelerometer alone
    
    return {
        'roll': roll,
        'pitch': pitch,
        'roll_rate': gx,    # Rotation rate around X axis
        'pitch_rate': gy,   # Rotation rate around Y axis
        'yaw_rate': gz,     # Rotation rate around Z axis
        'accel': (ax, ay, az),
        'gyro': (gx, gy, gz)
    }

def set_motor_speed(left_speed, right_speed):
    """
    Set motor speeds (-1.0 to 1.0)
    Positive = forward, Negative = backward, 0 = stop
    """
    # Clamp speeds to valid range
    left_speed = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))
    
    # Left motor
    if left_speed > 0:
        in1.value = abs(left_speed)
        in2.value = 0
    elif left_speed < 0:
        in1.value = 0
        in2.value = abs(left_speed)
    else:
        in1.value = 0
        in2.value = 0
    
    # Right motor
    if right_speed > 0:
        in3.value = abs(right_speed)
        in4.value = 0
    elif right_speed < 0:
        in3.value = 0
        in4.value = abs(right_speed)
    else:
        in3.value = 0
        in4.value = 0

def stop():
    """Stop both motors"""
    set_motor_speed(0, 0)

class ComplementaryFilter:
    """
    Complementary filter to combine accelerometer and gyroscope data.
    Reduces noise from accelerometer while preventing gyroscope drift.
    """
    def __init__(self, alpha=0.96, initial_angle=0.0):
        """
        Args:
            alpha: Filter coefficient (0.0-1.0)
                   Higher = more trust in gyro (better for fast movements)
                   Lower = more trust in accel (better for stability)
            initial_angle: Starting angle estimate (degrees)
        """
        self.alpha = alpha
        self.filtered_angle = initial_angle
        self.last_time = time.time()
    
    def update(self, accel_angle, gyro_rate, dt=None):
        """
        Update filtered angle estimate.
        
        Args:
            accel_angle: Angle from accelerometer (degrees)
            gyro_rate: Angular rate from gyroscope (degrees/second)
            dt: Time step (seconds). If None, uses actual elapsed time.
        
        Returns:
            Filtered angle estimate (degrees)
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = LOOP_TIME  # Fallback to expected loop time
        
        # Integrate gyroscope rate to get angle change
        gyro_angle = self.filtered_angle + gyro_rate * dt
        
        # Complementary filter: blend gyro and accel
        # Normalize accel angle to handle wrapping
        accel_normalized = normalize_angle(accel_angle)
        gyro_normalized = normalize_angle(gyro_angle)
        
        # Calculate difference for proper blending across 180/-180 boundary
        angle_diff = normalize_angle(accel_normalized - gyro_normalized)
        
        # Blend: alpha * gyro + (1-alpha) * accel
        self.filtered_angle = normalize_angle(
            gyro_normalized + (1 - self.alpha) * angle_diff
        )
        
        self.last_time = current_time
        return self.filtered_angle
    
    def reset(self, angle=0.0):
        """Reset filter to a specific angle"""
        self.filtered_angle = angle
        self.last_time = time.time()

class LowPassFilter:
    """
    First-order low-pass filter (exponential moving average).
    Reduces high-frequency noise while maintaining fast response.
    
    Formula: filtered = alpha * new_value + (1 - alpha) * filtered_previous
    Higher alpha = less filtering (faster response, more noise)
    Lower alpha = more filtering (slower response, less noise)
    """
    def __init__(self, alpha=0.8, initial_value=0.0):
        """
        Args:
            alpha: Filter coefficient (0.0-1.0)
                   Higher = less filtering, faster response
                   Lower = more filtering, slower response
            initial_value: Starting filtered value
        """
        self.alpha = alpha
        self.filtered_value = initial_value
    
    def update(self, new_value):
        """
        Update filtered value with new measurement.
        
        Args:
            new_value: New raw measurement
        
        Returns:
            Filtered value
        """
        self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    
    def reset(self, value=0.0):
        """Reset filter to a specific value"""
        self.filtered_value = value
    
    def get_value(self):
        """Get current filtered value without updating"""
        return self.filtered_value

class PIDController:
    """PID controller for self-balancing"""
    def __init__(self, kp, ki, kd, target=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
    
    def compute(self, current_value, current_rate):
        """Compute PID output"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = LOOP_TIME  # Fallback to expected loop time
        
        # Calculate error
        error = self.target - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        # Limit integral to prevent windup
        self.integral = max(-10.0, min(10.0, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term (using rate of change)
        # Negative because we want to oppose the motion
        d_term = self.kd * (-current_rate)
        
        # Total output
        output = p_term + i_term + d_term
        
        self.last_error = error
        self.last_time = current_time
        
        return output

# ===== Main Control Loop =====
def main():
    print("\n" + "="*80)
    print("Self-Balancing Robot - PID Control")
    print("="*80)
    print(f"PID Parameters: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"Target Angle: {TARGET_ANGLE}°")
    print(f"Control Loop: {1/LOOP_TIME:.1f} Hz ({LOOP_TIME*1000:.1f} ms)")
    print(f"Complementary Filter: alpha={FILTER_ALPHA:.3f} ({FILTER_ALPHA*100:.1f}% gyro, {(1-FILTER_ALPHA)*100:.1f}% accel)")
    
    # Calculate fixed roll offset: IMU roll -180° = 0° (robot flat)
    # Add manual offset for fine-tuning
    roll_offset = -180.0 + MANUAL_ROLL_OFFSET
    print(f"Roll Offset: {roll_offset:.2f}° (IMU -180° + manual offset {MANUAL_ROLL_OFFSET:.2f}°)")
    
    # Print noise filtering status
    print(f"\nNoise Filtering:")
    if ENABLE_RATE_FILTER:
        print(f"  Rate Filter: ENABLED (alpha={RATE_FILTER_ALPHA:.2f})")
    else:
        print(f"  Rate Filter: DISABLED")
    if ENABLE_ANGLE_FILTER:
        print(f"  Angle Filter: ENABLED (alpha={ANGLE_FILTER_ALPHA:.2f})")
    else:
        print(f"  Angle Filter: DISABLED")
    
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    
    # Initialize PID controller
    pid = PIDController(Kp, Ki, Kd, TARGET_ANGLE)
    
    # Initialize complementary filter
    angle_filter = ComplementaryFilter(alpha=FILTER_ALPHA, initial_angle=0.0)
    
    # Initialize low-pass filters for noise reduction
    rate_filter = LowPassFilter(alpha=RATE_FILTER_ALPHA, initial_value=0.0) if ENABLE_RATE_FILTER else None
    angle_lp_filter = LowPassFilter(alpha=ANGLE_FILTER_ALPHA, initial_value=0.0) if ENABLE_ANGLE_FILTER else None
    
    # Ensure motors are stopped
    stop()
    
    print("Starting balance control...")
    time.sleep(0.5)
    
    try:
        loop_count = 0
        start_time = time.time()
        
        while True:
            loop_start = time.time()
            
            # Read IMU
            imu_data = read_imu()
            accel_roll = imu_data['roll']  # Raw accelerometer angle
            roll_rate_raw = imu_data['roll_rate']  # Raw gyroscope rate
            
            # Apply fixed offset: IMU roll -180° = 0° (robot flat) + manual offset
            accel_roll_corrected = normalize_angle(accel_roll - roll_offset)
            
            # Filter gyroscope rate to reduce noise in derivative term
            # This is critical for reducing twitching with high Kd values
            if ENABLE_RATE_FILTER:
                roll_rate = rate_filter.update(roll_rate_raw)
            else:
                roll_rate = roll_rate_raw
            
            # Update complementary filter (use filtered rate if enabled)
            roll_filtered = angle_filter.update(accel_roll_corrected, roll_rate, dt=LOOP_TIME)
            
            # Optional: Apply additional low-pass filter to angle for extra smoothing
            if ENABLE_ANGLE_FILTER:
                roll_corrected = angle_lp_filter.update(roll_filtered)
            else:
                roll_corrected = roll_filtered
            
            # Check if robot has fallen (too far from upright)
            if abs(roll_corrected) > 45.0:
                print(f"\n⚠️  Robot fallen! Angle: {roll_corrected:.1f}°")
                stop()
                print("Waiting for robot to be upright again...")
                while abs(roll_corrected) > 20.0:
                    imu_data = read_imu()
                    accel_roll = imu_data['roll']
                    roll_rate_raw = imu_data['roll_rate']
                    accel_roll_corrected = normalize_angle(accel_roll - roll_offset)
                    # Apply rate filter if enabled
                    if ENABLE_RATE_FILTER:
                        roll_rate = rate_filter.update(roll_rate_raw)
                    else:
                        roll_rate = roll_rate_raw
                    roll_filtered = angle_filter.update(accel_roll_corrected, roll_rate, dt=0.1)
                    # Apply angle filter if enabled
                    if ENABLE_ANGLE_FILTER:
                        roll_corrected = angle_lp_filter.update(roll_filtered)
                    else:
                        roll_corrected = roll_filtered
                    time.sleep(0.1)
                print("Resuming balance control...")
                pid.integral = 0.0  # Reset integral on recovery
                angle_filter.reset(0.0)  # Reset filter on recovery
                if rate_filter:
                    rate_filter.reset(0.0)
                if angle_lp_filter:
                    angle_lp_filter.reset(0.0)
                time.sleep(0.5)
                continue
            
            # Compute PID output
            pid_output = pid.compute(roll_corrected, roll_rate)
            
            # Convert PID output to motor speed (what would be applied)
            # Positive: if robot tilts right (positive roll),
            # we need to move right (positive motor speed) to catch it
            motor_speed = pid_output / 100.0  # Scale down PID output
            
            # Apply deadband (stop if very close to target)
            if abs(roll_corrected) < ANGLE_DEADBAND and abs(roll_rate) < 5.0:
                motor_speed = 0.0
            
            # Limit motor speed
            motor_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, motor_speed))
            
            # Apply to both motors equally (for balance, not turning)
            left_motor_speed = motor_speed
            right_motor_speed = motor_speed
            
            # Apply motor speeds
            set_motor_speed(left_motor_speed, right_motor_speed)
            
            # Helper function to show direction
            def motor_direction(speed):
                if speed > 0.001:
                    return "F"  # Forward
                elif speed < -0.001:
                    return "B"  # Backward
                else:
                    return "-"  # Stopped
            
            # Print status every 25 loops (~0.5 seconds at 50Hz)
            if loop_count % 25 == 0:
                elapsed = time.time() - start_time
                rate_display = f"{roll_rate:7.1f}°/s"
                if ENABLE_RATE_FILTER:
                    rate_display += f" (raw: {roll_rate_raw:6.1f})"
                print(f"Time: {elapsed:6.2f}s | "
                      f"Roll: {roll_corrected:7.2f}° | "
                      f"Accel: {accel_roll_corrected:7.2f}° | "
                      f"Rate: {rate_display} | "
                      f"PID: {pid_output:8.2f} | "
                      f"Left: {left_motor_speed:6.3f} ({motor_direction(left_motor_speed)}) | "
                      f"Right: {right_motor_speed:6.3f} ({motor_direction(right_motor_speed)})")
            
            loop_count += 1
            
            # Maintain consistent loop timing
            elapsed = time.time() - loop_start
            sleep_time = LOOP_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Loop is taking too long, warn user
                if loop_count % 100 == 0:
                    print(f"\n⚠️  Loop time exceeded! ({elapsed*1000:.1f}ms > {LOOP_TIME*1000:.1f}ms)")
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stop()
        print("Motors stopped. Robot safe.")

if __name__ == "__main__":
    main()
