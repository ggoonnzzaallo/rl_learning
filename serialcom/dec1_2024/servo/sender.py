import serial
import time
import random

# Initialize the serial connection (adjust port and baud rate as needed)
arduino = serial.Serial(port='/dev/tty.usbmodem1101', baudrate=9600, timeout=1)

#use this in the terminal to find the port: ls /dev/tty.*

time.sleep(2)  # Wait for Arduino to initialize

# Example function to send data and receive a response
def send_to_arduino(value):
    try:
        # Send the value as a string
        arduino.write(f"{value}\n".encode())
        
        # Wait for a response
        response = arduino.readline().decode().strip()
        print(f"Arduino response: {response}")
    except Exception as e:
        print(f"Error: {e}")

# Example RL loop
for i in range(10):
    action = random.randint(0, 180)  # Random angle between 0 and 180 degrees
    print(f"Sending: {action}")
    send_to_arduino(action)
    time.sleep(10)  # Increased delay to 2 seconds
