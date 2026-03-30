"""Quick test: cycle the Pi Traffic Light on Jetson GPIO pins 29, 31, 32."""

import time
import Jetson.GPIO as GPIO

# Physical BOARD pin numbers
RED = 29
YELLOW = 31
GREEN = 32

GPIO.setmode(GPIO.BOARD)
GPIO.setup(RED, GPIO.OUT)
GPIO.setup(YELLOW, GPIO.OUT)
GPIO.setup(GREEN, GPIO.OUT)

try:
    # All on
    print("All ON")
    GPIO.output(RED, True)
    GPIO.output(YELLOW, True)
    GPIO.output(GREEN, True)
    time.sleep(2)

    # All off
    print("All OFF")
    GPIO.output(RED, False)
    GPIO.output(YELLOW, False)
    GPIO.output(GREEN, False)
    time.sleep(1)

    # Cycle each
    for name, pin in [("RED", RED), ("YELLOW", YELLOW), ("GREEN", GREEN)]:
        print(f"{name} on")
        GPIO.output(pin, True)
        time.sleep(1)
        GPIO.output(pin, False)

    print("Done")
finally:
    GPIO.cleanup()
