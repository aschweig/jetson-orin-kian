"""Status LEDs via Jetson GPIO (Pi Traffic Light on pins 29-32)."""

import atexit

try:
    import Jetson.GPIO as GPIO

    RED = 29
    YELLOW = 31

    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(RED, GPIO.OUT)
    GPIO.setup(YELLOW, GPIO.OUT)

    GPIO.output(RED, True)
    GPIO.output(YELLOW, False)

    atexit.register(GPIO.cleanup)
    _available = True
except Exception:
    _available = False


def busy():
    """Red on, yellow off — processing."""
    if not _available:
        return
    GPIO.output(RED, True)
    GPIO.output(YELLOW, False)


def idle():
    """Yellow on, red off — ready/listening."""
    if not _available:
        return
    GPIO.output(YELLOW, True)
    GPIO.output(RED, False)


def off():
    """All off."""
    if not _available:
        return
    GPIO.output(RED, False)
    GPIO.output(YELLOW, False)
