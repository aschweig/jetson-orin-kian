"""Microphone utilities."""

import time
import numpy as np
import sounddevice as sd

DEVICE_RATE = 48000


def mic_rms(duration: float = 0.05) -> float:
    """Record a short burst from the mic and return its RMS.

    Includes a small settling delay to avoid interfering with sd.play().
    """
    time.sleep(0.03)
    audio = sd.rec(int(DEVICE_RATE * duration), samplerate=DEVICE_RATE,
                   channels=1, dtype="float32", blocking=True)
    return float(np.sqrt(np.mean(audio ** 2)))
