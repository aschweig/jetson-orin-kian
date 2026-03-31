"""Microphone singleton — one sd.InputStream shared by VAD and barge-in."""

import time

import numpy as np
import sounddevice as sd

DEVICE_RATE = 48000
DEVICE_CHUNK_SAMPLES = 1536  # matches VAD's 512 samples at 16kHz × 3


class AudioInput:
    """Singleton persistent input stream with a ring buffer for RMS."""

    _instance: "AudioInput | None" = None

    def __init__(self):
        self._callbacks: list = []
        # Ring buffer for barge-in level checks (keeps last ~50ms at 48kHz)
        self._level_len = int(DEVICE_RATE * 0.05)
        self._level_buf = np.zeros(self._level_len, dtype=np.float32)
        self._level_pos = 0
        self._stream = sd.InputStream(
            samplerate=DEVICE_RATE,
            channels=1,
            dtype="float32",
            blocksize=DEVICE_CHUNK_SAMPLES,
            callback=self._audio_callback,
            latency="high",
        )

    @classmethod
    def get(cls) -> "AudioInput":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()

    def add_callback(self, cb):
        self._callbacks.append(cb)

    def remove_callback(self, cb):
        self._callbacks.remove(cb)

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0]
        # Update ring buffer (always, even while VAD is paused)
        n = len(mono)
        pos = self._level_pos
        end = pos + n
        if end <= self._level_len:
            self._level_buf[pos:end] = mono
        else:
            first = self._level_len - pos
            self._level_buf[pos:] = mono[:first]
            self._level_buf[:n - first] = mono[first:]
        self._level_pos = end % self._level_len
        # Forward to registered callbacks (e.g. VAD)
        for cb in self._callbacks:
            cb(mono)

    def rms(self) -> float:
        """Return RMS of the ring buffer."""
        return float(np.sqrt(np.mean(self._level_buf ** 2)))


def mic_rms(duration: float = 0.25, interval: float = 0.05) -> float:
    """Sample the ring buffer multiple times over duration, return the max RMS.

    This catches speech anywhere in the window, not just at the end.
    """
    mic = AudioInput.get()
    peak = 0.0
    elapsed = 0.0
    while elapsed < duration:
        time.sleep(interval)
        elapsed += interval
        rms = mic.rms()
        if rms > peak:
            peak = rms
    return peak
