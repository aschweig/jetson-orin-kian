"""Text-to-speech using Piper with playback thread."""

import asyncio
import io
import queue
import threading
import wave

import numpy as np
import sounddevice as sd
from pathlib import Path
from piper import PiperVoice
from piper.config import SynthesisConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TTS_SAMPLE_RATE = 22050  # Piper default


class TTSPlayer:
    """Synthesizes text and plays audio via a background thread."""

    def __init__(self, model_path: str = str(PROJECT_ROOT / "models" / "en_US-lessac-medium.onnx"),
                 speed: float = 1.0, pitch_shift: float = 1.0):
        self._voice = PiperVoice.load(model_path)
        self._syn_config = SynthesisConfig(length_scale=1.0 / speed)
        self._playback_rate = int(TTS_SAMPLE_RATE * pitch_shift)  # higher = raised pitch
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    def _playback_worker(self):
        while True:
            chunk = self._audio_queue.get()
            if chunk is None:
                break
            sd.play(chunk, samplerate=self._playback_rate)
            sd.wait()
            self._audio_queue.task_done()

    def _synthesize(self, text: str) -> np.ndarray:
        chunks = []
        for audio_chunk in self._voice.synthesize(text, self._syn_config):
            chunks.append(audio_chunk.audio_float_array)
        return np.concatenate(chunks)

    async def speak(self, text: str):
        """Synthesize text and queue for playback."""
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, self._synthesize, text)
        self._audio_queue.put(audio)

    def drain(self):
        """Wait for all queued audio to finish playing."""
        self._audio_queue.join()

    def stop(self):
        """Signal playback thread to exit."""
        self._audio_queue.put(None)
