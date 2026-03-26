"""Speech-to-text using faster-whisper."""

import asyncio
from functools import partial

from faster_whisper import WhisperModel

MODEL_SIZE = "tiny"  # tiny=~75MB, base=~150MB


class STT:
    def __init__(self, model_size: str = MODEL_SIZE):
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    async def transcribe(self, audio_f32_16k) -> str:
        """Transcribe a float32 16kHz numpy array to text."""
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, partial(self._model.transcribe, audio_f32_16k, beam_size=1, language="en")
        )
        return " ".join(seg.text.strip() for seg in segments)
