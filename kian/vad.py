"""Voice Activity Detection using Silero VAD (ONNX)."""

import asyncio
from pathlib import Path

import numpy as np
import onnxruntime
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAD_MODEL_PATH = str(PROJECT_ROOT / "models" / "silero_vad.onnx")

DEVICE_RATE = 48000  # native rate for PulseAudio default
VAD_RATE = 16000
DOWNSAMPLE_RATIO = DEVICE_RATE // VAD_RATE  # 3
CHUNK_SAMPLES_16K = 512  # Silero VAD expects 512 samples at 16kHz (32ms)
DEVICE_CHUNK_SAMPLES = CHUNK_SAMPLES_16K * DOWNSAMPLE_RATIO  # 1536 at 48kHz


class SileroVAD:
    """Silero VAD via ONNX Runtime — no PyTorch needed."""

    def __init__(self, model_path: str = VAD_MODEL_PATH):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(model_path, sess_options=opts)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)  # 64 samples context at 16kHz

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """Return speech probability for a 512-sample float32 chunk at 16kHz."""
        # Prepend context from previous chunk (Silero requires this)
        x = np.concatenate([self._context, audio_chunk])[np.newaxis, :].astype(np.float32)
        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(VAD_RATE, dtype=np.int64),
        }
        output, self._state = self._session.run(None, ort_inputs)
        self._context = audio_chunk[-64:]  # save last 64 samples as context
        return float(output[0][0])


class VADStream:
    """Streams mic audio and yields speech segments via Silero VAD."""

    def __init__(self, threshold: float = 0.5, silence_ms: int = 1200):
        self._vad = SileroVAD()
        self._threshold = threshold
        self._silence_ms = silence_ms
        self._paused = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def pause(self):
        """Stop processing audio (call before STT/LLM/TTS)."""
        self._paused = True
        # Drain any queued audio
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def resume(self):
        """Resume processing audio (call after TTS completes)."""
        self._vad.reset()
        self._paused = False

    def _audio_callback(self, indata, frames, time, status):
        if self._paused:
            return
        audio = indata[:, 0].copy()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, audio)

    async def stream_speech(self):
        """Yield complete speech segments as float32 numpy arrays at 16kHz."""
        self._loop = asyncio.get_running_loop()
        chunk_ms = (CHUNK_SAMPLES_16K / VAD_RATE) * 1000  # 32ms
        silence_chunks = int(self._silence_ms / chunk_ms)

        stream = sd.InputStream(
            samplerate=DEVICE_RATE,
            channels=1,
            dtype="float32",
            blocksize=DEVICE_CHUNK_SAMPLES,
            callback=self._audio_callback,
        )

        with stream:
            speech_buf: list[np.ndarray] = []
            silent_count = 0
            in_speech = False

            while True:
                chunk_48k = await self._queue.get()
                if chunk_48k is None:
                    break

                chunk_16k = chunk_48k[::DOWNSAMPLE_RATIO]
                prob = self._vad(chunk_16k)

                if prob >= self._threshold:
                    speech_buf.append(chunk_16k)
                    silent_count = 0
                    in_speech = True
                elif in_speech:
                    speech_buf.append(chunk_16k)
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        yield np.concatenate(speech_buf)
                        speech_buf.clear()
                        silent_count = 0
                        in_speech = False
                        self._vad.reset()
