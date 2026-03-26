"""Voice Activity Detection using webrtcvad."""

import asyncio
import struct

import numpy as np
import sounddevice as sd
import webrtcvad

DEVICE_RATE = 48000  # native rate of USB audio device
VAD_RATE = 16000  # rate expected by webrtcvad and whisper
DOWNSAMPLE_RATIO = DEVICE_RATE // VAD_RATE  # 3
CHUNK_MS = 30  # webrtcvad requires 10, 20, or 30 ms frames
DEVICE_CHUNK_SAMPLES = int(DEVICE_RATE * CHUNK_MS / 1000)  # samples at 48kHz


def _downsample(audio: np.ndarray, ratio: int) -> np.ndarray:
    """Simple decimation downsample (48kHz -> 16kHz)."""
    return audio[::ratio]


class VADStream:
    """Streams mic audio and yields speech segments via webrtcvad."""

    def __init__(self, aggressiveness: int = 3, silence_ms: int = 500):
        self._vad = webrtcvad.Vad(aggressiveness)  # 0-3, higher = more aggressive filtering
        self.silence_ms = silence_ms
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def _audio_callback(self, indata, frames, time, status):
        audio = indata[:, 0].copy()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, audio)

    async def stream_speech(self):
        """Yield complete speech segments as float32 numpy arrays at 16kHz."""
        self._loop = asyncio.get_running_loop()
        silence_chunks = int(self.silence_ms / CHUNK_MS)

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

                chunk_16k = _downsample(chunk_48k, DOWNSAMPLE_RATIO)

                # webrtcvad needs 16-bit PCM bytes at 16kHz
                pcm = struct.pack(f"{len(chunk_16k)}h", *(int(s * 32767) for s in chunk_16k))
                is_speech = self._vad.is_speech(pcm, VAD_RATE)

                if is_speech:
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
