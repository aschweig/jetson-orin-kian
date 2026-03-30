"""Text-to-speech using Piper with playback thread."""

import asyncio
import queue
import random
import re
import threading
import wave

import numpy as np
import pulsectl
import sounddevice as sd
from pathlib import Path
from piper import PiperVoice
from piper.config import SynthesisConfig

import kian.llm as llm_mod
from kian.latex_to_speech import latex_to_speech

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TTS_SAMPLE_RATE = 22050  # Piper default

VOICES = list(MODELS_DIR.glob("en_*-medium.onnx"))

# Volume presets (PulseAudio 0.0–1.0)
VOLUME_WHISPER = 0.30
VOLUME_INSIDE = 0.50
VOLUME_LOUD = 0.80
VOLUME_SHOUT = 1.0


class TTSPlayer:
    """Synthesizes text and plays audio via a background thread."""

    def __init__(self, model_path: str | None = None,
                 speed: float = 1.0, pitch_shift: float = 1.0):
        if model_path is None:
            model_path = self._resolve_voice()
        self._model_path = Path(model_path)
        print(f"[TTS] voice: {self._model_path.stem}")
        self._voice = PiperVoice.load(model_path)
        self._syn_config = SynthesisConfig(length_scale=1.0 / speed)
        self._playback_rate = int(TTS_SAMPLE_RATE * pitch_shift)  # higher = raised pitch
        self._volume = 1.0  # 0.0–2.0
        self._beep_audio = self._load_beep()
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    @staticmethod
    def _resolve_voice() -> str:
        """Return saved voice path if valid, otherwise pick randomly."""
        if llm_mod.voice:
            matches = [v for v in VOICES if v.stem == llm_mod.voice]
            if matches:
                return str(matches[0])
        chosen = random.choice(VOICES)
        llm_mod.voice = chosen.stem
        llm_mod.save_settings()
        return str(chosen)

    @staticmethod
    def _load_beep() -> np.ndarray:
        beep_path = PROJECT_ROOT / "beepboop.wav"
        with wave.open(str(beep_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def _playback_worker(self):
        """Play audio chunks sequentially. Each chunk is played and fully
        drained before moving to the next."""
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

    # Pronunciation overrides: word → how to say it
    PRONUNCIATION = {
        "Kian": "Key-in",
    }

    # Forbidden words → safe replacements (matched case-insensitively)
    WORD_FILTER = {
        "ass": "donkey",
        "butt": "behind",
    }
    _WORD_FILTER_RE = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in WORD_FILTER) + r")\b",
        re.IGNORECASE,
    )

    def _fix_pronunciation(self, text: str) -> str:
        text = latex_to_speech(text)
        text = text.replace("*", "")
        text = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\ufe0f]', '', text)
        text = self._WORD_FILTER_RE.sub(
            lambda m: self.WORD_FILTER[m.group(0).lower()], text
        )
        for word, replacement in self.PRONUNCIATION.items():
            text = text.replace(word, replacement)
        return text

    async def speak(self, text: str, tail_silence: float = 0.0):
        """Synthesize text and queue for playback.

        If tail_silence > 0, silence is appended directly to the audio chunk
        so the stream stays fed with no gap.
        """
        text = self._fix_pronunciation(text)
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, self._synthesize, text)
        if tail_silence > 0:
            silence = np.zeros(int(self._playback_rate * tail_silence), dtype=np.float32)
            audio = np.concatenate([audio, silence])
        self._audio_queue.put(audio)

    def beep(self):
        """Play the beep-boop acknowledgment sound (non-blocking)."""
        self._audio_queue.put(self._beep_audio)

    def set_volume(self, level: float, persist: bool = True):
        """Set system volume via PulseAudio (0.0–1.0)."""
        with pulsectl.Pulse("kian") as pulse:
            sink = pulse.get_sink_by_name(pulse.server_info().default_sink_name)
            pulse.volume_set_all_chans(sink, level)
        print(f"[TTS] volume: {level:.0%}")
        if persist:
            llm_mod.volume = level
            llm_mod.save_settings()

    def change_voice(self):
        """Switch to a random different voice and persist."""
        if len(VOICES) < 2:
            return
        others = [v for v in VOICES if v.name != self._model_path.name]
        chosen = random.choice(others)
        self._model_path = chosen
        print(f"[TTS] voice: {chosen.stem}")
        self._voice = PiperVoice.load(str(chosen))
        llm_mod.voice = chosen.stem
        llm_mod.save_settings()

    def drain(self):
        """Wait for all queued audio to finish playing."""
        self._audio_queue.join()

    def reset_synth(self):
        """Reload the Piper voice model to clear any accumulated state."""
        self._voice = PiperVoice.load(str(self._model_path))

    def stop(self):
        """Signal playback thread to exit and wait for it to finish."""
        self._audio_queue.put(None)
        self._playback_thread.join(timeout=3)
