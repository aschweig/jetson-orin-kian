"""Kian voice assistant - main pipeline."""

import asyncio
import re
import sys
import time

from kian.vad import VADStream
from kian.stt import STT
from kian.llm import LLM
from kian.tts import TTSPlayer

# Split on sentence/clause boundaries, but only when we have enough text
SPLIT_RE = re.compile(r"[.!?,;:\n]")
MIN_CHUNK_LEN = 25
SILENCE_RESET_S = 8 * 60  # reset conversation after 8 minutes of silence


async def watch_quit(shutdown: asyncio.Event):
    """Watch stdin for 'q' or 'Q' to trigger graceful shutdown."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    while True:
        line = await reader.readline()
        if not line or line.strip().lower() == b"q":
            print("\nShutting down...")
            shutdown.set()
            return


async def pipeline():
    shutdown = asyncio.Event()

    print("Loading models...")
    vad = VADStream()
    stt = STT()
    llm = LLM()
    tts = TTSPlayer(pitch_shift=1.15)
    print("Ready. Speak! (press Q + Enter to quit)")

    quit_task = asyncio.create_task(watch_quit(shutdown))
    last_speech = time.monotonic()

    async for speech_audio in vad.stream_speech():
        if shutdown.is_set():
            break

        # Reset context after prolonged silence
        now = time.monotonic()
        if now - last_speech > SILENCE_RESET_S:
            llm.reset()
            print("[RESET] conversation context cleared (8 min silence)")
        last_speech = now

        # Pause listening during processing
        vad.pause()

        # STT
        audio_sec = len(speech_audio) / 16000
        print(f"\n[VAD {audio_sec:.1f}s]")
        t0 = time.monotonic()
        text = await stt.transcribe(speech_audio)
        stt_s = time.monotonic() - t0
        if not text.strip():
            print(f"[STT {stt_s:.1f}s] (empty)")
            vad.resume()
            continue
        print(f"[STT {stt_s:.1f}s] {text}")

        # LLM → TTS streaming
        t_llm = time.monotonic()
        first_token = True
        buf = ""
        async for token in llm.chat_stream(text):
            if shutdown.is_set():
                break
            if first_token:
                print(f"[LLM {time.monotonic() - t_llm:.1f}s] first token")
                first_token = False
            buf += token
            # Check for punctuation split point with enough text
            m = SPLIT_RE.search(buf)
            if m and m.end() >= MIN_CHUNK_LEN:
                fragment = buf[: m.end()].strip()
                buf = buf[m.end() :]
                if fragment:
                    t_tts = time.monotonic()
                    await tts.speak(fragment)
                    print(f"[TTS {time.monotonic() - t_tts:.1f}s] {fragment}")

        # Flush remaining text
        if buf.strip() and not shutdown.is_set():
            t_tts = time.monotonic()
            await tts.speak(buf.strip())
            print(f"[TTS {time.monotonic() - t_tts:.1f}s] {buf.strip()}")

        print(f"[LLM {time.monotonic() - t_llm:.1f}s] total")
        tts.drain()

        # Resume listening
        vad.resume()
        print()

    quit_task.cancel()
    tts.stop()
    print("Goodbye.")


def main():
    try:
        asyncio.run(pipeline())
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
