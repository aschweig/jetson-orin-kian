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
    tts = TTSPlayer()
    print("Ready. Speak! (press Q + Enter to quit)")

    quit_task = asyncio.create_task(watch_quit(shutdown))

    async for speech_audio in vad.stream_speech():
        if shutdown.is_set():
            break

        # STT
        audio_sec = len(speech_audio) / 16000
        print(f"\n[VAD] {audio_sec:.1f}s of speech detected")
        t0 = time.monotonic()
        text = await stt.transcribe(speech_audio)
        stt_ms = (time.monotonic() - t0) * 1000
        if not text.strip():
            print(f"[STT] empty ({stt_ms:.0f}ms)")
            continue
        print(f"[STT] {stt_ms:.0f}ms: {text}")

        # LLM → TTS streaming
        buf = ""
        async for token in llm.chat_stream(text):
            if shutdown.is_set():
                break
            buf += token
            # Check for punctuation split point with enough text
            m = SPLIT_RE.search(buf)
            if m and m.end() >= MIN_CHUNK_LEN:
                fragment = buf[: m.end()].strip()
                buf = buf[m.end() :]
                if fragment:
                    print(fragment, end=" ", flush=True)
                    await tts.speak(fragment)

        # Flush remaining text
        if buf.strip() and not shutdown.is_set():
            print(buf.strip(), end=" ", flush=True)
            await tts.speak(buf.strip())

        tts.drain()
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
