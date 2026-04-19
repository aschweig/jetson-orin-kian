"""Kian voice assistant - main pipeline."""

import argparse
import asyncio
import re
import sys
import time

from pathlib import Path

from kian.vad import VADStream
from kian.stt import STT
from kian import llm as llm_mod
from kian.llm import create_llm
from kian.naughty import NaughtyDetector
from kian.tts import TTSPlayer, VOLUME_WHISPER, VOLUME_INSIDE, VOLUME_LOUD, VOLUME_SHOUT
from kian import leds
from kian.mic import mic_rms
from kian.wiki import WikiLookup
from kian import safety

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VALID_NAMES = {
    n.strip().lower()
    for n in (PROJECT_ROOT / "names.txt").read_text().splitlines()
    if n.strip()
}

# Barge-in detection: RMS threshold for mic level between TTS sentences
BARGEIN_THRESHOLD = 0.05  # 0.0–1.0, tune based on your mic

# Split on sentence boundaries only (punctuation followed by space).
SPLIT_RE = re.compile(r"([.!?])\s+")
MIN_CHUNK_LEN = 25  # don't split tiny fragments
MAX_CHUNK_LEN = 400  # force-flush buffer if no sentence boundary found
PAUSE_SILENCE_S = 0.050   # 50ms silence to let TTS echo decay
BARGEIN_LISTEN_S = 0.250  # mic listen window for barge-in
BARGEIN_COOLDOWN_S = 0.80 # pause after barge-in before listening again
SILENCE_RESET_S = 8 * 60  # reset conversation after 8 minutes of silence

# --- Control phrases (matched case-insensitively, punctuation stripped) ---
# Each maps display label -> set of phrases

CONTROL_PHRASES = {
    "Reset context": {
        "lets start fresh",
        "lets start over",
        "can we start fresh",
        "can we start over",
        "forget everything",
        "reset",
        "reset please",
    },
    "Change voice": {
        "change your voice",
        "new voice",
        "switch voices",
    },
    "Reset + new voice": {
        "can i talk to someone else",
        "i want to talk to someone else",
    },
    "Whisper": {
        "whisper",
        "whisper please",
        "be quiet",
        "be really quiet",
        "shh",
        "shhh",
        "shush",
    },
    "Inside voice": {
        "inside voice",
        "use your inside voice",
        "normal voice",
        "normal volume",
    },
    "Loud": {
        "loud",
        "be loud",
        "speak up",
        "louder",
        "louder please",
        "i cant hear you",
    },
    "Shout": {
        "shout",
        "yell",
        "maximum volume",
        "max volume",
    },
    "Reset all": {
        "reset all",
    },
}

# Also: "Your name is X", "I'll call you X", "I'm in 3rd grade", etc.
# These are regex-based and listed separately at startup.

_STRIP_PUNCT = re.compile(r"[^\w\s]")

# Word-to-number map for grades spoken as words
_WORD_TO_NUM: dict[str, int] = {
    "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10,
    "eleven": 11, "eleventh": 11,
    "twelve": 12, "twelfth": 12,
}
_NUM_WORDS = "|".join(_WORD_TO_NUM)

# Patterns for name/age changes (matched on original text, case-insensitive)
_NAME_RE = re.compile(
    r"^(?:your name is|you are|i(?:'ll| will) (?:call|name) you)\s+(\w+)\.?$",
    re.IGNORECASE,
)
_GRADE_RE = re.compile(
    rf"^(?:i'm|im|i am)\s+(?:in\s+(?:the\s+)?)?(?:grade\s+(\d+|{_NUM_WORDS})|(\d+)(?:st|nd|rd|th)?\s*(?:grade|grader)|({_NUM_WORDS})\s*(?:grade|grader))\.?$",
    re.IGNORECASE,
)
_GRADE_WORD_RE = re.compile(
    r"^(?:i'm|im|i am)\s+(?:in\s+(?:the\s+)?)?(pre-?\s*k|kindergart[ea]n)\.?$",
    re.IGNORECASE,
)


_ACTION_MAP = {
    "Reset context": "reset",
    "Change voice": "new_voice",
    "Reset + new voice": "reset_and_new_voice",
    "Whisper": "whisper",
    "Inside voice": "inside_voice",
    "Loud": "loud",
    "Shout": "shout",
    "Reset all": "reset_all",
}


def _match_control(text: str) -> tuple[str, str | None] | None:
    """Return (action, extra_data) or None."""
    normalized = _STRIP_PUNCT.sub("", text).lower().strip()
    for label, phrases in CONTROL_PHRASES.items():
        if normalized in phrases:
            return (_ACTION_MAP[label], None)

    # "I'm in kindergarten" / "I'm in pre-K"
    m = _GRADE_WORD_RE.search(text)
    if m:
        word = m.group(1).lower().replace(" ", "")
        grade = -1 if word.startswith("pre") else 0
        return ("set_grade", str(grade))

    # "I'm in 3rd grade" / "I am in grade 3" / "I'm in grade two"
    m = _GRADE_RE.search(text)
    if m:
        raw = (m.group(1) or m.group(2) or m.group(3)).lower()
        grade = _WORD_TO_NUM.get(raw) or int(raw)
        grade = max(1, min(12, grade))
        return ("set_grade", str(grade))

    # "Your name is Sophie" / "I'll call you Sophie"
    m = _NAME_RE.search(text)
    if m:
        name = m.group(1).capitalize()
        if name.lower() in _VALID_NAMES:
            return ("set_name", name)
        else:
            return ("invalid_name", name)

    return None



def _is_repetitive(text: str, phrase_len: int = 3, max_repeats: int = 5) -> bool:
    """Detect if text contains a short phrase repeated excessively."""
    words = text.lower().split()
    if len(words) < phrase_len * max_repeats:
        return False
    # Check the last N words for repeating n-grams
    window = words[-(phrase_len * max_repeats * 3):]
    counts: dict[tuple, int] = {}
    for i in range(len(window) - phrase_len + 1):
        ngram = tuple(window[i : i + phrase_len])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] >= max_repeats:
            return True
    return False


async def _stream_response(llm, tts, user_text, shutdown):
    """Producer-consumer LLM→TTS pipeline with barge-in detection.

    Producer (task 1): streams LLM tokens, splits into sentences, puts them
    on a queue.  Stops early if the barge-in flag is set or if repetition
    is detected (LLM loop).

    Consumer (task 2): takes sentences off the queue, plays them with a short
    tail silence, then listens on the mic for barge-in.  Sets the flag and
    exits if the user speaks.

    Returns (interrupted, naughty_hit, llm_loop, full_response).
    """
    bargein = asyncio.Event()            # atomic flag
    sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
    naughty_hit = False
    safety_hit = False
    llm_loop = False
    full_response: list[str] = []

    async def producer():
        nonlocal naughty_hit, safety_hit, llm_loop
        naughty = NaughtyDetector()
        loop = asyncio.get_event_loop()
        buf = ""
        first_token = True
        t_llm = time.monotonic()

        async def _check_and_enqueue(text: str):
            """Safety-check a sentence, then enqueue for TTS or abort."""
            nonlocal safety_hit
            # Throttle: wait until consumer needs more before running CPU-heavy check
            while sentence_q.qsize() >= 2 and not bargein.is_set():
                await asyncio.sleep(0.1)
            if bargein.is_set():
                return True  # abort silently
            t_safety = time.monotonic()
            safe = await loop.run_in_executor(None, safety.classify, text)
            elapsed = time.monotonic() - t_safety
            if safe:
                print(f"[SAFETY {elapsed:.1f}s] SAFE")
            else:
                print(f"[SAFETY {elapsed:.1f}s] UNSAFE -- {text}")
                safety_hit = True
                return False
            await sentence_q.put(text)
            return True

        async for token in llm.chat_stream(user_text):
            if shutdown.is_set() or bargein.is_set():
                break
            if first_token:
                print(f"[LLM {time.monotonic() - t_llm:.1f}s] first token")
                first_token = False
                t_last_token = time.monotonic()
            else:
                now = time.monotonic()
                gap = now - t_last_token
                if gap > 2.0:
                    print(f"[LLM] token gap {gap:.1f}s")
                t_last_token = now

            if naughty.check(token):
                print("[NAUGHTY] blocked")
                naughty_hit = True
                break

            full_response.append(token)
            buf += token

            # Repetition detection on accumulated output
            if _is_repetitive(buf):
                print("[LLM] repetition loop detected, stopping")
                llm_loop = True
                break

            m = None
            for candidate in SPLIT_RE.finditer(buf):
                if candidate.end() >= MIN_CHUNK_LEN:
                    m = candidate
                    break
            if m:
                fragment = buf[: m.end()].strip()
                buf = buf[m.end() :]
                if fragment:
                    if not await _check_and_enqueue(fragment):
                        break
            elif len(buf) >= MAX_CHUNK_LEN:
                # No sentence boundary found — force-flush to avoid runaway buffer
                print("[LLM] buffer overflow, force-flushing")
                if not await _check_and_enqueue(buf.strip()):
                    break
                buf = ""

        # Flush remaining text
        if buf.strip() and not bargein.is_set() and not shutdown.is_set() and not naughty_hit and not safety_hit and not llm_loop:
            await _check_and_enqueue(buf.strip())

        # Signal consumer to exit (only if not interrupted)
        if not bargein.is_set():
            await sentence_q.put(None)

        print(f"[LLM {time.monotonic() - t_llm:.1f}s] done ({len(full_response)} tokens)")

    async def consumer():
        loop = asyncio.get_event_loop()
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                break

            print(f"[TTS] {sentence}")
            await tts.speak(sentence, tail_silence=PAUSE_SILENCE_S)
            tts.drain()

            # Listen for barge-in in the inter-sentence gap
            rms = await loop.run_in_executor(None, mic_rms, BARGEIN_LISTEN_S, 0.05)
            if rms >= BARGEIN_THRESHOLD:
                print("[INTERRUPTED]")
                tts.flush()
                bargein.set()
                break

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    await asyncio.gather(producer_task, consumer_task)

    # Drain any leftover sentences from the queue
    while not sentence_q.empty():
        try:
            sentence_q.get_nowait()
        except asyncio.QueueEmpty:
            break

    return bargein.is_set(), naughty_hit or safety_hit, llm_loop, full_response


async def watch_quit(shutdown: asyncio.Event):
    """Watch stdin for 'q' or 'Q' to trigger graceful shutdown."""
    loop = asyncio.get_event_loop()

    def _read_stdin():
        for line in sys.stdin:
            if line.strip().lower() == "q":
                return

    await loop.run_in_executor(None, _read_stdin)
    print("\nShutting down...")
    shutdown.set()


async def pipeline(backend: str = "llamacpp", model: str | None = None):
    shutdown = asyncio.Event()

    from tests.test_control_phrases import run as test_control_phrases
    test_control_phrases()

    print("Loading models...")
    safety.load()
    vad = VADStream()
    stt = STT()
    llm = create_llm(backend=backend, model=model)
    tts = TTSPlayer(pitch_shift=1.15)
    wiki = WikiLookup()
    if hasattr(llm, "set_on_evict_title"):
        def _evict_wiki_title(title: str):
            if title in wiki._retrieved:
                wiki._retrieved.discard(title)
                print(f"[WIKI] evicted: {title}")
        llm.set_on_evict_title(_evict_wiki_title)
    from kian.llm import volume as saved_volume
    tts.set_volume(saved_volume if saved_volume is not None else VOLUME_INSIDE, persist=False)
    leds.idle()
    print("Control phrases:")
    for label, phrases in CONTROL_PHRASES.items():
        print(f"  {label}:")
        for p in sorted(phrases):
            print(f"    - \"{p}\"")
    print("  Set name:")
    print("    - \"Your name is X\"")
    print("    - \"I'll call you X\"")
    print("    - \"I will name you X\"")
    print("  Set grade:")
    print("    - \"I'm in Nth grade\"")
    print("    - \"I'm in grade N\"")
    print()
    grade_str = llm_mod._GRADE_NAMES.get(llm_mod.child_grade, f"grade {llm_mod.child_grade}")
    model_name = getattr(llm, "model_name", None) or "(unknown)"
    print(f"Ready. I am {llm_mod.assistant_name}, talking to a child in {grade_str}. LLM: {model_name}")
    quit_task = None
    if sys.stdin.isatty():
        print("(press Q + Enter to quit)")
        quit_task = asyncio.create_task(watch_quit(shutdown))
    await tts.speak("Hey Hi!")
    tts.drain()

    print()
    last_speech = time.monotonic()

    async for speech_audio in vad.stream_speech():
        if shutdown.is_set():
            break

        # Reset context after prolonged silence
        now = time.monotonic()
        if now - last_speech > SILENCE_RESET_S:
            llm.reset()
            wiki.reset()
            print("[RESET] conversation context cleared (8 min silence)")
        last_speech = now

        # Pause listening during processing
        vad.pause()
        leds.busy()
        tts.beep()

        # STT
        audio_sec = len(speech_audio) / 16000
        print(f"\n[VAD {audio_sec:.1f}s]")
        t0 = time.monotonic()
        text = await stt.transcribe(speech_audio)
        stt_s = time.monotonic() - t0
        if not text.strip():
            print(f"[STT {stt_s:.1f}s] (empty)")
            leds.idle()
            vad.resume()
            continue
        print(f"[STT {stt_s:.1f}s] {text}")

        # Check if user is still speaking (barge-in at end of STT)
        rms = await asyncio.get_event_loop().run_in_executor(None, mic_rms, BARGEIN_LISTEN_S, 0.05)
        if rms >= BARGEIN_THRESHOLD:
            print(f"[STT barge-in RMS {rms:.3f} — still speaking, re-listening]")
            leds.idle()
            vad.resume()
            continue

        # Check for naughty input — deflect without adding to context
        input_naughty = NaughtyDetector()
        if any(input_naughty.check(w) for w in text.split()):
            print("[NAUGHTY] input blocked")
            await tts.speak("I'm not sure what I'm hearing.")
            tts.drain()
            tts.beep()
            tts.drain()
            leds.idle()
            vad.resume()
            continue

        # Check for control phrases
        ctrl = _match_control(text)
        if ctrl:
            action, data = ctrl
            print(f"[CTRL] {action} {data or ''}")
            if action in ("reset", "reset_and_new_voice"):
                llm.reset()
                wiki.reset()
            if action in ("new_voice", "reset_and_new_voice"):
                tts.drain()
                tts.change_voice()
            if action == "invalid_name":
                response = "Sorry, I wouldn't even know what to call myself with that name!"
            elif action == "set_name":
                llm_mod.assistant_name = data
                llm_mod.save_settings()
                llm.reset()
                wiki.reset()
                response = f"OK! My name is {data} now!"
            elif action == "set_grade":
                llm_mod.child_grade = int(data)
                llm_mod.save_settings()
                llm.reset()
                wiki.reset()
                grade_str = llm_mod._GRADE_NAMES.get(int(data), f"grade {data}")
                response = f"Got it! {grade_str.capitalize()}!"
            elif action == "reset_all":
                llm_mod.assistant_name = "Kian"
                llm_mod.save_settings()
                llm.reset()
                wiki.reset()
                tts.set_voice("en_GB-alba-medium")
                response = "OK! I'm Kian again!"
            elif action == "whisper":
                tts.set_volume(VOLUME_WHISPER)
                response = "OK, I'll whisper."
            elif action == "inside_voice":
                tts.set_volume(VOLUME_INSIDE)
                response = "OK, inside voice!"
            elif action == "loud":
                tts.set_volume(VOLUME_LOUD)
                response = "OK, nice and loud!"
            elif action == "shout":
                tts.set_volume(VOLUME_SHOUT)
                response = "OK, MAXIMUM VOLUME!"
            else:
                response = {
                    "reset": "OK, starting fresh!",
                    "new_voice": "How about this voice?",
                    "reset_and_new_voice": "Hi there! Nice to meet you!",
                }[action]
            await tts.speak(response)
            tts.drain()
            tts.beep()
            tts.drain()
            leds.idle()
            vad.resume()
            continue

        # Wiki context injection (as system-level reference)
        wiki_ctx = wiki.search(text, debug=True) if wiki.available else None
        if wiki_ctx:
            # Extract title from "[Reference: Title]\nexcerpt..."
            title_line = wiki_ctx.splitlines()[0]
            wiki_title = title_line.removeprefix("[Reference: ").removesuffix("]")
            print(f"[WIKI] {wiki_title}")
            llm.set_wiki_context(wiki_ctx, title=wiki_title)
        else:
            llm.set_wiki_context(None)
        llm_input = text

        # LLM → TTS streaming (producer-consumer with barge-in)
        interrupted, naughty_hit, llm_loop, full_response = await _stream_response(
            llm, tts, llm_input, shutdown,
        )

        if naughty_hit:
            llm.reset()
            wiki.reset()
            await tts.speak("I don't really know how to answer that.")
            tts.drain()

        if llm_loop:
            llm.reset()
            wiki.reset()
            await tts.speak("Hmm, I lost my train of thought. Can you ask me again?")
            tts.drain()

        tts.drain()

        # After barge-in, wait for mic to settle before listening again
        if interrupted:
            await asyncio.sleep(BARGEIN_COOLDOWN_S)

        # Resume listening
        tts.beep()
        tts.drain()
        leds.idle()
        vad.resume()
        print("[Listening]\n")

    if quit_task is not None:
        quit_task.cancel()
    tts.stop()
    leds.off()
    print("Goodbye.")


def parse_args():
    p = argparse.ArgumentParser(description="Kian voice assistant")
    p.add_argument(
        "--backend", choices=["server", "ollama", "llamacpp"], default="server",
        help="LLM backend (default: server). 'llamacpp' is deprecated.",
    )
    p.add_argument(
        "--model",
        help="Model path or identifier (default depends on backend)",
    )
    p.add_argument(
        "--speaker",
        help="Substring to match PulseAudio sink name (speaker)",
    )
    p.add_argument(
        "--mic",
        help="Substring to match PulseAudio source name (microphone)",
    )
    return p.parse_args()


def setup_audio_devices(speaker_match: str | None, mic_match: str | None):
    """Set PulseAudio sink/source for this process via environment variables.

    Uses PULSE_SINK / PULSE_SOURCE which override PulseAudio defaults
    per-process and can't be reset by module-default-device-restore.
    Retries for up to 30s to handle USB devices not yet available at boot.
    """
    if not speaker_match and not mic_match:
        return
    import os
    import pulsectl

    for attempt in range(15):
        with pulsectl.Pulse("kian-setup") as pulse:
            sink_name = None
            source_name = None
            if speaker_match:
                for s in pulse.sink_list():
                    if speaker_match.lower() in s.name.lower():
                        sink_name = s.name
                        break
            if mic_match:
                for s in pulse.source_list():
                    if ".monitor" in s.name:
                        continue
                    if mic_match.lower() in s.name.lower():
                        source_name = s.name
                        break

            if (sink_name or not speaker_match) and (source_name or not mic_match):
                if sink_name:
                    os.environ["PULSE_SINK"] = sink_name
                    print(f"[AUDIO] speaker: {sink_name}")
                if source_name:
                    os.environ["PULSE_SOURCE"] = source_name
                    print(f"[AUDIO] mic: {source_name}")
                return

        print(f"[AUDIO] waiting for devices... (attempt {attempt + 1}/15)")
        time.sleep(2)

    print("[AUDIO] WARNING: devices not found after 30s")


def main():
    args = parse_args()
    setup_audio_devices(args.speaker, args.mic)
    try:
        asyncio.run(pipeline(backend=args.backend, model=args.model))
    except KeyboardInterrupt:
        pass
    except (ConnectionError, FileNotFoundError, TimeoutError) as e:
        print(f"\n{e}")
    finally:
        leds.off()
        print("\nGoodbye.")
        # Force exit — daemon threads (TTS playback, stdin reader) and
        # sounddevice streams can hang otherwise.
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
