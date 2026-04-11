"""LLM backend selection and common interface."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

import re

_PROMPT_BASE = (
    "You are {name}, a helpful educating cartoon animal -- but it's a secret. "
    "You are talking to an imaginative and curious child in grade {grade}. "
    "Your output will be spoken aloud, so never use markdown, asterisks, bullet points, emojis, "
    "or any formatting. Use plain spoken English only. "
    "{length_hint}"
)

_LATEX_OK = "Simple LaTeX is okay for math (e.g. $\\frac{a}{b}$, $x^2$, $\\sqrt{x}$). "

_HINT_SHORT = _LATEX_OK + "Reply in one to three short sentences."

_HINT_EXPLAIN = (
    _LATEX_OK
    + "Explain clearly and thoroughly in about 50 to 150 words. "
    "Be accurate and organized but keep it conversational."
)

_HINT_STORY = (
    "Do not use any LaTeX. "
    "Be vivid and detailed -- use dialogue, characters, and fun descriptions "
    "to bring the subject to life. Aim for 100 to 300 words."
)

_STORY_RE = re.compile(
    r"tell me a story|think of a story|make up a story|imagine a|tell the story|tell a story"
    r"|describe a|describe the|paint a picture|can you imagine|tell me about a (time|place)"
    r"|(can|please|will you|do) share (an|your) idea for a (comedy|romance|drama|novel|book|movie|tv show|tv program|screenplay|scene|skit|joke|play|tiktok|youtube|video)",
    re.IGNORECASE,
)

_EXPLAIN_RE = re.compile(
    r"tell me about|teach me|explain to me|explain (it|this)|please explain|i want to know (how|why)|i (don't|do not) understand|^explain \w+"
    r"|can you explain|can you tell( me)? about|can you (teach|lecture|clarify|explain|share)|what do you (think|know|suppose)|what can you (tell|say)|how does|how do"
    r"|why does|why do|why is|who is|who was|where is"
    r"|what does (the |a |an )?[A-Za-z]+( [A-Za-z]+){,3} (look|sound|feel|smell) like",
    re.IGNORECASE,
)


def _pick_length_hint(user_text: str, has_wiki: bool) -> tuple[str, str]:
    """Choose response length hint based on user input. Returns (hint, mode_name)."""
    if _STORY_RE.search(user_text):
        return _HINT_STORY, "story"
    if has_wiki or _EXPLAIN_RE.search(user_text):
        return _HINT_EXPLAIN, "explain"
    return _HINT_SHORT, "short"

# Mutable settings (loaded from settings.json if it exists)
assistant_name = "Kian"
child_grade = 3
voice = None  # voice model filename (stem), or None for random
volume = None  # volume level (float 0.0–1.0), or None for default


def load_settings() -> None:
    global assistant_name, child_grade, voice, volume
    if SETTINGS_PATH.exists():
        data = json.loads(SETTINGS_PATH.read_text())
        assistant_name = data.get("assistant_name", assistant_name)
        child_grade = data.get("child_grade", child_grade)
        voice = data.get("voice", voice)
        volume = data.get("volume", volume)


def save_settings() -> None:
    SETTINGS_PATH.write_text(json.dumps({
        "assistant_name": assistant_name,
        "child_grade": child_grade,
        "voice": voice,
        "volume": volume,
    }, indent=2) + "\n")


# Load on import
load_settings()


_GRADE_NAMES = {-1: "pre-K", 0: "kindergarten"}


def system_prompt(length_hint: str = _HINT_SHORT) -> str:
    grade_str = _GRADE_NAMES.get(child_grade, str(child_grade))
    return _PROMPT_BASE.format(name=assistant_name, grade=grade_str, length_hint=length_hint)

MIN_PHRASE_WORDS = 4  # ignore common short overlaps


def _longest_common_phrase(a: str, b: str) -> str | None:
    """Find the longest common multi-word phrase between two strings."""
    words_a = a.lower().split()
    words_b = b.lower().split()
    if len(words_a) < MIN_PHRASE_WORDS or len(words_b) < MIN_PHRASE_WORDS:
        return None

    # Build set of all n-grams from b, then find longest match in a
    b_ngrams: set[tuple[str, ...]] = set()
    for n in range(MIN_PHRASE_WORDS, len(words_b) + 1):
        for i in range(len(words_b) - n + 1):
            b_ngrams.add(tuple(words_b[i : i + n]))

    best = None
    for n in range(len(words_a), MIN_PHRASE_WORDS - 1, -1):
        for i in range(len(words_a) - n + 1):
            ngram = tuple(words_a[i : i + n])
            if ngram in b_ngrams:
                best = " ".join(ngram)
                return best
    return best


def update_system_prompt(history: list[dict], user_text: str = "", has_wiki: bool = False) -> None:
    """Rewrite the system prompt with the appropriate length hint and
    repetition avoidance."""
    hint, mode = _pick_length_hint(user_text, has_wiki)
    print(f"[MODE] {mode}")
    assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
    base = system_prompt(length_hint=hint)

    if len(assistant_msgs) >= 2:
        phrase = _longest_common_phrase(assistant_msgs[-1], assistant_msgs[-2])
        if phrase:
            base += f" Try to avoid repeating the phrase: '{phrase}'."

    history[0] = {"role": "system", "content": base}


class LLMBackend(Protocol):
    async def chat_stream(self, user_text: str) -> AsyncIterator[str]: ...
    def reset(self) -> None: ...


def create_llm(backend: str = "llamacpp", model: str | None = None) -> LLMBackend:
    """Factory: instantiate the requested LLM backend."""
    if backend == "llamacpp":
        from kian.llm_llamacpp import LlamaLLM
        return LlamaLLM(model_path=model)
    elif backend == "ollama":
        from kian.llm_ollama import OllamaLLM
        return OllamaLLM(model=model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}  (choose 'llamacpp' or 'ollama')")
