"""LLM backend selection and common interface."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

SYSTEM_PROMPT_TEMPLATE = (
    "You are {name}, a helpful educating cartoon animal -- but it's a secret. "
    "You are talking to an imaginative and curious child in grade {grade}. "
    "Keep responses concise and conversational. Your output "
    "will be spoken aloud, so never use markdown, asterisks, bullet points, emojis, "
    "or any formatting. Use plain spoken English only. "
    "Simple LaTeX is okay for math (e.g. $\\frac{{a}}{{b}}$, $x^2$, $\\sqrt{{x}}$)."
)

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


def system_prompt() -> str:
    grade_str = _GRADE_NAMES.get(child_grade, str(child_grade))
    return SYSTEM_PROMPT_TEMPLATE.format(name=assistant_name, grade=grade_str)

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


def update_system_prompt(history: list[dict]) -> None:
    """If the last two assistant responses share a repeated phrase, add an
    avoidance hint to the system prompt."""
    assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
    base = system_prompt()

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
    elif backend == "mlc":
        from kian.llm_mlc import MlcLLM
        return MlcLLM(model=model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}  (choose 'llamacpp' or 'mlc')")
