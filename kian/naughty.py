"""Naughty word/phrase detection for streaming LLM output."""

import codecs
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NAUGHTY_PATH = PROJECT_ROOT / "naughty.txt"

_STRIP = re.compile(r"[^\w\s]")


def _load_phrases() -> tuple[set[tuple[str, ...]], int]:
    """Load naughty phrases, return (set of word-tuples, max phrase length)."""
    phrases: set[tuple[str, ...]] = set()
    max_len = 1
    for line in NAUGHTY_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = codecs.decode(line, "rot_13")
        words = tuple(_STRIP.sub("", line).lower().split())
        if words:
            phrases.add(words)
            max_len = max(max_len, len(words))
    return phrases, max_len


_PHRASES, _MAX_LEN = _load_phrases()


class NaughtyDetector:
    """Checks a stream of tokens for naughty words/phrases."""

    def __init__(self):
        self._words: list[str] = []

    def check(self, token: str) -> bool:
        """Add token to the stream. Return True if a naughty phrase is detected."""
        # Tokenize the new text into words
        new_words = _STRIP.sub("", token).lower().split()
        self._words.extend(new_words)

        # Keep only the last _MAX_LEN words (sliding window)
        if len(self._words) > _MAX_LEN:
            self._words = self._words[-_MAX_LEN:]

        # Check all n-gram sizes against the phrase set
        for n in range(1, min(len(self._words), _MAX_LEN) + 1):
            ngram = tuple(self._words[-n:])
            if ngram in _PHRASES:
                return True
        return False

    def reset(self):
        self._words.clear()
