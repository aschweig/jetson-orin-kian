"""Simple Wikipedia context retrieval via SQLite FTS5."""

import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "models" / "simplewiki.db"

_STRIP_RE = re.compile(r"[^\w\s]")

BM25_THRESHOLD = -10.0  # more negative = more relevant; reject above this
MIN_TITLE_LEN = 6

# Common words that pollute FTS ranking
_STOPWORDS = {
    "what", "when", "where", "which", "while", "who", "whom", "whose",
    "that", "than", "then", "them", "they", "this", "those", "these",
    "there", "their", "about", "above", "after", "also", "been", "before",
    "being", "below", "between", "both", "does", "doing", "down", "during",
    "each", "from", "have", "having", "here", "into", "just", "more",
    "most", "much", "only", "other", "over", "same", "should", "some",
    "such", "tell", "very", "were", "will", "with", "would", "your",
    "could", "like", "know", "want", "help", "make", "think",
    "how", "are", "you", "the", "and", "for", "but", "not", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "its", "let",
    "say", "she", "too", "use",
    # Conversational
    "thank", "thanks", "please", "sorry", "okay", "yeah", "yes",
    "play", "game", "funny", "cool", "nice", "good", "great", "fine",
    "homework", "school",
}

# --- Topic extraction patterns ---
# Each pattern has a named group "topic" capturing the subject.
# Patterns are tried in order; first match wins.
_TOPIC_PATTERNS = [
    # "what is/are X", "what was/were X"
    r"what\s+(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?(?P<topic>.+)",
    # "tell me about X", "tell me more about X"
    r"tell\s+me\s+(?:more\s+)?about\s+(?:the\s+)?(?P<topic>.+)",
    # "explain (to me) (about) X", "explain the X"
    r"explain\s+(?:to\s+me\s+)?(?:about\s+)?(?:the\s+)?(?P<topic>.+)",
    # "how does/do X work", "how X works"
    r"how\s+(?:does|do)\s+(?P<topic>.+?)\s+work",
    r"how\s+(?P<topic>.+?)\s+works",
    # "I'd like to learn/know (more) about X"
    r"(?:i'd|id|i\s+would)\s+like\s+to\s+(?:learn|know)\s+(?:more\s+)?about\s+(?:the\s+)?(?P<topic>.+)",
    # "I want to learn/know (more) about X"
    r"i\s+want\s+to\s+(?:learn|know|talk)\s+(?:more\s+)?about\s+(?:the\s+)?(?P<topic>.+?)(?:\s+today)?",
    # "can you tell me about X"
    r"can\s+you\s+tell\s+me\s+(?:more\s+)?about\s+(?:the\s+)?(?P<topic>.+)",
    # "can you teach me about X"
    r"can\s+you\s+teach\s+me\s+(?:more\s+)?about\s+(?:the\s+)?(?P<topic>.+)",
    # "can you research/search (about/for) X"
    r"can\s+you\s+(?:research|search|study|investigate)\s+(?:about\s+|for\s+)?(?:the\s+)?(?P<topic>.+)",
    # "can you check on X"
    r"can\s+you\s+check\s+on\s+(?:the\s+)?(?P<topic>.+)",
    # "on how X works"
    r"on\s+how\s+(?P<topic>.+?)\s+works",
    # "review X", "review the basics of X"
    r"review\s+(?:the\s+)?(?:basics\s+of\s+)?(?P<topic>.+)",
    # "what are the details of X"
    r"what\s+are\s+the\s+details\s+of\s+(?P<topic>.+)",
    # "what happened in/at/during X"
    r"what\s+happened\s+(?:in|at|during)\s+(?:the\s+)?(?P<topic>.+)",
    # "who is/was X"
    r"who\s+(?:is|was)\s+(?P<topic>.+)",
    # "where is/was X"
    r"where\s+(?:is|was)\s+(?P<topic>.+)",
    # "why is/are/do/does X"
    r"why\s+(?:is|are|do|does)\s+(?:the\s+)?(?P<topic>.+)",
    # "learn more about X", "know more about X"
    r"(?:learn|know)\s+(?:more\s+)?about\s+(?:the\s+)?(?P<topic>.+)",
    # "I'm curious about X"
    r"(?:i'm|im|i\s+am)\s+curious\s+about\s+(?:the\s+)?(?P<topic>.+)",
    # "I have to do/write/present/make a report/essay/presentation/talk about X"
    r"(?:i\s+have\s+to|i\s+need\s+to|i\s+got\s+to|i've\s+got\s+to)\s+(?:do|write|present|make|give)\s+(?:an?\s+)?(?:report|essay|presentation|talk|paper|project)\s+(?:about|on)\s+(?:the\s+)?(?P<topic>.+)",
    # "X is/are/were/was cool/interesting/..." (at start of input)
    r"^(?P<topic>.+?)\s+(?:is|are|were|was)\s+(?:so\s+|really\s+|super\s+|pretty\s+)?(?:cool|weird|interesting|crazy|important|complicated|fascinating|confusing|hard|difficult|amazing|awesome|neat|strange)",
]
_TOPIC_RES = [re.compile(p, re.IGNORECASE) for p in _TOPIC_PATTERNS]


_QUALIFIER_RE = re.compile(
    r"^(?:the\s+)?(?:history|basics|details|science|study|concept|idea|story|"
    r"origin|origins|meaning|definition|purpose|importance)\s+(?:of|behind)\s+(?:the\s+)?",
    re.IGNORECASE,
)


def strip_qualifiers(topic: str) -> str | None:
    """Strip common qualifiers like 'history of' from a topic.

    Returns the stripped topic, or None if nothing changed.
    """
    stripped = _QUALIFIER_RE.sub("", topic).strip()
    return stripped if stripped != topic and len(stripped) >= 3 else None


def extract_topic(text: str) -> str | None:
    """Extract a topic from user input using regex patterns.

    Returns the topic string or None if no pattern matches.
    """
    text = text.strip().rstrip(".!?")
    for pattern in _TOPIC_RES:
        m = pattern.search(text)
        if m:
            topic = m.group("topic").strip().rstrip(".!?,")
            # Truncate at first sentence boundary
            sent_end = re.search(r"[.!?]", topic)
            if sent_end:
                topic = topic[:sent_end.start()].strip()
            # Strip leading "the"
            topic = re.sub(r"^the\s+", "", topic, flags=re.IGNORECASE)
            if len(topic) >= 3:
                return topic
    return None


def _words(text: str) -> list[str]:
    """Lowercase word list."""
    return _STRIP_RE.sub("", text).lower().split()


def _plural_variants(word: str) -> list[str]:
    """Generate simple plural/singular variants of a word."""
    variants = [word]
    if word.endswith("ies"):
        variants.append(word[:-3] + "y")
    elif word.endswith("ves"):
        variants.append(word[:-3] + "f")
        variants.append(word[:-3] + "fe")
    elif word.endswith("ses") or word.endswith("xes") or word.endswith("zes") \
            or word.endswith("ches") or word.endswith("shes"):
        variants.append(word[:-2])
    elif word.endswith("s") and not word.endswith("ss"):
        variants.append(word[:-1])
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        variants.append(word[:-1] + "ies")
    elif word.endswith("f"):
        variants.append(word[:-1] + "ves")
    elif word.endswith("fe"):
        variants.append(word[:-2] + "ves")
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        variants.append(word + "es")
    else:
        variants.append(word + "s")
    return variants


def _title_variants(title: str) -> list[str]:
    """Generate plural/singular variants of a title."""
    title_lower = title.lower()
    words = title_lower.split()
    if not words:
        return [title_lower]
    last = words[-1]
    prefix = " ".join(words[:-1])
    variants = set()
    for v in _plural_variants(last):
        full = f"{prefix} {v}".strip() if prefix else v
        variants.add(full)
    return list(variants)


class WikiLookup:
    """FTS5 search over Simple Wikipedia excerpts."""

    def __init__(self, db_path: Path = DB_PATH):
        if not db_path.exists():
            print(f"[WIKI] database not found: {db_path}")
            self._conn = None
            return
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA query_only=1")
        self._retrieved: set[str] = set()
        print(f"[WIKI] loaded {db_path.name}")

    @property
    def available(self) -> bool:
        return self._conn is not None

    _FYI_RE = re.compile(r"^Is this helpful and if yes can you explain it to me:\n(.+)$", re.MULTILINE)

    def reset(self):
        """Clear the set of already-retrieved titles."""
        self._retrieved.clear()

    def evict_from_message(self, message_content: str):
        """Remove wiki titles found in a dropped message from the retrieved set."""
        for m in self._FYI_RE.finditer(message_content):
            title = m.group(1).strip()
            if title in self._retrieved:
                self._retrieved.discard(title)
                print(f"[WIKI] evicted: {title}")
            else:
                print(f"[WIKI] WARNING: evicting '{title}' but not in retrieved set")

    def _fts_query(self, fts_expr: str, limit: int = 10) -> list[tuple[str, str]]:
        """Run FTS5 search with a pre-built expression, return list of (title, excerpt)."""
        try:
            return self._conn.execute(
                """
                SELECT a.title, a.excerpt
                FROM articles_fts AS f
                JOIN articles AS a ON a.id = f.rowid
                WHERE articles_fts MATCH ?
                  AND a.title NOT IN ({})
                  AND bm25(articles_fts) <= ?
                ORDER BY bm25(articles_fts) LIMIT ?
                """.format(",".join("?" * len(self._retrieved))),
                (fts_expr, *self._retrieved, BM25_THRESHOLD, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    def _fts_unquoted(self, topic: str, limit: int = 10) -> list[tuple[str, str]]:
        """FTS search with OR of individual words (broad)."""
        words = _words(topic)
        fts_words = {w for w in words if len(w) >= 3 and w not in _STOPWORDS}
        if not fts_words:
            return []
        fts_terms = " OR ".join(f'"{w}"' for w in fts_words)
        return self._fts_query(fts_terms, limit)

    def _fts_quoted(self, topic: str, limit: int = 10) -> list[tuple[str, str]]:
        """FTS search with exact phrase match (specific)."""
        # FTS5 phrase: all words in sequence
        words = _words(topic)
        if len(words) < 2:
            return []  # single word is same as unquoted
        phrase = " ".join(words)
        return self._fts_query(f'"{phrase}"', limit)

    def _lookup_by_title(self, topic: str) -> tuple[str, str] | None:
        """Direct DB lookup by title (case-insensitive, with plural variants)."""
        if not self._conn:
            return None
        candidates = _title_variants(topic)
        for candidate in candidates:
            row = self._conn.execute(
                "SELECT title, excerpt FROM articles WHERE title = ? COLLATE NOCASE AND title NOT IN ({})"
                .format(",".join("?" * len(self._retrieved))),
                (candidate, *self._retrieved),
            ).fetchone()
            if row:
                return row
        return None

    def _resolve_alias(self, topic: str) -> str:
        """If topic matches a redirect alias, return the target title."""
        if not self._conn:
            return topic
        row = self._conn.execute(
            "SELECT title FROM aliases WHERE alias = ? COLLATE NOCASE",
            (topic,),
        ).fetchone()
        return row[0] if row else topic

    def _find_title_match(self, rows: list[tuple[str, str]], topic: str) -> tuple[str, str] | None:
        """Find the first result whose title matches the topic (with plural variants).

        Also checks aliases (redirects) to resolve alternate names.
        """
        # Resolve alias first: "World War 2" -> "World War II"
        resolved = self._resolve_alias(topic)
        # Check both the original topic and the resolved form
        candidates = {topic.lower(), resolved.lower()}
        for title, excerpt in rows:
            if len(title) < MIN_TITLE_LEN:
                continue
            variants = set(_title_variants(title))
            if variants & candidates:
                return (title, excerpt)
        return None

    def search(self, user_input: str, debug: bool = False) -> str | None:
        """Search for relevant Wikipedia excerpts.

        1. Extract topic via regex.
        2. Unquoted FTS search — look for title match among results.
        3. Quoted FTS search — if top result differs from title match, include it.
        4. Fall back to top unquoted result if no quoted match.
        Returns up to two articles (title match + best contextual match).
        """
        if not self._conn:
            return None

        topic = extract_topic(user_input)
        if not topic:
            if debug:
                print("[WIKI] no topic extracted")
            return None

        # Resolve alias (e.g. "World War 2" -> "World War II")
        resolved = self._resolve_alias(topic)
        search_topic = resolved if resolved != topic else topic
        if debug:
            if resolved != topic:
                print(f'[WIKI] topic: "{topic}" -> "{resolved}"')
            else:
                print(f'[WIKI] topic: "{topic}"')

        # Step 1: unquoted (broad) search
        unquoted = self._fts_unquoted(search_topic)
        if debug and unquoted:
            print("[WIKI] unquoted:")
            for i, (t, _) in enumerate(unquoted[:3]):
                print(f"[WIKI]   {i+1}. {t}")

        # Step 2: find title match — try full topic first, then stripped, then FTS results
        title_match = self._lookup_by_title(search_topic)
        if not title_match and unquoted:
            title_match = self._find_title_match(unquoted, search_topic)

        # If no match, try stripping qualifiers ("history of X" -> "X")
        if not title_match:
            stripped = strip_qualifiers(search_topic)
            if stripped:
                if debug:
                    print(f"[WIKI] stripped topic: \"{stripped}\"")
                title_match = self._lookup_by_title(stripped)
                if not title_match and unquoted:
                    title_match = self._find_title_match(unquoted, stripped)

        if debug:
            print(f"[WIKI] title match: {title_match[0] if title_match else None}")

        # Step 3: quoted (exact phrase) search
        quoted = self._fts_quoted(search_topic)
        if debug and quoted:
            print("[WIKI] quoted:")
            for i, (t, _) in enumerate(quoted[:3]):
                print(f"[WIKI]   {i+1}. {t}")

        # Pick the single best result: title match > quoted top > unquoted top
        result = None
        if title_match:
            result = title_match
        elif quoted:
            result = quoted[0]
        elif unquoted:
            result = unquoted[0]

        if not result or result[0] in self._retrieved:
            return None

        title, excerpt = result
        self._retrieved.add(title)
        return f'[Reference: {title}]\n{excerpt}'
