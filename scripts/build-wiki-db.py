#!/usr/bin/env python3
"""Download Simple Wikipedia dump and build a SQLite + FTS5 database.

Usage:
    uv run python scripts/build-wiki-db.py

Downloads simplewiki-latest-pages-articles-multistream.xml.bz2 (~250MB),
extracts article titles and first ~150 words, and stores them in
models/simplewiki.db with an FTS5 index for fast full-text search.

Filters:
  - Titles must be >= 5 characters
  - Articles must have >= 80 words after cleanup
  - Only main namespace articles (ns=0), no redirects
  - Excerpts capped at 150 words
"""

import bz2
import html
import re
import sqlite3
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMP_PATH = PROJECT_ROOT / "models" / "simplewiki-latest-pages-articles-multistream.xml.bz2"
DB_PATH = PROJECT_ROOT / "models" / "simplewiki.db"

MIN_TITLE_LEN = 5
MIN_WORDS = 80
MAX_WORDS = 150


def download_dump():
    """Download the dump if not already present."""
    if DUMP_PATH.exists():
        size_mb = DUMP_PATH.stat().st_size / 1024 / 1024
        print(f"Dump already exists: {DUMP_PATH} ({size_mb:.0f} MB)")
        return
    print(f"Downloading {DUMP_URL} ...")
    print(f"  -> {DUMP_PATH}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded * 100 / total_size
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(DUMP_URL, DUMP_PATH, reporthook=_progress)
    print()
    print("Download complete.")


# --- Wikitext cleanup ---

# Remove templates {{...}}, including nested
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
# [[Link|Display]] -> Display, [[Link]] -> Link
_WIKILINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
# External links [http://... text] -> text
_EXTLINK_RE = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_EXTLINK_BARE_RE = re.compile(r"\[https?://\S+\]")
# HTML tags
_HTML_RE = re.compile(r"<[^>]+>")
# Section headers == Title ==
_HEADER_RE = re.compile(r"^=+\s*.*?\s*=+\s*$", re.MULTILINE)
# Ref tags and their content
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/?>", re.DOTALL)
# Bold/italic markup
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
# File/Image/Category links
_FILE_RE = re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.IGNORECASE)
# Tables
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
# Magic words / behavior switches
_MAGIC_RE = re.compile(r"__[A-Z]+__")


def clean_wikitext(text: str) -> str:
    """Strip wikitext markup, returning plain text."""
    # Remove tables first (can be large)
    text = _TABLE_RE.sub("", text)
    # Remove file/image/category links
    text = _FILE_RE.sub("", text)
    # Remove refs
    text = _REF_RE.sub("", text)
    # Remove templates (multiple passes for nesting)
    for _ in range(5):
        new = _TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new
    # Remove HTML
    text = _HTML_RE.sub("", text)
    # Remove headers
    text = _HEADER_RE.sub("", text)
    # Convert wikilinks
    text = _WIKILINK_RE.sub(r"\1", text)
    # Convert external links
    text = _EXTLINK_RE.sub(r"\1", text)
    text = _EXTLINK_BARE_RE.sub("", text)
    # Remove bold/italic
    text = _BOLD_ITALIC_RE.sub("", text)
    # Remove magic words
    text = _MAGIC_RE.sub("", text)
    # Decode HTML entities: &nbsp; -> space, &amp; -> &, &#123; -> {, etc.
    text = html.unescape(text)
    # Clean up any remaining wiki markup artifacts
    text = text.replace("]]", "").replace("[[", "")
    # Collapse whitespace (including non-breaking spaces from &nbsp;)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[\s]+", " ", text)
    return text.strip()


def extract_excerpt(text: str) -> str | None:
    """Extract first ~150 words from cleaned article text.

    Truncates at the last sentence boundary (period) within the word limit.
    Returns None if the article has fewer than MIN_WORDS words.
    """
    cleaned = clean_wikitext(text)
    words = cleaned.split()
    if len(words) < MIN_WORDS:
        return None
    excerpt = " ".join(words[:MAX_WORDS])
    # Truncate at the last period to end on a complete sentence
    last_period = excerpt.rfind(".")
    if last_period > 0:
        excerpt = excerpt[:last_period + 1]
    return excerpt


def iter_articles(path: Path):
    """Yield (title, text) from a MediaWiki XML dump (bz2 compressed)."""
    # Namespace for MediaWiki export schema
    ns = None
    with bz2.open(path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            # Detect namespace from first element
            if ns is None and "}" in elem.tag:
                ns = elem.tag.split("}")[0] + "}"

            if elem.tag != f"{ns}page":
                continue

            # Only main namespace
            ns_elem = elem.find(f"{ns}ns")
            if ns_elem is None or ns_elem.text != "0":
                elem.clear()
                continue

            # Yield redirects as (from_title, None, target_title)
            redirect = elem.find(f"{ns}redirect")
            if redirect is not None:
                title_elem = elem.find(f"{ns}title")
                if title_elem is not None:
                    target = redirect.get("title", "")
                    if target:
                        yield (title_elem.text.strip(), None, target)
                elem.clear()
                continue

            title_elem = elem.find(f"{ns}title")
            revision = elem.find(f"{ns}revision")
            if title_elem is None or revision is None:
                elem.clear()
                continue

            text_elem = revision.find(f"{ns}text")
            if text_elem is None or not text_elem.text:
                elem.clear()
                continue

            title = title_elem.text.strip()
            text = text_elem.text

            # Free memory
            elem.clear()

            yield title, text, None  # (title, text, None) for articles


def build_db():
    """Parse dump and build SQLite + FTS5 database."""
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            excerpt TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE aliases (
            alias TEXT NOT NULL COLLATE NOCASE,
            title TEXT NOT NULL,
            PRIMARY KEY (alias)
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE articles_fts USING fts5(
            title, excerpt, content=articles, content_rowid=id
        )
    """)
    # Triggers to keep FTS in sync
    conn.executescript("""
        CREATE TRIGGER articles_ai AFTER INSERT ON articles BEGIN
            INSERT INTO articles_fts(rowid, title, excerpt)
            VALUES (new.id, new.title, new.excerpt);
        END;
    """)

    total = 0
    inserted = 0
    redirects = 0
    skipped_title = 0
    skipped_short = 0

    print(f"Parsing {DUMP_PATH} ...")
    for title, text, redirect_target in iter_articles(DUMP_PATH):
        total += 1

        # Redirect: store as alias
        if text is None and redirect_target:
            if len(title) >= MIN_TITLE_LEN:
                conn.execute(
                    "INSERT OR IGNORE INTO aliases (alias, title) VALUES (?, ?)",
                    (title, redirect_target),
                )
                redirects += 1
            continue

        if len(title) < MIN_TITLE_LEN:
            skipped_title += 1
            continue

        excerpt = extract_excerpt(text)
        if excerpt is None:
            skipped_short += 1
            continue

        conn.execute(
            "INSERT INTO articles (title, excerpt) VALUES (?, ?)",
            (title, excerpt),
        )
        inserted += 1

        if inserted % 10000 == 0:
            conn.commit()
            print(f"  {inserted:,} articles indexed ({total:,} processed)")

    conn.commit()

    # Optimize FTS index
    print("Optimizing FTS index...")
    conn.execute("INSERT INTO articles_fts(articles_fts) VALUES ('optimize')")
    conn.commit()
    conn.close()

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\nDone.")
    print(f"  Total pages scanned: {total:,}")
    print(f"  Articles indexed:    {inserted:,}")
    print(f"  Aliases (redirects): {redirects:,}")
    print(f"  Skipped (short title < {MIN_TITLE_LEN} chars): {skipped_title:,}")
    print(f"  Skipped (< {MIN_WORDS} words):  {skipped_short:,}")
    print(f"  Database size:       {size_mb:.1f} MB")
    print(f"  Path:                {DB_PATH}")


def main():
    download_dump()
    build_db()


if __name__ == "__main__":
    main()
