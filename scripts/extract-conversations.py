#!/usr/bin/env python3
"""Extract per-engine conversations from benchmark log files.

Parses a benchmark log and yields one block per engine, containing all
prompt/response pairs for that engine across all runs in the file.

Usage:
    uv run python scripts/extract-conversations.py benchmark-logs/20260407-004916.log
    uv run python scripts/extract-conversations.py benchmark-logs/*.log
"""

import re
import sys
from pathlib import Path

ENGINE_RE = re.compile(r"^={10,}$")
ENGINE_NAME_RE = re.compile(r"^Engine:\s+(.+)$")
PROMPT_RE = re.compile(r"^--- Prompt (\d+) ---$")
STATS_RE = re.compile(r"^\s+TTFT=")


def extract_engines(log_path: Path) -> dict[str, list[dict]]:
    """Parse a benchmark log file, returning {engine: [conversations]}.

    Each conversation is a dict with keys:
        engine, prompts: [{user, assistant, stats}]
    """
    engines: dict[str, list[dict]] = {}
    lines = log_path.read_text().splitlines()

    i = 0
    while i < len(lines):
        # Look for engine header: === / Engine: name / ===
        if ENGINE_RE.match(lines[i]):
            if i + 2 < len(lines) and ENGINE_NAME_RE.match(lines[i + 1]):
                engine = ENGINE_NAME_RE.match(lines[i + 1]).group(1)
                i += 3  # skip past closing ===

                # Parse prompts for this engine block
                conversation = {"engine": engine, "prompts": []}
                while i < len(lines) and not ENGINE_RE.match(lines[i]):
                    m = PROMPT_RE.match(lines[i])
                    if m:
                        i += 1
                        # User line
                        user = ""
                        if i < len(lines) and lines[i].startswith("User: "):
                            user = lines[i][6:]
                            i += 1
                        # Assistant line(s)
                        assistant_lines = []
                        if i < len(lines) and lines[i].startswith("Assistant: "):
                            assistant_lines.append(lines[i][11:])
                            i += 1
                            # Multi-line responses
                            while i < len(lines) and not STATS_RE.match(lines[i]) and not PROMPT_RE.match(lines[i]) and not ENGINE_RE.match(lines[i]) and lines[i].strip():
                                assistant_lines.append(lines[i])
                                i += 1
                        assistant = "\n".join(assistant_lines)
                        # Stats line
                        stats = ""
                        if i < len(lines) and STATS_RE.match(lines[i]):
                            stats = lines[i].strip()
                            i += 1
                        conversation["prompts"].append({
                            "user": user,
                            "assistant": assistant,
                            "stats": stats,
                        })
                    else:
                        i += 1

                if engine not in engines:
                    engines[engine] = []
                engines[engine].append(conversation)
                continue
        i += 1

    return engines


def format_conversation(engine: str, conversations: list[dict]) -> str:
    """Format all runs for one engine as a readable block."""
    parts = [f"Engine: {engine}", f"Runs: {len(conversations)}", ""]
    for run_idx, conv in enumerate(conversations, 1):
        parts.append(f"--- Run {run_idx} ---")
        for p in conv["prompts"]:
            parts.append(f"User: {p['user']}")
            parts.append(f"Assistant: {p['assistant']}")
            if p["stats"]:
                parts.append(f"  {p['stats']}")
            parts.append("")
    return "\n".join(parts)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log-file> [log-file ...]", file=sys.stderr)
        sys.exit(1)

    # Aggregate across all input files
    all_engines: dict[str, list[dict]] = {}
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        engines = extract_engines(path)
        for engine, convos in engines.items():
            if engine not in all_engines:
                all_engines[engine] = []
            all_engines[engine].extend(convos)

    # Output each engine separately
    for engine in sorted(all_engines):
        convos = all_engines[engine]
        print("=" * 60)
        print(format_conversation(engine, convos))
        print()


if __name__ == "__main__":
    main()
