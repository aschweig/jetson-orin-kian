#!/usr/bin/env python3
"""Qualitative evaluation of benchmark conversations using claude -p.

Extracts per-engine conversations from benchmark logs, feeds each to
claude -p with a rubric, aggregates scores, and writes a CSV.

Usage:
    uv run python scripts/qualitative-study.py benchmark-logs/20260408*.log
    uv run python scripts/qualitative-study.py benchmark-logs/*.log
    uv run python scripts/qualitative-study.py benchmark-logs/*.log --csv results.csv
"""

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "quality-results.csv"

DISPLAY_NAMES = {
    "llamacpp:Qwen3.5-2B-Q4_K_M": "llamacpp:Qwen3.5-2B",
    "llamacpp:granite-3.3-2b-instruct-Q4_K_M": "llamacpp:Granite 3.3-2B",
    "ollama:qwen3.5:2b-q4_K_M": "ollama:Qwen3.5-2B",
    "ollama:qwen3:4b-q4_K_M": "ollama:Qwen3-4B",
    "ollama:qwen3.5:4b-q4_K_M": "ollama:Qwen3.5-4B",
    "ollama:llama3.2:3b-instruct-q4_K_M": "ollama:Llama 3.2-3B",
    "ollama:ministral-3:3b": "ollama:Ministral-3 3B",
    "ollama:granite3.3:2b": "ollama:Granite 3.3-2B",
    "ollama:granite4:3b": "ollama:Granite 4-3B",
    "ollama:nemotron-3-nano:4b": "ollama:Nemotron-3 Nano 4B",
    "llamacpp:ibm-granite_granite-4.0-micro-IQ4_XS": "llamacpp:Granite 4.0 Micro IQ4",
    "llamacpp:granite-4.0-micro-Q4_K_M": "llamacpp:Granite 4.0 Micro",
    "llamacpp:granite-4.0-h-micro-Q4_K_M": "llamacpp:Granite 4.0 H-Micro",
    "llamacpp:qwen3-4b-instruct-2507-q4_k_m": "llamacpp:Qwen3-4B",
    "llamacpp:HuggingFaceTB_SmolLM3-3B-Q4_K_M": "llamacpp:SmolLM3-3B",
}

RUBRIC = """\
You are evaluating an AI assistant talking to a young child (grade 3).
You will be given 8 prompt-response pairs from one conversation.

Important notes:
- The spelling "cheetae" for "cheetah" is acceptable, not an error.
- In prompt 8 (storytelling), factual liberties are fine — animals can \
talk, race each other, have human attributes, etc. Only count errors \
in prompts 2-7 where factual accuracy matters.

Answer these questions with ONLY a number for each:

ERROR_COUNT: How many of prompts 2-7 had an answer that included factual \
errors or broke the 4th wall (e.g., mentioning "your science class", \
referencing being an AI, or saying things that do not make sense)? \
Do not count prompts 1 or 8.

CHATTY_COUNT: How many of prompts 1, 2, 3, 4, and 7 had answers that \
were more than a sentence or two, OR that used unnecessarily repetitive \
phrases (e.g., starting every answer with "Great question!" or "That's \
a great question!", repeating the same filler across responses, or \
echoing the same phrase multiple times within a single response)? \
(These should be short, concise, and natural-sounding responses.)

POOR_PROSE_COUNT: Of prompts 5, 6, and 8: prompt 5 should explain well \
why animals differ in speed, prompt 6 should explain how muscles work in \
a way a child can understand, and prompt 8 should be an entertaining \
short story for a child. How many of these 3 had poor prose — weak \
explanations, repetitive or formulaic structure, or language that would \
not engage a child? (0-3, where 0 means all three were good)

FLUB: From prompts 2-7, quote the single worst flub — something clearly \
wrong, not just imprecise. Real flubs include: stating a completely \
wrong animal or speed (e.g., "tuna swim 150 mph"), nonsensical phrases \
(e.g., "muscles get hot and glow"), ungrammatical sentences, or \
breaking the 4th wall by referencing being an AI or cartoon (e.g., \
"I am only a cartoon animal so"). NOT flubs: imprecise but defensible \
statements, encouraging phrases like "great question for your science \
class", or mild simplifications appropriate for a child. \
Max 10 words, quoted verbatim from the response. If there are none, write "none".

Reply in EXACTLY this format, nothing else:
ERROR_COUNT: <number>
CHATTY_COUNT: <number>
POOR_PROSE_COUNT: <number>
FLUB: <quote or none>"""


def extract_engines(log_path: Path) -> dict[str, str]:
    """Parse a log file and return {engine_name: conversation_text}."""
    text = log_path.read_text()
    engines = {}
    # Find all "Engine: <name>" lines
    parts = re.split(r"^Engine: (.+)$", text, flags=re.MULTILINE)
    # parts = [preamble, engine1, content1, engine2, content2, ...]
    for i in range(1, len(parts) - 1, 2):
        engine = parts[i].strip()
        content = parts[i + 1]
        # Content starts with "\n===\n" (closing separator) — skip it,
        # then take everything up to the next === (next engine's opener)
        chunks = re.split(r"^={10,}$", content, flags=re.MULTILINE)
        # chunks[0] is empty/whitespace, chunks[1] has the conversation
        conversation = chunks[1].strip() if len(chunks) > 1 else ""
        if conversation:
            engines[engine] = conversation
    return engines


def evaluate(conversation: str) -> dict | None:
    """Call claude -p with the rubric and parse the response."""
    stdin_text = RUBRIC + "\n\n--- CONVERSATION ---\n" + conversation
    try:
        result = subprocess.run(
            ["claude", "-p"],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    ERROR: {e}", file=sys.stderr)
        return None

    response = result.stdout.strip()
    if result.returncode != 0:
        print(f"    ERROR: claude exited {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"    {result.stderr.strip()}", file=sys.stderr)
        return None

    errors = re.search(r"ERROR_COUNT:\s*(\d+)", response)
    chatty = re.search(r"CHATTY_COUNT:\s*(\d+)", response)
    poor_prose = re.search(r"POOR_PROSE_COUNT:\s*(\d+)", response)
    flub = re.search(r"FLUB:\s*(.+)", response)

    if not (errors and chatty and poor_prose):
        print(f"    WARNING: could not parse response:", file=sys.stderr)
        print(f"    {response}", file=sys.stderr)
        return None

    flub_text = flub.group(1).strip().strip('"') if flub else "none"

    return {
        "errors": int(errors.group(1)),
        "chatty": int(chatty.group(1)),
        "poor_prose": int(poor_prose.group(1)),
        "flub": flub_text,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qualitative evaluation of benchmark conversations")
    parser.add_argument("logs", nargs="+", help="Benchmark log files")
    parser.add_argument("--csv", metavar="FILE", default=str(DEFAULT_CSV),
                        help=f"Write per-run scores to CSV (default: {DEFAULT_CSV.name})")
    parser.add_argument("--no-tex", action="store_true",
                        help="Skip running quality-table.py after evaluation")
    args = parser.parse_args()

    log_files = [Path(p) for p in args.logs]
    for p in log_files:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)

    # Collect all results: {engine: [(run_name, score_dict)]}
    scores = defaultdict(list)

    for log_path in sorted(log_files):
        if not log_path.exists():
            continue
        run_name = log_path.stem
        print(f"\nProcessing {log_path.name} ...", file=sys.stderr)
        engines = extract_engines(log_path)

        for engine, conversation in engines.items():
            name = DISPLAY_NAMES.get(engine, engine)
            # Show first line of each prompt response
            print(f"\n  {name}:", file=sys.stderr)
            for line in conversation.splitlines():
                if line.startswith("Assistant: "):
                    preview = line[11:]
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    print(f"    > {preview}", file=sys.stderr)

            result = evaluate(conversation)
            if result:
                scores[engine].append((run_name, result))
                flub_str = f' flub="{result["flub"]}"' if result["flub"] != "none" else ""
                print(
                    f"    => errors={result['errors']} "
                    f"chatty={result['chatty']} "
                    f"poor_prose={result['poor_prose']}{flub_str}",
                    file=sys.stderr,
                )

    if not scores:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    # Identify the evaluator model
    try:
        model_result = subprocess.run(
            ["claude", "-p", "Reply with ONLY your model name and version, nothing else."],
            capture_output=True, text=True, timeout=30,
        )
        evaluator = model_result.stdout.strip() if model_result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        evaluator = "unknown"

    print(f"\nEvaluator: {evaluator}")

    # Sort by mean errors ascending, then prose descending
    engines_sorted = sorted(
        scores.keys(),
        key=lambda e: (
            sum(s["errors"] for _, s in scores[e]) / len(scores[e]),
            sum(s["poor_prose"] for _, s in scores[e]) / len(scores[e]),
        ),
    )

    # Pick the best example flub per engine
    best_flubs = {}
    for engine in engines_sorted:
        flubs = [d["flub"] for _, d in scores[engine] if d["flub"] != "none"]
        if not flubs:
            best_flubs[engine] = ""
            continue
        if len(flubs) == 1:
            best_flubs[engine] = flubs[0]
            continue
        # Ask claude to pick the most illustrative one
        flub_list = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(flubs))
        name = DISPLAY_NAMES.get(engine, engine)
        print(f"  Picking best flub for {name} ...", file=sys.stderr)
        try:
            pick = subprocess.run(
                ["claude", "-p"],
                input=(
                    f"These are flubs from an AI assistant called {name}. "
                    f"Pick the single most illustrative or amusing one. "
                    f"Reply with ONLY the quoted flub, nothing else.\n\n{flub_list}"
                ),
                capture_output=True, text=True, timeout=60,
            )
            best_flubs[engine] = pick.stdout.strip().strip('"') if pick.returncode == 0 else flubs[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            best_flubs[engine] = flubs[0]

    # List all flubs by engine
    print("\n=== All Flubs ===")
    for engine in engines_sorted:
        name = DISPLAY_NAMES.get(engine, engine)
        flubs = [(run, d["flub"]) for run, d in scores[engine] if d["flub"] != "none"]
        if flubs:
            print(f"\n  {name}:")
            for run, flub in flubs:
                print(f"    [{run}] \"{flub}\"")
        else:
            print(f"\n  {name}: (none)")

    # Print summary table
    print()
    print(
        f"{'Engine':<35} {'Err Avg':>8} {'Err Max':>8} "
        f"{'Chatty':>7} {'Poor Prose':>11}  n  Example flub"
    )
    print("-" * 110)
    for engine in engines_sorted:
        name = DISPLAY_NAMES.get(engine, engine)
        s = [d for _, d in scores[engine]]
        n = len(s)
        err_avg = sum(d["errors"] for d in s) / n
        err_max = max(d["errors"] for d in s)
        chatty_avg = sum(d["chatty"] for d in s) / n
        poor_prose_avg = sum(d["poor_prose"] for d in s) / n
        flub = best_flubs.get(engine, "")
        flub_str = f'  "{flub}"' if flub else ""
        print(
            f"{name:<35} {err_avg:>8.1f} {err_max:>8} "
            f"{chatty_avg:>7.1f} {poor_prose_avg:>11.1f}  {n}{flub_str}"
        )

    # Always write CSV
    csv_path = Path(args.csv)
    with open(csv_path, "w") as f:
        f.write("Engine,Run,Errors,Chatty,PoorProse,Flub,BestFlub,Evaluator\n")
        for engine in engines_sorted:
            best = best_flubs.get(engine, "")
            for run_name, d in scores[engine]:
                flub_escaped = d["flub"].replace('"', '""')
                best_escaped = best.replace('"', '""')
                f.write(
                    f"{engine},{run_name},{d['errors']},{d['chatty']},"
                    f"{d['poor_prose']},\"{flub_escaped}\","
                    f"\"{best_escaped}\",\"{evaluator}\"\n"
                )
    print(f"\nSaved to {csv_path}")

    # Generate LaTeX table
    if not args.no_tex:
        print("\nGenerating quality table ...", file=sys.stderr)
        script = Path(__file__).resolve().parent / "quality-table.py"
        subprocess.run([sys.executable, str(script), str(csv_path)], check=False)


if __name__ == "__main__":
    main()
