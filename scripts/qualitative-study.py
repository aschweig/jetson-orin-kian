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

# Cache-warming probe prompts — excluded from qualitative evaluation because
# their responses are capped at 1 token by the benchmark runner.
SINGLE_TOKEN_PROMPTS = {"Ummm", "Okay"}

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
You are evaluating an AI assistant (a cartoon-animal persona named Kian) \
talking to a young child (grade 3). You will be given 20 prompt-response \
pairs from one conversation.

Kian should stay in character as a cartoon animal. Breaking character \
includes any of:
- Mentioning being an AI, language model, assistant, chatbot, or software
- Using the phrase "cartoon animal" or otherwise naming the persona type
- Referring to its training, knowledge cutoff, or model
- Assuming context that wasn't provided (e.g., "in your science class", \
"your teacher said", "on the worksheet")

Distinguish imaginative from factual prompts yourself:
- Imaginative: stories, game suggestions, preference questions like \
"do you like turtles?" — factual errors don't apply.
- Factual: science questions, comparisons, explanations, historical or \
mythological questions — factual errors matter.

The spelling "cheetae" for "cheetah" is acceptable, not an error.

Answer with ONLY a number or short string for each:

OVERALL: Score the conversation from 1 (poor) to 5 (excellent), weighing \
naturalness, engagement, age-appropriateness, and freedom from the issues \
below.

FOURTH_WALL_BREAKS: Count of responses that broke the cartoon-animal \
persona. 0 if none.

FACTUAL_ERRORS: Count of responses on factual prompts that contained \
clear factual errors. Skip imaginative prompts. 0 if none.

FLUB: The single most notable flub from the conversation — a broken- \
character moment, a factual howler, or a nonsensical phrase. Quoted \
verbatim from the response, max 10 words. If nothing noteworthy, "none".

Reply in EXACTLY this format, nothing else:
OVERALL: <1-5>
FOURTH_WALL_BREAKS: <number>
FACTUAL_ERRORS: <number>
FLUB: <quote or none>"""


def strip_dummy_prompts(conversation: str) -> str:
    """Remove prompt blocks whose User: line matches a cache-warming probe."""
    blocks = re.split(r"(?=^--- Prompt \d+ ---$)", conversation, flags=re.MULTILINE)
    kept = []
    for block in blocks:
        m = re.search(r"^User: (.+)$", block, flags=re.MULTILINE)
        if m and m.group(1).strip() in SINGLE_TOKEN_PROMPTS:
            continue
        kept.append(block)
    return "".join(kept)


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
            ["claude", "-p", "--model", "claude-opus-4-6"],
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

    overall = re.search(r"OVERALL:\s*(\d+)", response)
    fourth_wall = re.search(r"FOURTH_WALL_BREAKS:\s*(\d+)", response)
    factual = re.search(r"FACTUAL_ERRORS:\s*(\d+)", response)
    flub = re.search(r"FLUB:\s*(.+)", response)

    if not (overall and fourth_wall and factual):
        print(f"    WARNING: could not parse response:", file=sys.stderr)
        print(f"    {response}", file=sys.stderr)
        return None

    flub_text = flub.group(1).strip().strip('"') if flub else "none"

    return {
        "overall": int(overall.group(1)),
        "fourth_wall": int(fourth_wall.group(1)),
        "factual_errors": int(factual.group(1)),
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
            # Strip cache-warming probe prompts before evaluation
            conversation = strip_dummy_prompts(conversation)
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
                    f"    => overall={result['overall']} "
                    f"4th-wall={result['fourth_wall']} "
                    f"factual={result['factual_errors']}{flub_str}",
                    file=sys.stderr,
                )

    if not scores:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    # Identify the evaluator model
    try:
        model_result = subprocess.run(
            ["claude", "-p", "--model", "claude-opus-4-6", "Reply with ONLY your model name and version, nothing else."],
            capture_output=True, text=True, timeout=30,
        )
        evaluator = model_result.stdout.strip() if model_result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        evaluator = "unknown"

    print(f"\nEvaluator: {evaluator}")

    # Sort by mean overall descending, then factual errors ascending as tiebreaker
    engines_sorted = sorted(
        scores.keys(),
        key=lambda e: (
            -sum(s["overall"] for _, s in scores[e]) / len(scores[e]),
            sum(s["factual_errors"] for _, s in scores[e]) / len(scores[e]),
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
                ["claude", "-p", "--model", "claude-opus-4-6"],
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
        f"{'Engine':<35} {'Overall':>8} {'4th-Wall':>9} "
        f"{'Fact Err':>9} {'Fact Max':>9}  n  Example flub"
    )
    print("-" * 115)
    for engine in engines_sorted:
        name = DISPLAY_NAMES.get(engine, engine)
        s = [d for _, d in scores[engine]]
        n = len(s)
        overall_avg = sum(d["overall"] for d in s) / n
        fw_avg = sum(d["fourth_wall"] for d in s) / n
        fact_avg = sum(d["factual_errors"] for d in s) / n
        fact_max = max(d["factual_errors"] for d in s)
        flub = best_flubs.get(engine, "")
        flub_str = f'  "{flub}"' if flub else ""
        print(
            f"{name:<35} {overall_avg:>8.1f} {fw_avg:>9.1f} "
            f"{fact_avg:>9.1f} {fact_max:>9}  {n}{flub_str}"
        )

    # Always write CSV
    csv_path = Path(args.csv)
    with open(csv_path, "w") as f:
        f.write("Engine,Run,Overall,FourthWall,FactualErrors,Flub,BestFlub,Evaluator\n")
        for engine in engines_sorted:
            best = best_flubs.get(engine, "")
            for run_name, d in scores[engine]:
                flub_escaped = d["flub"].replace('"', '""')
                best_escaped = best.replace('"', '""')
                f.write(
                    f"{engine},{run_name},{d['overall']},{d['fourth_wall']},"
                    f"{d['factual_errors']},\"{flub_escaped}\","
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
