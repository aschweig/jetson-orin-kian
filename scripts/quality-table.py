#!/usr/bin/env python3
"""Generate docs/quality-table.tex from qualitative evaluation CSV.

Reads the CSV produced by qualitative-study.py, builds a LaTeX table,
and generates a discussion paragraph via claude -p.

Usage:
    uv run python scripts/quality-table.py
    uv run python scripts/quality-table.py path/to/quality-results.csv
"""

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "quality-results.csv"
LATEX_OUT = PROJECT_ROOT / "docs" / "quality-table.tex"

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

DISCUSSION_PROMPT = """\
You are writing a short discussion section for an academic litepaper about \
on-device voice assistants for children, running on a Jetson Orin Nano.

Below is a qualitative evaluation table. Each row is a small language model \
evaluated across {n} independent multi-turn conversations with a rubric \
scored by {evaluator}. The columns are:
- Overall: holistic conversation quality, 1-5 (higher is better)
- 4th-Wall: count of responses that broke the cartoon-animal persona (lower is better)
- Fact Err Avg / Max: factual errors per conversation on factual prompts (lower is better)
- Example flub: a representative broken-character moment, factual howler, or nonsensical phrase

{table}

Write 1-2 short paragraphs of discussion contrasting the strongest and weakest \
performers. Call out any cases where a model is strong on one axis but weak on \
another (e.g., high Overall but frequent fourth-wall breaks, or low factual \
errors but stilted prose). Note differences between Ollama and llama.cpp (server) \
variants of the same base model if present. Close with a sentence identifying the \
best choices for this use case.

Then add a Benchmark Limitations subsection with content like:

--- Example Limitations subsection ---
\\subsection{{Benchmark Limitations}}

This benchmark is a task-specific internal evaluation, not a general capability \
ranking. The prompt set is small ({n_prompts} prompts) and English-only, covering \
a narrow slice of the topics a child might ask about. Qualitative scores were \
assigned by an LLM judge ({evaluator}) rather than human raters, which may not \
capture all dimensions of child-appropriateness such as tone, pacing, or emotional \
register. The benchmark also does not evaluate the content safety pipeline or the \
LaTeX-to-speech system. Results should be interpreted as a comparative signal for \
model selection on this hardware, not as absolute quality claims.
--- End example ---

Output ONLY the LaTeX body text (no \\begin{{document}}). \
Use \\textbf{{Model Name}} when first mentioning a model. Do not use \
markdown. Do not wrap in a code block. Do not use thinking tags."""


def load_csv(path: Path) -> tuple[dict, dict, str]:
    """Load CSV and return (scores_by_engine, best_flubs, evaluator).

    scores_by_engine: {engine: [{"overall": int, "fourth_wall": int, ...}, ...]}
    best_flubs: {engine: str}
    evaluator: str
    """
    scores = defaultdict(list)
    best_flubs = {}
    evaluator = "unknown"

    with open(path) as f:
        for row in csv.DictReader(f):
            engine = row["Engine"]
            scores[engine].append({
                "overall": int(row["Overall"]),
                "fourth_wall": int(row["FourthWall"]),
                "factual_errors": int(row["FactualErrors"]),
                "flub": row["Flub"],
            })
            if row.get("BestFlub"):
                best_flubs[engine] = row["BestFlub"]
            if row.get("Evaluator") and row["Evaluator"] != "unknown":
                evaluator = row["Evaluator"]

    return dict(scores), best_flubs, evaluator


def make_latex(scores: dict, best_flubs: dict, evaluator: str) -> str:
    """Generate a LaTeX fragment: quality table + discussion paragraph."""
    # Count prompts from benchmark file
    prompts_file = PROJECT_ROOT / "scripts" / "benchmark-prompts.txt"
    n_prompts = len([l for l in prompts_file.read_text().strip().splitlines() if l.strip()]) if prompts_file.exists() else "?"

    # Sort by mean overall descending, then factual errors ascending as tiebreaker
    engines_sorted = sorted(
        scores.keys(),
        key=lambda e: (
            -sum(d["overall"] for d in scores[e]) / len(scores[e]),
            sum(d["factual_errors"] for d in scores[e]) / len(scores[e]),
        ),
    )

    # --- Scores table (no flubs) ---
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l|r|r|r|r}",
        r"\toprule",
        r"Engine & Overall & 4th-Wall & Fact Err Avg & Fact Err Max \\",
        r"\midrule",
    ]

    plain_rows = []
    flub_rows = []
    n_sample = None
    for engine in engines_sorted:
        name = DISPLAY_NAMES.get(engine, engine)
        s = scores[engine]
        n = len(s)
        if n_sample is None:
            n_sample = n
        overall_avg = sum(d["overall"] for d in s) / n
        fw_avg = sum(d["fourth_wall"] for d in s) / n
        fact_avg = sum(d["factual_errors"] for d in s) / n
        fact_max = max(d["factual_errors"] for d in s)
        flub = best_flubs.get(engine, "")

        # Plain-text row for LLM prompt (ASCII-only flub)
        flub_ascii = "".join(c if ord(c) < 128 else "*" for c in flub)
        flub_plain = f'"{flub_ascii}"' if flub_ascii else "---"
        plain_rows.append(
            f"{name:<35} {overall_avg:>7.1f} {fw_avg:>8.1f} "
            f"{fact_avg:>7.1f} {fact_max:>7}  {flub_plain}"
        )

        # Scores row
        name_tex = name.replace("_", r"\_")
        lines.append(
            f"{name_tex} & {overall_avg:.1f} & {fw_avg:.1f} & "
            f"{fact_avg:.1f} & {fact_max} \\\\"
        )

        # Flub row (for second table) — strip non-ASCII to avoid pdflatex errors
        flub_tex = "".join(c if ord(c) < 128 else "*" for c in flub)
        flub_tex = flub_tex.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        if flub_tex:
            flub_tex = r"\textit{``" + flub_tex + r"''}"
        else:
            flub_tex = "---"
        flub_rows.append(f"{name_tex} & {flub_tex} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Attribution footnote and caption
    evaluator_tex = evaluator.replace("_", r"\_")
    lines.append(
        r"\caption{Qualitative response quality across "
        + f"{n_sample} benchmark runs ({n_prompts} prompts each). "
        + r"Overall is a 1--5 holistic score (higher is better); "
        + r"4th-Wall counts responses breaking the cartoon-animal persona; "
        + r"Fact Err counts factual errors on factual prompts (lower is better). "
        + "Evaluated by " + evaluator_tex + ".}"
    )
    lines.append(r"\label{tab:quality}")
    lines.append(r"\end{table}")

    # --- Flubs table ---
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|l}")
    lines.append(r"\toprule")
    lines.append(r"Engine & Representative flub \\")
    lines.append(r"\midrule")
    for row in flub_rows:
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Representative factual errors from qualitative evaluation.}")
    lines.append(r"\label{tab:flubs}")
    lines.append(r"\end{table}")

    # --- Discussion (outside table environment) ---
    # Generate discussion via claude -p
    plain_header = (
        f"{'Engine':<35} {'Overall':>7} {'4th-Wall':>8} "
        f"{'FactAvg':>7} {'FactMax':>7}  Example flub"
    )
    plain_table = plain_header + "\n" + "\n".join(plain_rows)
    prompt = DISCUSSION_PROMPT.format(
        n=n_sample, n_prompts=n_prompts, evaluator=evaluator, table=plain_table,
    )

    print("  Generating discussion via claude -p ...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "claude-opus-4-6"],
            input=prompt,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Strip thinking tags and non-ASCII to avoid pdflatex errors
            import re
            discussion = re.sub(r"<think>.*?</think>\s*", "", result.stdout.strip(), flags=re.DOTALL)
            discussion = "".join(c if ord(c) < 128 else "*" for c in discussion)
        else:
            print("    WARNING: claude -p failed, using placeholder", file=sys.stderr)
            discussion = r"\textit{Discussion to be written.}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    WARNING: {e}, using placeholder", file=sys.stderr)
        discussion = r"\textit{Discussion to be written.}"

    lines.append("")
    lines.append(discussion)

    return "\n".join(lines)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    scores, best_flubs, evaluator = load_csv(csv_path)

    print(f"Loaded {sum(len(v) for v in scores.values())} rows "
          f"across {len(scores)} engines from {csv_path}")
    print(f"Evaluator: {evaluator}")

    tex = make_latex(scores, best_flubs, evaluator)
    LATEX_OUT.write_text(tex + "\n")
    print(f"\nWrote {LATEX_OUT}")


if __name__ == "__main__":
    main()
