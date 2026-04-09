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
- Err Avg / Err Max: factual errors per conversation (lower is better)
- Chatty: how many prompts were overly verbose or used repetitive phrases (lower is better, max 5)
- Poor Prose: how many of the 3 longer-form responses had weak prose (lower is better, max 3)
- Example flub: a representative factual error

{table}

Write 1-2 short paragraphs (suitable for a \\subsection in a LaTeX litepaper) \
discussing conclusions from this table. Highlight the IBM Granite and Qwen3 \
model families, and Meta Llama, as contenders for on-device voice assistance. \
Note trade-offs (accuracy vs verbosity, model size, etc). Be concise and \
data-driven — cite numbers from the table.

Output ONLY the LaTeX body text (no \\subsection, no \\begin{{document}}). \
Use \\textbf{{Model Name}} when first mentioning a model. Do not use \
markdown. Do not wrap in a code block."""


def load_csv(path: Path) -> tuple[dict, dict, str]:
    """Load CSV and return (scores_by_engine, best_flubs, evaluator).

    scores_by_engine: {engine: [{"errors": int, "chatty": int, ...}, ...]}
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
                "errors": int(row["Errors"]),
                "chatty": int(row["Chatty"]),
                "poor_prose": int(row["PoorProse"]),
                "flub": row["Flub"],
            })
            if row.get("BestFlub"):
                best_flubs[engine] = row["BestFlub"]
            if row.get("Evaluator") and row["Evaluator"] != "unknown":
                evaluator = row["Evaluator"]

    return dict(scores), best_flubs, evaluator


def make_latex(scores: dict, best_flubs: dict, evaluator: str) -> str:
    """Generate a LaTeX fragment: quality table + discussion paragraph."""
    # Sort by mean errors ascending, then prose descending
    engines_sorted = sorted(
        scores.keys(),
        key=lambda e: (
            sum(d["errors"] for d in scores[e]) / len(scores[e]),
            sum(d["poor_prose"] for d in scores[e]) / len(scores[e]),
        ),
    )

    lines = [
        r"\begin{tabular}{l|r|r|r|r|l}",
        r"\toprule",
        r"Engine & Err Avg & Err Max & Chatty & Poor Prose & Example flub \\",
        r"\midrule",
    ]

    plain_rows = []
    n_sample = None
    for engine in engines_sorted:
        name = DISPLAY_NAMES.get(engine, engine)
        s = scores[engine]
        n = len(s)
        if n_sample is None:
            n_sample = n
        err_avg = sum(d["errors"] for d in s) / n
        err_max = max(d["errors"] for d in s)
        chatty_avg = sum(d["chatty"] for d in s) / n
        poor_prose_avg = sum(d["poor_prose"] for d in s) / n
        flub = best_flubs.get(engine, "")

        # Plain-text row for LLM prompt
        if flub:
            plain_rows.append(
                f"{name:<35} {err_avg:>5.1f} {err_max:>5} "
                f"{chatty_avg:>6.1f} {poor_prose_avg:>5.1f}  "
                f'"{flub}"'
            )
        else:
            plain_rows.append(
                f"{name:<35} {err_avg:>5.1f} {err_max:>5} "
                f"{chatty_avg:>6.1f} {poor_prose_avg:>5.1f}  ---"
            )

        # LaTeX row
        flub_tex = flub.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        if flub_tex:
            flub_tex = r"\textit{``" + flub_tex + r"''}"
        else:
            flub_tex = "---"
        name_tex = name.replace("_", r"\_")
        lines.append(
            f"{name_tex} & {err_avg:.1f} & {err_max} & "
            f"{chatty_avg:.1f} & {poor_prose_avg:.1f} & "
            f"{flub_tex} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Attribution footnote
    evaluator_tex = evaluator.replace("_", r"\_")
    lines.append(r"\vspace{2pt}")
    lines.append(
        r"\noindent{\footnotesize Evaluated by "
        + evaluator_tex
        + f", {n_sample} conversations per engine."
        + "}"
    )

    # Generate discussion via claude -p
    plain_header = (
        f"{'Engine':<35} {'ErrAvg':>6} {'ErrMax':>6} "
        f"{'Chatty':>6} {'PoorProse':>9}  Example flub"
    )
    plain_table = plain_header + "\n" + "\n".join(plain_rows)
    prompt = DISCUSSION_PROMPT.format(
        n=n_sample, evaluator=evaluator, table=plain_table,
    )

    print("  Generating discussion via claude -p ...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["claude", "-p"],
            input=prompt,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            discussion = result.stdout.strip()
        else:
            print("    WARNING: claude -p failed, using placeholder", file=sys.stderr)
            discussion = r"\textit{Discussion to be written.}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    WARNING: {e}, using placeholder", file=sys.stderr)
        discussion = r"\textit{Discussion to be written.}"

    lines.append("")
    lines.append(r"\vspace{6pt}")
    lines.append(r"\noindent")
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
