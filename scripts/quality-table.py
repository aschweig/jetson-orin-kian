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

Below is a previous version of the discussion. Use it as a model for style \
and structure. Update all numbers and claims to match the new data above. \
Add coverage of any new models in the table. Remove coverage of models no \
longer in the table. Correct any claims that the new data does not support.

--- Previous discussion ---
Among the models evaluated, Qwen3-4B under llama.cpp produced no factual \
errors, no verbose responses, and no weak prose across all 10 runs, while \
Granite 4.0 Horizon Micro matched it on accuracy with only a slight weakness \
in prose quality (1.1). The smaller Granite 3.3-2B kept errors near zero \
(0.3 avg) with clean, concise output. Llama 3.2-3B was competitive at 0.8 \
errors on average with low verbosity (0.3 chatty), though its error ceiling \
was higher (max 3). Notably, inference engine choice had a pronounced effect: \
Qwen3-4B produced no errors under llama.cpp but degraded under its Ollama \
variant (1.8 errors avg, 0.8 chatty), and Granite 3.3-2B similarly worsened \
from 0.3 to 1.2 errors. Since the two runtimes may use different \
quantization implementations, prompt templates, or sampling defaults, this \
suggests that runtime configuration matters as much as model selection itself.

A clear accuracy--verbosity trade-off emerged across the Granite family. The \
full-precision \\textbf{{Granite 4.0 Micro}} produced only 1.0 errors on \
average but scored worst on prose quality (3.0), generating stilted \
longer-form responses. Its IQ4-quantized variant matched on accuracy while \
improving prose to 2.0, and the 3B Ollama variant (\\textbf{{Granite 4-3B}}) \
traded slightly higher error rates (1.1) for markedly better prose (1.4) and \
modest verbosity (0.2). \\textbf{{Ministral-3 3B}} illustrated the opposite \
failure mode: reasonable accuracy (1.2) but extreme chattiness (2.1), which \
is particularly undesirable in a child-facing voice interface where verbose \
responses discourage the back-and-forth dialogue that drives learning. \
\\textbf{{Qwen3.5-2B}} showed the cost of aggressive size reduction, with \
errors climbing from 2.0 under llama.cpp to 3.6 under Ollama---the latter \
hallucinating that ``the fastest fish in water is the dolphin.'' For this \
use case, the results favor Qwen3-4B (llama.cpp) and Granite 4.0 Horizon \
Micro as the most promising options among the models tested.

\\subsection{{Benchmark Limitations}}

This benchmark is a task-specific internal evaluation, not a general \
capability ranking. The prompt set is small (8 prompts) and English-only, \
covering a narrow slice of the topics a child might ask about. Qualitative \
scores were assigned by an LLM judge (Claude Opus 4.6) rather than human \
raters, which may not capture all dimensions of child-appropriateness such \
as tone, pacing, or emotional register. The benchmark also does not evaluate \
the content safety pipeline or the LaTeX-to-speech system. Results should be \
interpreted as a comparative signal for model selection on this hardware, \
not as absolute quality claims.
--- End previous discussion ---

Write 1-2 short paragraphs of discussion followed by a Benchmark Limitations \
subsection as shown above. Update the prompt count if it has changed.

Output ONLY the LaTeX body text (no \\begin{{document}}). \
Use \\textbf{{Model Name}} when first mentioning a model. Do not use \
markdown. Do not wrap in a code block. Do not use thinking tags."""


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
    # Count prompts from benchmark file
    prompts_file = PROJECT_ROOT / "scripts" / "benchmark-prompts.txt"
    n_prompts = len([l for l in prompts_file.read_text().strip().splitlines() if l.strip()]) if prompts_file.exists() else "?"

    # Sort by mean errors ascending, then prose descending
    engines_sorted = sorted(
        scores.keys(),
        key=lambda e: (
            sum(d["errors"] for d in scores[e]) / len(scores[e]),
            sum(d["poor_prose"] for d in scores[e]) / len(scores[e]),
        ),
    )

    # --- Scores table (no flubs) ---
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l|r|r|r|r}",
        r"\toprule",
        r"Engine & Err Avg & Err Max & Chatty & Poor Prose \\",
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
        err_avg = sum(d["errors"] for d in s) / n
        err_max = max(d["errors"] for d in s)
        chatty_avg = sum(d["chatty"] for d in s) / n
        poor_prose_avg = sum(d["poor_prose"] for d in s) / n
        flub = best_flubs.get(engine, "")

        # Plain-text row for LLM prompt (ASCII-only flub)
        flub_ascii = "".join(c if ord(c) < 128 else "*" for c in flub)
        if flub_ascii:
            plain_rows.append(
                f"{name:<35} {err_avg:>5.1f} {err_max:>5} "
                f"{chatty_avg:>6.1f} {poor_prose_avg:>5.1f}  "
                f'"{flub_ascii}"'
            )
        else:
            plain_rows.append(
                f"{name:<35} {err_avg:>5.1f} {err_max:>5} "
                f"{chatty_avg:>6.1f} {poor_prose_avg:>5.1f}  ---"
            )

        # Scores row
        name_tex = name.replace("_", r"\_")
        lines.append(
            f"{name_tex} & {err_avg:.1f} & {err_max} & "
            f"{chatty_avg:.1f} & {poor_prose_avg:.1f} \\\\"
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
        + r"All metrics are lower-is-better: Err = factual errors per conversation, "
        + r"Chatty = verbose or repetitive responses (of 5), "
        + r"Poor Prose = weak explanatory/story responses (of 3). "
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
