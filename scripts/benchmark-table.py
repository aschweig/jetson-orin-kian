#!/usr/bin/env python3
"""Generate benchmark summary tables from benchmark-results-all.csv.

Produces:
  - docs/benchmark-table.tex   LaTeX tabular body for \\input{} in litepaper.tex
  - Updates the markdown table in README.md between sentinel comments

Usage:
    uv run python scripts/benchmark-table.py
    uv run python scripts/benchmark-table.py path/to/results.csv
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "benchmark-results-all.csv"
LATEX_OUT = PROJECT_ROOT / "docs" / "benchmark-table.tex"
PROBE_LATEX_OUT = PROJECT_ROOT / "docs" / "benchmark-probe-table.tex"
README = PROJECT_ROOT / "README.md"

# Display names: raw engine string -> pretty name
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


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> list[dict]:
    """Group by Engine, compute TTFT stats split by trim/non-trim, plus mean tok/s."""
    # Exclude single-token probe rows (e.g. "Ummm", "Okay") from stats.
    rows = [r for r in rows if int(r.get("Tokens", 2)) > 1]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["Engine"]].append(r)

    results = []
    for engine, data in groups.items():
        tps_vals = [float(r["TokPerSec"]) for r in data]
        mean_tps = sum(tps_vals) / len(tps_vals)

        # Split by trim status
        nt_rows = [r for r in data if int(r.get("Trimmed", 0)) == 0]
        tr_rows = [r for r in data if int(r.get("Trimmed", 0)) > 0]
        nt_ttfts = [float(r["TTFT"]) for r in nt_rows]
        tr_ttfts = [float(r["TTFT"]) for r in tr_rows]

        results.append({
            "engine": engine,
            "name": DISPLAY_NAMES.get(engine, engine),
            "nt_avg": sum(nt_ttfts) / len(nt_ttfts) if nt_ttfts else 0,
            "nt_max": max(nt_ttfts) if nt_ttfts else 0,
            "tr_avg": sum(tr_ttfts) / len(tr_ttfts) if tr_ttfts else 0,
            "tr_max": max(tr_ttfts) if tr_ttfts else 0,
            "tps": mean_tps,
            "n": len(data),
            "n_nt": len(nt_rows),
            "n_tr": len(tr_rows),
        })

    results.sort(key=lambda r: r["nt_avg"])
    return results


def make_latex(stats: list[dict]) -> str:
    """Generate LaTeX tabular body (rows only, no begin/end table)."""
    lines = []
    lines.append(r"\begin{tabular}{l|rr|rr|r}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c|}{Steady-State TTFT} & \multicolumn{2}{c|}{Post-Trim TTFT} & \\")
    lines.append(r"Engine & Avg (s) & Max (s) & Avg (s) & Max (s) & tok/s \\")
    lines.append(r"\midrule")

    for r in stats:
        name = r["name"].replace("_", r"\_")
        if r["n_tr"]:
            tr_avg = f"{r['tr_avg']:.3f}"
            tr_max = f"{r['tr_max']:.3f}"
        else:
            tr_avg = "---"
            tr_max = "---"
        lines.append(
            f"{name} & {r['nt_avg']:.3f} & {r['nt_max']:.3f} & "
            f"{tr_avg} & {tr_max} & {r['tps']:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def summarize_probes(rows: list[dict]) -> list[dict]:
    """Per-engine TTFT stats on the single-token probe prompts ("Ummm", "Okay"),
    restricted to probes that did not trigger a context trim."""
    rows = [
        r for r in rows
        if int(r.get("Tokens", 2)) == 1 and int(r.get("Trimmed", 0)) == 0
    ]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["Engine"]].append(r)

    results = []
    for engine, data in groups.items():
        ttfts = [float(r["TTFT"]) for r in data]
        if not ttfts:
            continue
        results.append({
            "engine": engine,
            "name": DISPLAY_NAMES.get(engine, engine),
            "avg": sum(ttfts) / len(ttfts),
            "max": max(ttfts),
            "n": len(ttfts),
        })

    results.sort(key=lambda r: r["avg"])
    return results


def make_probe_latex(stats: list[dict]) -> str:
    """LaTeX tabular body for the probe-prompt TTFT table (non-trim only)."""
    lines = []
    lines.append(r"\begin{tabular}{l|r|r}")
    lines.append(r"\toprule")
    lines.append(r"Engine & Avg TTFT (s) & Max TTFT (s) \\")
    lines.append(r"\midrule")
    for r in stats:
        name = r["name"].replace("_", r"\_")
        lines.append(f"{name} & {r['avg']:.3f} & {r['max']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_markdown(stats: list[dict]) -> str:
    """Generate markdown table."""
    lines = []
    lines.append("| Engine | Avg TTFT | Max TTFT | Post-Trim Avg | Post-Trim Max | tok/s |")
    lines.append("|--------|----------|----------|---------------|---------------|-------|")

    for r in stats:
        if r["n_tr"]:
            tr_avg = f"{r['tr_avg']:.2f}s"
            tr_max = f"{r['tr_max']:.2f}s"
        else:
            tr_avg = "---"
            tr_max = "---"
        lines.append(
            f"| {r['name']} | {r['nt_avg']:.2f}s | {r['nt_max']:.2f}s | "
            f"{tr_avg} | {tr_max} | {r['tps']:.1f} |"
        )

    return "\n".join(lines)


def update_readme(md_table: str):
    """Replace the benchmark table in README.md."""
    text = README.read_text()

    # Match from the markdown table header to the blank line after the table
    pattern = re.compile(
        r"(\| Engine \| (?:Mean TTFT|Avg TTFT) .*\n(?:\|.*\n)*)\n",
        re.MULTILINE,
    )
    replacement = md_table + "\n\n"
    new_text, count = pattern.subn(replacement, text)
    if count == 0:
        print("WARNING: Could not find benchmark table in README.md", file=sys.stderr)
        return
    README.write_text(new_text)
    print(f"Updated {README}")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(csv_path)

    stats = summarize(rows)

    # Print summary to stdout
    hdr1 = f"{'':40} {'Steady-State TTFT':>17}  {'Post-Trim TTFT':>17}"
    hdr2 = f"{'Engine':<40} {'Avg':>8} {'Max':>8}  {'Avg':>8} {'Max':>8}  {'tok/s':>6}   n"
    print(hdr1)
    print(hdr2)
    print("-" * len(hdr2))
    for r in stats:
        if r["n_tr"]:
            tr_avg = f"{r['tr_avg']:>8.3f}"
            tr_max = f"{r['tr_max']:>8.3f}"
        else:
            tr_avg = f"{'---':>8}"
            tr_max = f"{'---':>8}"
        print(f"{r['name']:<40} {r['nt_avg']:>8.3f} {r['nt_max']:>8.3f}  "
              f"{tr_avg} {tr_max}  {r['tps']:>6.1f}  {r['n']:>3}")

    # Write LaTeX
    latex = make_latex(stats)
    LATEX_OUT.write_text(latex + "\n")
    print(f"\nWrote {LATEX_OUT}")

    # Probe-prompt LaTeX (TTFT on "Ummm"/"Okay" filler inputs, non-trim only)
    probe_stats = summarize_probes(rows)
    if probe_stats:
        probe_latex = make_probe_latex(probe_stats)
        PROBE_LATEX_OUT.write_text(probe_latex + "\n")
        print(f"Wrote {PROBE_LATEX_OUT}")

    # Update README
    md_table = make_markdown(stats)
    update_readme(md_table)


if __name__ == "__main__":
    main()
