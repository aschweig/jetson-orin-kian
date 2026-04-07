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
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "benchmark-results-all.csv"
LATEX_OUT = PROJECT_ROOT / "docs" / "benchmark-table.tex"
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
}


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> list[dict]:
    """Group by Engine, compute Mean TTFT, SE, p95 TTFT, mean TPS, mean GPU%."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["Engine"]].append(r)

    results = []
    for engine, data in groups.items():
        ttfts = [float(r["TTFT"]) for r in data]
        tps_vals = [float(r["TokPerSec"]) for r in data]
        gpu_vals = [float(r["GPU%"]) for r in data]
        n = len(ttfts)

        mean_ttft = sum(ttfts) / n
        sd_ttft = math.sqrt(sum((x - mean_ttft) ** 2 for x in ttfts) / (n - 1)) if n > 1 else 0
        se_ttft = sd_ttft / math.sqrt(n)
        sorted_ttfts = sorted(ttfts)
        p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
        p95_ttft = sorted_ttfts[p95_idx]
        mean_tps = sum(tps_vals) / n
        mean_gpu = sum(gpu_vals) / n
        # VRAM/TotalRAM may not be present in older CSVs
        vram_vals = [float(r.get("VRAM_GB", 0)) for r in data]
        total_ram_vals = [float(r.get("TotalRAM_GB", 0)) for r in data]
        mean_vram = sum(vram_vals) / n if vram_vals else 0
        mean_total_ram = sum(total_ram_vals) / n if total_ram_vals else 0

        results.append({
            "engine": engine,
            "name": DISPLAY_NAMES.get(engine, engine),
            "mean_ttft": mean_ttft,
            "se": se_ttft,
            "p95_ttft": p95_ttft,
            "tps": mean_tps,
            "gpu": mean_gpu,
            "vram": mean_vram,
            "total_ram": mean_total_ram,
            "n": n,
        })

    results.sort(key=lambda r: r["mean_ttft"])
    return results


# Models with vision/multimodal capability (marked with footnote in LaTeX)
MULTIMODAL = {
    "ollama:Qwen3.5-2B",
    "ollama:Qwen3.5-4B",
    "llamacpp:Qwen3.5-2B",
    "ollama:Ministral-3 3B",
}


def make_latex(stats: list[dict]) -> str:
    """Generate LaTeX tabular body (rows only, no begin/end table)."""
    has_multimodal = any(r["name"] in MULTIMODAL for r in stats)
    lines = []
    lines.append(r"\begin{tabular}{l|r|r|r|r|r|r}")
    lines.append(r"\toprule")
    lines.append(r"Engine & Mean TTFT (s) & SE & p95 TTFT (s) & tok/s & GPU (\%) & RAM (GB) \\")
    lines.append(r"\midrule")

    # Split into full-offload and partial-offload groups
    full = [r for r in stats if r["gpu"] >= 99]
    partial = [r for r in stats if r["gpu"] < 99]

    def _row(r):
        name = r["name"].replace("_", r"\_")
        if r["name"] in MULTIMODAL:
            name += r"\textsuperscript{$\dagger$}"
        return (
            f"{name} & {r['mean_ttft']:.3f} & {r['se']:.3f} & "
            f"{r['p95_ttft']:.3f} & {r['tps']:.1f} & {r['gpu']:.0f} & "
            f"{r['total_ram']:.1f} \\\\"
        )

    for r in full:
        lines.append(_row(r))
    if partial:
        lines.append(r"\midrule")
        for r in partial:
            lines.append(_row(r))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if has_multimodal:
        lines.append(r"\vspace{2pt}")
        lines.append(r"\noindent{\footnotesize $\dagger$\,Multimodal (vision) model.}")
    return "\n".join(lines)


def make_markdown(stats: list[dict]) -> str:
    """Generate markdown table rows (full-offload models only)."""
    lines = []
    lines.append("| Engine | Mean TTFT | p95 TTFT | tok/s | GPU% | RAM |")
    lines.append("|--------|-----------|----------|-------|------|-----|")

    full = [r for r in stats if r["gpu"] >= 99]
    for r in full:
        lines.append(
            f"| {r['name']} | {r['mean_ttft']:.2f}s | "
            f"{r['p95_ttft']:.2f}s | {r['tps']:.1f} | {r['gpu']:.0f}% | "
            f"{r['total_ram']:.1f} GB |"
        )

    return "\n".join(lines)


def update_readme(md_table: str):
    """Replace the benchmark table in README.md."""
    text = README.read_text()

    # Match from the markdown table header to the blank line before "Models that"
    pattern = re.compile(
        r"(\| Engine \| Mean TTFT .*\n(?:\|.*\n)*)\n",
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

    # Verify exactly 5 runs
    EXPECTED_RUNS = 5
    runs = set()
    for r in rows:
        if "Run" in r:
            runs.add(r["Run"])
    num_runs = len(runs) if runs else 1

    stats = summarize(rows)

    # Print summary to stdout
    print(f"{'Engine':<40} {'TTFT':>8} {'SE':>6} {'p95':>8} {'tok/s':>6} {'GPU%':>5} {'VRAM':>6} {'RAM':>6}  n")
    print("-" * 100)
    for r in stats:
        print(f"{r['name']:<40} {r['mean_ttft']:>8.3f} {r['se']:>6.3f} "
              f"{r['p95_ttft']:>8.3f} {r['tps']:>6.1f} {r['gpu']:>5.0f} "
              f"{r['vram']:>5.1f}G {r['total_ram']:>5.1f}G  {r['n']}")

    if num_runs != EXPECTED_RUNS:
        print(f"\nFound {num_runs} runs (expected {EXPECTED_RUNS}). "
              f"Preview only — not updating LaTeX or README.")
        return

    # Write LaTeX
    latex = make_latex(stats)
    LATEX_OUT.write_text(latex + "\n")
    print(f"\nWrote {LATEX_OUT}")

    # Update README
    md_table = make_markdown(stats)
    update_readme(md_table)


if __name__ == "__main__":
    main()
