#!/usr/bin/env bash
# Run the full benchmark suite N times, producing benchmark-results-all.csv
# with a Run column prepended.
#
# Usage:
#   scripts/benchmark-all.sh          # 5 runs (default)
#   scripts/benchmark-all.sh 3        # 3 runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UNLOAD="$SCRIPT_DIR/unload-ollama.sh"
BENCHMARK="$SCRIPT_DIR/benchmark-llm.py"
OUTFILE="$PROJECT_ROOT/benchmark-results-all.csv"

RUNS="${1:-5}"

echo "Running benchmark $RUNS times → $OUTFILE"

# Write CSV header
echo "Run,Engine,PromptNo,TTFT,TotalTime,Tokens,TokPerSec,GPU%,VRAM_GB,TotalRAM_GB" > "$OUTFILE"

for run in $(seq 1 "$RUNS"); do
    echo ""
    echo "============================================================"
    echo "  Run $run / $RUNS"
    echo "============================================================"

    # Clean GPU state before each run
    "$UNLOAD" 2>/dev/null || true
    sleep 5

    # Memory diagnostics (temporary)
    echo "  --- Memory diagnostics (pre-run $run) ---"
    echo "  Top memory consumers:"
    ps aux --sort=-%mem | head -10 | awk '{printf "    %-8s %5s%% %s\n", $1, $4, $11}'
    echo "  GPU (nvmap) clients:"
    sudo cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | sed 's/^/    /'
    echo "  /proc/meminfo:"
    grep -E 'MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree' /proc/meminfo | sed 's/^/    /'
    echo "  ---"

    tmpcsv=$(mktemp)
    if uv run python "$BENCHMARK" --csv "$tmpcsv"; then
        echo "  Run $run completed successfully."
    else
        echo "  Run $run had errors (partial results kept)."
    fi

    # Append whatever rows were collected (skip header)
    if [ -f "$tmpcsv" ]; then
        tail -n +2 "$tmpcsv" | while IFS= read -r line; do
            echo "$run,$line" >> "$OUTFILE"
        done
        rm -f "$tmpcsv"
    fi

    # Clean up after each run
    "$UNLOAD" 2>/dev/null || true
    sleep 2
done

echo ""
echo "Done. Results in $OUTFILE"
echo "$(wc -l < "$OUTFILE") rows (including header)"
