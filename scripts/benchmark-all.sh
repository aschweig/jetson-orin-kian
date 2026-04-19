#!/usr/bin/env bash
# Run the full benchmark suite N times, producing benchmark-results-all.csv
# with a Run column prepended.
#
# Usage:
#   scripts/benchmark-all.sh          # 10 runs (default)
#   scripts/benchmark-all.sh 3        # 3 runs
#   scripts/benchmark-all.sh 2 9      # 2 runs, starting at run 9 (appends to existing CSV)
#
# Sudo access (avoid password prompts during long runs):
#   This script and benchmark-llm.py invoke `sudo` for two commands:
#     - sudo cat /sys/kernel/debug/nvmap/iovmm/clients   (Jetson GPU diagnostics)
#     - sudo tee /proc/sys/vm/drop_caches                (page cache reset between models)
#   To enable passwordless sudo for ONLY those two commands, install:
#     sudo tee /etc/sudoers.d/kian-benchmark > /dev/null <<'EOF'
#     aaron ALL=(root) NOPASSWD: /usr/bin/cat /sys/kernel/debug/nvmap/iovmm/clients
#     aaron ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
#     EOF
#     sudo chmod 0440 /etc/sudoers.d/kian-benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UNLOAD="$SCRIPT_DIR/unload-ollama.sh"
BENCHMARK="$SCRIPT_DIR/benchmark-llm.py"
OUTFILE="$PROJECT_ROOT/benchmark-results-all.csv"

RUNS="${1:-5}"
START_RUN="${2:-1}"

if [ "$START_RUN" -gt 1 ] && [ -f "$OUTFILE" ]; then
    echo "Appending runs $START_RUN..$((START_RUN + RUNS - 1)) to $OUTFILE"
else
    echo "Running benchmark $RUNS times → $OUTFILE"
    echo "Run,Engine,PromptNo,TTFT,TotalTime,Tokens,TokPerSec,GPU%,VRAM_GB,TotalRAM_GB,Trimmed,SwapMB,CacheMB" > "$OUTFILE"
    START_RUN=1
fi

END_RUN=$((START_RUN + RUNS - 1))

for run in $(seq "$START_RUN" "$END_RUN"); do
    echo ""
    echo "============================================================"
    echo "  Run $run / $END_RUN"
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
