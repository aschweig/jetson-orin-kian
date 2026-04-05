#!/usr/bin/env bash
# Pull all Ollama models used by the benchmark.
set -euo pipefail

models=(
    qwen3.5:2b-q4_K_M
    qwen3:4b-q4_K_M
    qwen3.5:4b-q4_K_M
    llama3.2:3b-instruct-q4_K_M
    ministral-3:3b
    gemma3n:e2b
    granite3.3:2b
)

for model in "${models[@]}"; do
    echo "Pulling $model ..."
    ollama pull "$model"
    echo ""
done

echo "All models pulled."
ollama list
