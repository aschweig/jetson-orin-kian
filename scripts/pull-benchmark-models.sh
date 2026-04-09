#!/usr/bin/env bash
# Pull/download all models used by the benchmark.
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"

# GGUF models for llama-cpp-python benchmarks
declare -A gguf_models=(
    ["Qwen3-4B-Q4_K_M.gguf"]="https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    ["granite-3.3-2b-instruct-Q4_K_M.gguf"]="https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-Q4_K_M.gguf"
    ["granite-4.0-micro-Q4_K_M.gguf"]="https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF/resolve/main/granite-4.0-micro-Q4_K_M.gguf"
    ["granite-4.0-h-micro-Q4_K_M.gguf"]="https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF/resolve/main/granite-4.0-h-micro-Q4_K_M.gguf"
)

for file in "${!gguf_models[@]}"; do
    if [ ! -f "$MODELS_DIR/$file" ]; then
        echo "Downloading $file ..."
        wget -q --show-progress -O "$MODELS_DIR/$file" "${gguf_models[$file]}"
    else
        echo "$file already downloaded, skipping."
    fi
    echo ""
done

# Ollama models
models=(
    qwen3.5:2b-q4_K_M
    qwen3:4b-q4_K_M
    llama3.2:3b-instruct-q4_K_M
    ministral-3:3b
    granite3.3:2b
    granite4:3b
    nemotron-3-nano:4b
)

for model in "${models[@]}"; do
    echo "Pulling $model ..."
    ollama pull "$model"
    echo ""
done

echo "All models pulled."
ollama list
