#!/usr/bin/env bash
# Download the Bonsai-8B GGUF model from HuggingFace.
#
# Usage: ./scripts/download-bonsai.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models"
MODEL_FILE="$MODEL_DIR/Bonsai-8B.gguf"

if [[ -f "$MODEL_FILE" ]]; then
    echo "Model already exists: $MODEL_FILE"
    ls -lh "$MODEL_FILE"
    exit 0
fi

echo "Downloading Bonsai-8B.gguf (~1.8GB)..."
mkdir -p "$MODEL_DIR"

# huggingface-cli is the most reliable way, but curl works too
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download prism-ml/Bonsai-8B-gguf Bonsai-8B.gguf \
        --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
else
    curl -L -o "$MODEL_FILE" \
        "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf"
fi

echo "Downloaded: $(ls -lh "$MODEL_FILE" | awk '{print $5}')"
