#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR ..."

# Silero VAD ONNX (~2.3MB)
if [ ! -f silero_vad.onnx ]; then
    echo "Downloading Silero VAD model..."
    wget -q --show-progress https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
else
    echo "Silero VAD already downloaded, skipping."
fi

# Piper TTS voice (~75MB)
if [ ! -f en_US-lessac-medium.onnx ]; then
    echo "Downloading Piper TTS voice model..."
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
else
    echo "Piper TTS voice already downloaded, skipping."
fi

# Qwen3.5-2B GGUF for llama.cpp (~1.6GB)
if [ ! -f Qwen3.5-2B-Q4_K_M.gguf ]; then
    echo "Downloading Qwen3.5-2B Q4_K_M (~1.6GB)..."
    wget -q --show-progress https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf
else
    echo "Qwen3.5-2B GGUF already downloaded, skipping."
fi

# Granite 4.0 Micro GGUF for llama.cpp (~2.1GB)
if [ ! -f granite-4.0-micro-Q4_K_M.gguf ]; then
    echo "Downloading Granite 4.0 Micro Q4_K_M (~2.1GB)..."
    wget -q --show-progress https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF/resolve/main/granite-4.0-micro-Q4_K_M.gguf
else
    echo "Granite 4.0 Micro GGUF already downloaded, skipping."
fi

# Granite 4.0 H-Micro (hybrid SSM+attention) GGUF for llama.cpp (~1.9GB)
if [ ! -f granite-4.0-h-micro-Q4_K_M.gguf ]; then
    echo "Downloading Granite 4.0 H-Micro Q4_K_M (~1.9GB)..."
    wget -q --show-progress https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF/resolve/main/granite-4.0-h-micro-Q4_K_M.gguf
else
    echo "Granite 4.0 H-Micro GGUF already downloaded, skipping."
fi

# Qwen3-4B GGUF for llama.cpp (default backend, ~2.5GB)
if [ ! -f Qwen3-4B-Q4_K_M.gguf ]; then
    echo "Downloading Qwen3-4B Q4_K_M (~2.5GB)..."
    wget -q --show-progress https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
else
    echo "Qwen3-4B GGUF already downloaded, skipping."
fi

# Qwen3-4B via Ollama (alternative backend, ~2.7GB)
if command -v ollama &>/dev/null; then
    echo "Pulling Qwen3-4B via Ollama..."
    ollama pull qwen3:4b-q4_K_M
else
    echo "Ollama not installed, skipping Qwen3-4B pull."
    echo "  Install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

# Qwen2.5-0.5B Instruct GGUF for safety classifier (~415MB, CPU-only)
if [ ! -f qwen2.5-0.5b-instruct-q2_k.gguf ]; then
    echo "Downloading Qwen2.5-0.5B Instruct Q2_K (~415MB)..."
    wget -q --show-progress https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf
else
    echo "Qwen2.5-0.5B Instruct GGUF already downloaded, skipping."
fi

# Whisper (auto-downloaded on first run by faster-whisper)
echo ""
echo "Note: The Whisper model (~75MB for 'tiny') is downloaded automatically on first run."

echo ""
echo "Done. Models directory:"
ls -lh "$MODELS_DIR"
