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

# Qwen3.5-4B GGUF (~2.7GB)
if [ ! -f Qwen3.5-4B-Q4_K_M.gguf ]; then
    echo "Downloading Qwen3.5-4B Q4_K_M (~2.7GB)..."
    wget -q --show-progress https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
else
    echo "Qwen3.5-4B GGUF already downloaded, skipping."
fi

# Whisper (auto-downloaded on first run by faster-whisper)
echo ""
echo "Note: The Whisper model (~75MB for 'tiny') is downloaded automatically on first run."

echo ""
echo "Done. Models directory:"
ls -lh "$MODELS_DIR"
