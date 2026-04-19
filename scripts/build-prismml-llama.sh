#!/usr/bin/env bash
# Build PrismML's llama.cpp fork with CUDA support for Jetson Orin.
# This fork adds 1-bit quant support needed for Bonsai-8B.
#
# Usage: ./scripts/build-prismml-llama.sh [--clean]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/vendor/prismml-llama.cpp"

if [[ "${1:-}" == "--clean" ]] && [[ -d "$BUILD_DIR" ]]; then
    echo "Cleaning $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Cloning PrismML llama.cpp fork..."
    mkdir -p "$PROJECT_DIR/vendor"
    git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp "$BUILD_DIR"
else
    echo "Using existing clone at $BUILD_DIR"
    echo "  (pass --clean to re-clone)"
fi

cd "$BUILD_DIR"

echo "Building with CUDA support..."
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

echo ""
echo "Build complete. Binaries at:"
echo "  $BUILD_DIR/build/bin/llama-server"
echo "  $BUILD_DIR/build/bin/llama-cli"
echo ""
echo "To benchmark Bonsai-8B:"
echo "  $BUILD_DIR/build/bin/llama-bench -m models/Bonsai-8B.gguf -t 6 -ngl 99 -p 64 -n 32"
echo ""
echo "To run the server for Kian (GPU offload with -ngl 99):"
echo "  $BUILD_DIR/build/bin/llama-server -m models/Bonsai-8B.gguf -c 2048 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --threads 6 --host 127.0.0.1 --port 8080"
echo "  uv run kian --backend server"
