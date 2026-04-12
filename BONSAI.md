# Bonsai-8B on Jetson Orin Nano

Bonsai-8B is a 1-bit quantized Qwen3-8B from [PrismML](https://prismml.com/news/bonsai-8b).
It fits an 8.2B-parameter model in 1.07 GiB using their custom `Q1_0_g128` quantization format.

This document covers running Bonsai-8B as the LLM backend for Kian on a Jetson Orin Nano Super.

## Quick start

```bash
# 1. Build PrismML's llama.cpp fork (one-time, ~20 min on Orin)
./scripts/build-prismml-llama.sh

# 2. Download the model (~1.1 GB)
./scripts/download-bonsai.sh

# 3. Run Kian (auto-starts llama-server as a child process)
uv run kian --backend server
```

The `server` backend automatically launches `llama-server` with GPU offloading and
terminates it when Kian exits. No separate terminal needed.

## Manual server usage and benchmarking

To run the server manually (e.g. for tuning flags or running multiple clients):

```bash
./vendor/prismml-llama.cpp/build/bin/llama-server \
    -m models/Bonsai-8B.gguf -c 2048 -ngl 99 \
    -fa on -ctk q4_0 -ctv q4_0 --threads 6 \
    --host 127.0.0.1 --port 8080
```

If a server is already listening on port 8080, `--backend server` will use it
instead of spawning a new one.

To benchmark:

```bash
./vendor/prismml-llama.cpp/build/bin/llama-bench \
    -m models/Bonsai-8B.gguf -t 6 -ngl 99 -p 64 -n 32
```

## Why a custom llama.cpp fork?

Bonsai-8B uses PrismML's custom `Q1_0_g128` quantization type (1-bit weights with groups
of 128). This is **not** the same as the `TQ1_0`/`TQ2_0` ternary types in upstream
llama.cpp. The upstream project does not support `Q1_0_g128` at all.

PrismML maintains a fork with:
- Dequantization kernels for `Q1_0_g128` (CPU: generic + NEON-optimized ARM, x86/AVX2; GPU: CUDA)
- GGUF loader support for the new quant type

We use:
- **Repo:** https://github.com/PrismML-Eng/llama.cpp
- **Branch:** `prism` (the default branch)
- **Commit:** `ba7e817` (2026-04-06)

There is an upstream PR to merge CUDA 1-bit support into mainline llama.cpp
(ggml-org/llama.cpp#21629), but it has not been merged as of 2026-04-12.

## Why server mode?

Kian normally uses `llama-cpp-python` (the Python bindings) to load GGUF models
in-process. That doesn't work here because `llama-cpp-python` would need to be
compiled against PrismML's fork, and keeping that in sync is fragile.

Instead, Kian's `--backend server` talks to `llama-server` over HTTP using the
OpenAI-compatible `/v1/chat/completions` endpoint. This decouples the custom C++
build from the Python environment.

## Non-trivial details

### GPU offloading is critical

Without `-ngl 99`, the model runs on CPU only and generation drops to ~1.7 t/s
(unusable for voice). With full GPU offload:

| Metric | `-ngl 99` (GPU) | `-ngl 0` (CPU) |
|---|---|---|
| Prompt processing | 260 t/s | 100 t/s |
| Token generation | 14.6 t/s | 1.7 t/s |
| Model VRAM usage | ~1016 MiB | 0 |

The GPU has CUDA kernels for `Q1_0_g128` in PrismML's fork despite what the upstream
PR status might suggest. The fork's CUDA path works on Orin (sm_87).

### Flash attention flag syntax

The `-fa` flag in this fork requires an explicit value: `-fa on`, not just `-fa`.
Running `-fa` without a value causes it to consume the next argument (e.g. `-ctk`)
as the value and fail.

### KV cache quantization

`-ctk q4_0 -ctv q4_0` quantizes the KV cache to 4-bit, cutting context memory from
~324 MiB to ~81 MiB for a 2048-token context. This is important on the Orin Nano where
GPU memory is shared with the system (8 GB total).

### The model identifies itself as "Bonsai"

The model's chat template makes it self-identify as "Bonsai, an AI assistant developed
by PrismML." Kian's system prompt overrides this with the assistant's configured name,
so this isn't an issue in practice.

### No thinking mode leakage

The server disables Qwen3's thinking mode (`thinking = 0` in the server log). No
`<think>` tags appear in responses.

### Build target for Jetson Orin

The cmake build uses `-DCMAKE_CUDA_ARCHITECTURES="87"` for the Orin's Ampere GPU
(compute capability 8.7). Change this if targeting a different Jetson variant.

## Memory footprint

With the server running Bonsai-8B (`-ngl 99 -c 2048 -ctk q4_0 -ctv q4_0`):

| Component | Memory |
|---|---|
| Model weights (GPU) | 1016 MiB |
| KV cache (GPU) | 81 MiB |
| Compute buffers (GPU) | 304 MiB |
| Host-side buffers | ~103 MiB |
| **Total GPU** | **~1401 MiB** |

This leaves ~3.3 GiB of GPU memory free for the rest of the system (Whisper, safety
model, etc.).

## Origin

Suggested by Dan Zenchelsky (2026-04-12), who reported 114 t/s prompt / 16.5 t/s
generation on an Orin Nano using CPU-only inference (`--threads 6`). Our GPU-offloaded
results are significantly better for generation (14.6 vs 1.7 t/s CPU-only), suggesting
his numbers may have been on x86 or a different build configuration.
