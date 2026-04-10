# Kian

A local voice assistant running entirely on-device. Designed for NVIDIA Jetson Orin (8GB), but works on any Linux machine with a microphone.

![WIP](wip.jpg)

**Pipeline:** Mic → Silero VAD → faster-whisper (STT) → LLM → Piper (TTS) → Speaker

Two LLM backends:

| Backend | Model | How it runs |
|---------|-------|-------------|
| `ollama` (default) | Qwen3-4B (Q4_K_M) | Ollama server with OpenAI-compatible API |
| `llamacpp` | Qwen3.5-2B (GGUF) | In-process via llama-cpp-python |

All components run locally. No cloud APIs required.

## Requirements

- Linux (tested on Jetson Orin, Ubuntu 22.04, JetPack 6.2)
- Python 3.12+ (managed by uv)
- [uv](https://docs.astral.sh/uv/) package manager
- A microphone and speakers/headphones
- ~4GB disk for models
- [Ollama](https://ollama.com/) (required for the default backend)

## System Dependencies

```bash
sudo apt install libportaudio2 libsndfile1
```

These are required by `sounddevice` (audio I/O) and `piper-tts`. On Jetson Orin, these are not installed by default.

### CUDA PATH (Jetson)

Ensure `nvcc` is on your PATH. Add to `~/.bashrc`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

Then `source ~/.bashrc` or open a new terminal.

## Audio Device Setup

Kian uses the system default audio device for both input (mic) and output (speaker). If you're using a USB audio adapter, set it as the default before running Kian.

**GNOME (easiest):** Open Settings > Sound and select your device for both Input and Output.

**Command line (pactl):**

```bash
# List available devices
pactl list sources short   # input devices
pactl list sinks short     # output devices

# Set defaults (replace with your device names)
pactl set-default-source alsa_input.usb-YOUR_DEVICE_NAME.analog-stereo
pactl set-default-sink alsa_output.usb-YOUR_DEVICE_NAME.analog-stereo
```

**Verify:** You can test your audio setup with:

```bash
# Record 3 seconds and play back
arecord -D default -f S16_LE -r 48000 -c 2 -d 3 /tmp/test.wav
aplay -D default /tmp/test.wav
```

## Status LEDs (optional)

Kian can drive a [Pi Traffic Light](https://www.amazon.com/dp/B00RIIGD30) on the Jetson GPIO header to show status: **yellow = idle/listening**, **red = busy (processing)**.

Wire the traffic light to header pins 29 (red), 30 (GND), 31 (yellow), 32 (green — unused).

**First-time setup:**

```bash
sudo bash scripts/enable_gpio.sh
```

Then log out and back in for gpio group permissions to take effect.

**After each reboot**, re-run to configure the pinmux (the pin direction settings don't persist):

```bash
sudo bash scripts/enable_gpio.sh
```

If no GPIO is available (e.g. non-Jetson machine), the LEDs are silently skipped.

## Setup

1. **Install uv** (if you don't have it):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Install Ollama** (default LLM backend):

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   Ollama runs as a systemd service and starts automatically on boot.

4. **Download models:**

   ```bash
   bash scripts/download-models.sh
   ```

   This downloads:
   - **Silero VAD** (~2.3MB) into `models/`
   - **Piper TTS** voice (`en_US-lessac-medium`, ~75MB) into `models/`
   - **Qwen3.5-2B** GGUF (`Q4_K_M`, ~1.6GB) into `models/` (llamacpp fallback)
   - **Qwen3-4B** via Ollama (`Q4_K_M`, ~2.7GB)
   - **Qwen2.5-0.5B** GGUF (`Q2_K`, ~415MB) into `models/` (safety classifier)
   - **Whisper** (`tiny`, ~75MB) is downloaded automatically on first run by faster-whisper

5. **Rebuild llama-cpp-python with CUDA** (only needed for `--backend llamacpp`):

   ```bash
   CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-binary llama-cpp-python --no-cache llama-cpp-python
   ```

   This compiles llama.cpp from source with CUDA support. Takes several minutes.
   Skip this if you only plan to use the default Ollama backend.

6. **Run:**

   ```bash
   uv run kian
   ```

## Usage

```bash
# Default: Qwen3-4B via Ollama
uv run kian

# Qwen3.5-2B via llama.cpp (lower latency, smaller model)
uv run kian --backend llamacpp

# Custom model with llama.cpp
uv run kian --backend llamacpp --model models/some-other-model.gguf
```

Press Q + Enter to quit.

## Project Structure

```
kian/
├── kian/
│   ├── app.py          # async pipeline wiring everything together
│   ├── vad.py          # voice activity detection (Silero VAD)
│   ├── stt.py          # speech-to-text (faster-whisper)
│   ├── leds.py         # status LEDs via Jetson GPIO
│   ├── llm.py          # backend selection + shared interface
│   ├── llm_llamacpp.py # llama.cpp backend (Qwen3.5-2B)
│   ├── llm_ollama.py   # Ollama backend (Qwen3-4B)
│   ├── safety.py       # LLM-based content safety classifier
│   └── tts.py          # text-to-speech + playback thread (Piper)
├── models/             # model files (not checked in)
├── scripts/
│   ├── download-models.sh
│   └── enable_gpio.sh  # GPIO setup for status LEDs
└── pyproject.toml
```

## How It Works

1. **VAD** continuously listens on the mic and detects speech segments
2. **Whisper** transcribes each speech segment to text
3. **LLM** streams a response token-by-token (via llama.cpp or Ollama)
4. Tokens are buffered and split on punctuation boundaries (~25 chars min)
5. Each text fragment is synthesized by **Piper TTS** and queued for playback
6. A background thread plays audio chunks sequentially, so playback starts while the LLM is still generating

## Models

Models are stored in `models/` and excluded from git. You can swap them:

- **LLM (llama.cpp):** Any GGUF model works. Pass `--model path/to/model.gguf`.
- **LLM (Ollama):** Pass `--model "model:tag"` with any model from the [Ollama library](https://ollama.com/library).
- **TTS voice:** Browse [Piper voices](https://github.com/rhasspy/piper/blob/master/VOICES.md). Download the `.onnx` + `.onnx.json` pair.
- **Whisper:** Change `MODEL_SIZE` in `kian/stt.py` (`tiny`, `base`, `small`, `medium`).

## Headless Mode (recommended for production / toy use)

Running without the GNOME desktop frees ~1-1.3 GB of RAM, which can allow full GPU offload of the LLM for much faster inference.

**Switch to headless (persists across reboot):**

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

You'll get a text console login on HDMI with keyboard input. You can also SSH in.

**Switch back to desktop:**

```bash
sudo systemctl set-default graphical.target
sudo reboot
```

**One-time switch (no reboot, no permanent change):**

```bash
# Drop to text console
sudo systemctl isolate multi-user.target

# Go back to desktop
sudo systemctl isolate graphical.target
```

Note: PulseAudio may not auto-start in headless mode. If audio breaks, start it manually:

```bash
pulseaudio --start
```

## Benchmarks (Jetson Orin Nano, 8GB)

Measured over 5 runs x 8 prompts per engine. All models use Q4_K_M quantization and 2048-token context.

| Engine | Mean TTFT | p95 TTFT | tok/s | GPU% | RAM |
|--------|-----------|----------|-------|------|-----|
| llamacpp:Granite 3.3-2B | 0.09s | 0.20s | 25.4 | 100% | 1.6 GB |
| llamacpp:Granite 4.0 Micro IQ4 | 0.10s | 0.22s | 24.3 | 100% | 1.9 GB |
| llamacpp:Granite 4.0 Micro | 0.11s | 0.23s | 18.9 | 100% | 2.1 GB |
| llamacpp:Granite 4.0 H-Micro | 0.13s | 0.32s | 17.6 | 100% | 1.9 GB |
| llamacpp:Qwen3-4B | 0.17s | 0.30s | 15.1 | 100% | 2.5 GB |
| ollama:Granite 3.3-2B | 0.23s | 0.33s | 25.8 | 100% | 1.8 GB |
| llamacpp:Qwen3.5-2B | 0.32s | 0.51s | 25.1 | 100% | 1.3 GB |
| ollama:Granite 4-3B | 0.36s | 0.47s | 18.5 | 100% | 2.4 GB |
| ollama:Qwen3-4B | 0.51s | 0.65s | 15.5 | 100% | 3.4 GB |
| ollama:Llama 3.2-3B | 0.53s | 0.61s | 19.1 | 100% | 2.5 GB |
| ollama:Ministral-3 3B | 0.59s | 0.73s | 19.5 | 100% | 4.8 GB |
| ollama:Nemotron-3 Nano 4B | 1.02s | 1.56s | 15.6 | 100% | 5.2 GB |
| ollama:Qwen3.5-2B | 1.03s | 1.31s | 22.2 | 100% | 3.5 GB |

The default backend (Qwen3-4B via Ollama, **bold**) was selected for best accuracy and response quality.
See the [litepaper](docs/litepaper.tex) for qualitative evaluation details.

## Memory Budget (~8GB, headless)

Running headless (no desktop environment) frees 1--1.3 GB of RAM for GPU offload.

| Component | RAM |
|-----------|-----|
| OS/system (headless) | ~1.0 GB |
| Ollama + Qwen3-4B Q4_K_M | ~3.4 GB |
| Safety classifier (CPU) | ~0.4 GB |
| Whisper tiny | ~0.1 GB |
| Piper TTS + VAD | ~0.2 GB |
| KV cache (2K context) | ~0.3 GB |
| **Headroom** | **~2.4 GB** |
