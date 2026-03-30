# Kian

A local voice assistant running entirely on-device. Designed for NVIDIA Jetson Orin (8GB), but works on any Linux machine with a microphone.

**Pipeline:** Mic → Silero VAD → faster-whisper (STT) → LLM → Piper (TTS) → Speaker

Two LLM backends:

| Backend | Model | How it runs |
|---------|-------|-------------|
| `llamacpp` (default) | Qwen3.5-2B (GGUF) | In-process via llama-cpp-python |
| `mlc` | Qwen3-4B (MLC) | Docker container with OpenAI-compatible API |

All components run locally. No cloud APIs required.

## Requirements

- Linux (tested on Jetson Orin, Ubuntu 22.04, JetPack 6.2)
- Python 3.12+ (managed by uv)
- [uv](https://docs.astral.sh/uv/) package manager
- A microphone and speakers/headphones
- ~4GB disk for models
- Docker with NVIDIA runtime (only for `--backend mlc`)

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

3. **Rebuild llama-cpp-python with CUDA** (for GPU-accelerated LLM inference):

   ```bash
   CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-binary llama-cpp-python --no-cache llama-cpp-python
   ```

   This compiles llama.cpp from source with CUDA support. Takes several minutes.
   Without this step, the LLM runs on CPU only and will be very slow.

4. **Download models:**

   ```bash
   bash scripts/download-models.sh
   ```

   This downloads:
   - **Silero VAD** (~2.3MB) into `models/`
   - **Piper TTS** voice (`en_US-lessac-medium`, ~75MB) into `models/`
   - **Qwen3.5-2B** GGUF (`Q4_K_M`, ~1.6GB) into `models/`
   - **Whisper** (`tiny`, ~75MB) is downloaded automatically on first run by faster-whisper

5. **Run:**

   ```bash
   uv run kian
   ```

## MLC Backend (larger model via Docker)

The MLC backend runs Qwen3-4B inside a Docker container, which is more memory-efficient than llama.cpp for larger models on Jetson's shared 8GB RAM.

1. **Pre-pull the container image** (~7GB, one-time):

   ```bash
   sudo docker pull dustynv/mlc:0.20.0-r36.4.0
   ```

2. **Run with MLC:**

   ```bash
   uv run kian --backend mlc
   ```

   On first run, the Qwen3-4B model (~2.3GB) is downloaded automatically inside the container. The model cache is persisted in `models/hf-cache/`.

   The container is started and stopped automatically by kian.

## Usage

```bash
# Default: Qwen3.5-2B via llama.cpp
uv run kian

# Qwen3-4B via MLC (Docker)
uv run kian --backend mlc

# Custom model with llama.cpp
uv run kian --model models/some-other-model.gguf

# Custom model with MLC
uv run kian --backend mlc --model "HF://mlc-ai/some-other-model-MLC"
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
│   ├── llm_mlc.py      # MLC Docker backend (Qwen3-4B)
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
3. **LLM** streams a response token-by-token (via llama.cpp or MLC container)
4. Tokens are buffered and split on punctuation boundaries (~25 chars min)
5. Each text fragment is synthesized by **Piper TTS** and queued for playback
6. A background thread plays audio chunks sequentially, so playback starts while the LLM is still generating

## Models

Models are stored in `models/` and excluded from git. You can swap them:

- **LLM (llama.cpp):** Any GGUF model works. Pass `--model path/to/model.gguf`.
- **LLM (MLC):** Pass `--model "HF://org/model-MLC"` with any MLC-compiled model.
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

## Memory Budget (~8GB)

| Component | RAM |
|-----------|-----|
| OS/system | ~1.5 GB |
| Qwen3.5-2B Q4_K_M (llamacpp) | ~1.5 GB |
| Whisper tiny | ~0.1 GB |
| Piper TTS | ~0.1 GB |
| KV cache + buffers | ~1-2 GB |
| **Headroom** | **~3-4 GB** |

With the MLC backend, the Docker container manages its own GPU memory for the larger Qwen3-4B model.
