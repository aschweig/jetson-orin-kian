# Kian

A local voice assistant running entirely on-device. Designed for NVIDIA Jetson Orin (8GB), but works on any Linux machine with a microphone.

**Pipeline:** Mic → WebRTC VAD → faster-whisper (STT) → Qwen3.5-4B via llama.cpp (LLM) → Piper (TTS) → Speaker

All components run locally. No cloud APIs required.

## Requirements

- Linux (tested on Jetson Orin, Ubuntu 22.04, JetPack 6.1)
- Python 3.12+ (managed by uv)
- [uv](https://docs.astral.sh/uv/) package manager
- A microphone and speakers/headphones
- ~4GB disk for models

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
   - **Piper TTS** voice (`en_US-lessac-medium`, ~75MB) into `models/`
   - **Qwen3.5-4B** GGUF (`Q4_K_M`, ~2.7GB) into `models/`
   - **Whisper** (`tiny`, ~75MB) is downloaded automatically on first run by faster-whisper

5. **Run:**

   ```bash
   uv run kian
   ```

## Project Structure

```
kian/
├── kian/
│   ├── app.py     # async pipeline wiring everything together
│   ├── vad.py     # voice activity detection (webrtcvad)
│   ├── stt.py     # speech-to-text (faster-whisper)
│   ├── llm.py     # LLM inference (llama-cpp-python)
│   └── tts.py     # text-to-speech + playback thread (Piper)
├── models/        # model files (not checked in, see above)
├── scripts/
│   └── download-models.sh
└── pyproject.toml
```

## How It Works

1. **VAD** continuously listens on the mic and detects speech segments
2. **Whisper** transcribes each speech segment to text
3. **Qwen3.5-4B** streams a response token-by-token
4. Tokens are buffered and split on punctuation boundaries (~25 chars min)
5. Each text fragment is synthesized by **Piper TTS** and queued for playback
6. A background thread plays audio chunks sequentially, so playback starts while the LLM is still generating

## Models

Models are stored in `models/` and excluded from git. You can swap them:

- **LLM:** Any GGUF model works. Edit `DEFAULT_MODEL_PATH` in `kian/llm.py`.
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
| Qwen3.5-4B Q4_K_M | ~2.7 GB |
| Whisper tiny | ~0.1 GB |
| Piper TTS | ~0.1 GB |
| KV cache + buffers | ~1-2 GB |
| **Headroom** | **~1.5-2 GB** |
