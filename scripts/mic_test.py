#!/usr/bin/env python3
"""Record 3 seconds from the mic, print RMS, play it back. Loop forever."""

import numpy as np
import sounddevice as sd

RATE = 48000
DURATION = 3.0

# List input devices
print("Input devices:")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        default = " <-- DEFAULT" if i == sd.default.device[0] else ""
        print(f"  [{i}] {d['name']} (ch={d['max_input_channels']}, rate={d['default_samplerate']}){default}")
print()

# Try each input device
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] == 0:
        continue
    if "monitor" in d["name"].lower() or "APE" in d["name"]:
        continue
    print(f"Testing device [{i}] {d['name']}...")
    try:
        audio = sd.rec(int(RATE * 0.5), samplerate=RATE,
                       channels=1, dtype="float32", blocking=True, device=i)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        print(f"  RMS={rms:.4f}  peak={peak:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

# Pick the device with highest RMS for the main loop
best_dev = None
best_rms = 0
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] == 0 or "monitor" in d["name"].lower() or "APE" in d["name"]:
        continue
    try:
        audio = sd.rec(int(RATE * 0.3), samplerate=RATE,
                       channels=1, dtype="float32", blocking=True, device=i)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_dev = i
    except:
        pass

if best_dev is not None:
    print(f"Using device [{best_dev}] {sd.query_devices(best_dev)['name']} (loudest RMS={best_rms:.4f})")
else:
    best_dev = sd.default.device[0]
    print(f"Using default device [{best_dev}]")
print()

print(f"Record {DURATION}s, playback, repeat. Ctrl-C to quit.")
print()

while True:
    print("Recording... ", end="", flush=True)
    audio = sd.rec(int(RATE * DURATION), samplerate=RATE,
                   channels=1, dtype="float32", blocking=True, device=best_dev)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"done. RMS={rms:.4f}  peak={peak:.4f}")

    print("Playing back... ", end="", flush=True)
    sd.play(audio, samplerate=RATE)
    sd.wait()
    print("done.")
    print()
