#!/usr/bin/env python3
"""Decode naughty.txt -> naughty.tmp with ROT-13 on content lines.

Run after checkout to get the editable plaintext version.
"""

import codecs
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "naughty.txt"
DST = ROOT / "naughty.tmp"

lines = SRC.read_text().splitlines()
out = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        out.append(line)
    else:
        out.append(codecs.encode(line, "rot_13"))

DST.write_text("\n".join(out) + "\n")
print(f"Decoded {SRC} -> {DST}")
