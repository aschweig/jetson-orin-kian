#!/usr/bin/env python3
"""Encode naughty.txt -> naughty.txt with ROT-13 on content lines.

Run before committing. Comments and blank lines are preserved as-is.
"""

import codecs
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "naughty.tmp"
DST = ROOT / "naughty.txt"

lines = SRC.read_text().splitlines()
out = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        out.append(line)
    else:
        out.append(codecs.encode(line, "rot_13"))

DST.write_text("\n".join(out) + "\n")
print(f"Encoded {SRC} -> {DST}")
