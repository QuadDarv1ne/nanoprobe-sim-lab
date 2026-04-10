#!/usr/bin/env python3
"""Convert raw rtl_fm audio to WAV"""
import sys
import wave

import numpy as np

RAW_FILE = sys.argv[1] if len(sys.argv) > 1 else None
if not RAW_FILE:
    print("Usage: python raw_to_wav.py <input.raw>")
    sys.exit(1)

WAV_FILE = RAW_FILE.replace(".raw", ".wav")
SAMPLE_RATE = 32000  # rtl_fm wbfm default

print(f"Converting: {RAW_FILE}")

# Read raw 16-bit signed little-endian audio
audio = np.fromfile(RAW_FILE, dtype="<i2")
print(f"Read {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")

# Normalize
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = (audio / max_val * 32767).astype(np.int16)

# Write WAV
with wave.open(WAV_FILE, "w") as wav:
    wav.setnchannels(1)  # Mono
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(audio.tobytes())

print(f"Saved: {WAV_FILE}")
print(f"Size: {len(audio.tobytes())/1024:.1f} KB")
