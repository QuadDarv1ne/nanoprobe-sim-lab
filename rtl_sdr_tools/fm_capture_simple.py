#!/usr/bin/env python3
"""
⚠️  DEPRECATED: Используйте fm_radio_unified.py capture
"""
import os
import subprocess
from datetime import datetime

RTL_FM = r"M:\GitHub\nanoprobe-sim-lab\tools\rtl-sdr-blog\x64\rtl_fm.exe"
FREQUENCY = "106.0M"  # Strongest station from earlier scan
DURATION = 10  # seconds
OUTPUT_FILE = (
    f"M:\\GitHub\\nanoprobe-sim-lab\\fm_106MHz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw"
)

print(f"Capturing FM radio at {FREQUENCY} for {DURATION}s...")
print(f"Output: {OUTPUT_FILE}")

cmd = [
    RTL_FM,
    "-f",
    FREQUENCY,
    "-M",
    "wbfm",
    "-s",
    "32k",
    "-g",
    "40",
    "-E",
    "deemp",
]

with open(OUTPUT_FILE, "wb") as outfile:
    proc = subprocess.Popen(cmd, stdout=outfile, stderr=subprocess.PIPE)
    try:
        proc.wait(timeout=DURATION + 5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

size = os.path.getsize(OUTPUT_FILE)
print(f"Captured {size:,} bytes ({size/1024:.1f} KB)")
print(f"Duration: {size / (32000 * 2 * 1):.2f}s")  # 32kHz, 16-bit, mono
print(f"To convert to WAV: sox -t raw -r 32k -e signed -b 16 -c 1 {OUTPUT_FILE} output.wav")
