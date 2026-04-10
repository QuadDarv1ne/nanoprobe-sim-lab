#!/usr/bin/env python3
"""Capture multiple FM stations"""
import os
import subprocess
import wave
from datetime import datetime

import numpy as np

RTL_FM = r"M:\GitHub\nanoprobe-sim-lab\tools\rtl-sdr-blog\x64\rtl_fm.exe"
DURATION = 8  # seconds per station
GAIN = "40"

# Known Moscow FM stations (strongest from scan)
STATIONS = [
    ("106.0M", "FM_106.0"),
    ("100.5M", "Europa_Plus"),
    ("101.5M", "Russian_Radio"),
]


def capture_and_convert(freq, name):
    """Capture FM and convert to WAV"""
    raw_file = (
        f"M:\\GitHub\\nanoprobe-sim-lab\\fm_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw"
    )
    wav_file = raw_file.replace(".raw", ".wav")

    print(f"\n{'='*50}")
    print(f"Capturing: {name} @ {freq}")
    print(f"{'='*50}")

    cmd = [RTL_FM, "-f", freq, "-M", "wbfm", "-s", "32k", "-g", GAIN, "-E", "deemp"]

    with open(raw_file, "wb") as outfile:
        proc = subprocess.Popen(cmd, stdout=outfile, stderr=subprocess.PIPE)
        try:
            proc.wait(timeout=DURATION + 5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Convert to WAV
    if os.path.exists(raw_file) and os.path.getsize(raw_file) > 0:
        audio = np.fromfile(raw_file, dtype="<i2")
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = (audio / max_val * 32767).astype(np.int16)

            with wave.open(wav_file, "w") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(32000)
                wav.writeframes(audio.tobytes())

            print(f"✓ {name}: {len(audio)/32000:.1f}s, {os.path.getsize(wav_file)/1024:.1f} KB")
            return True

    print(f"✗ {name}: Failed")
    return False


print(f"FM Radio Multi-Station Capture")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: RTL-SDR Blog V4")

for freq, name in STATIONS:
    try:
        capture_and_convert(freq, name)
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        break
    except Exception as e:
        print(f"Error with {name}: {e}")

print(f"\n{'='*50}")
print(f"Capture complete!")
print(f"Files saved to: M:\\GitHub\\nanoprobe-sim-lab\\fm_*.wav")
print(f"To listen: Open .wav files with VLC or Media Player")
print(f"{'='*50}")
