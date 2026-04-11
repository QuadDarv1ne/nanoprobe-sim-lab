"""
Конвертер сырых I/Q данных в WAV для MMSSTV
Записывает сигнал с МКС и конвертирует в формат, понятный MMSSTV
"""

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

FREQUENCY = 145.800  # МКС SSTV
DURATION = 60  # секунд (SSTV передача обычно 60-120 сек)
GAIN = 40  # dB
SAMPLE_RATE = 22050  # Hz (стандарт для SSTV)

OUTPUT_DIR = Path("data/sstv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_wav = OUTPUT_DIR / f"iss_sstv_{timestamp}.wav"

print("=" * 60)
print("MMSSTV SSTV CAPTURE")
print("=" * 60)
print(f"Frequency: {FREQUENCY} MHz")
print(f"Gain: {GAIN} dB")
print(f"Duration: {DURATION} seconds")
print(f"Output: {output_wav}")
print("=" * 60)
print()

rtl_fm_path = Path("rtl_fm.exe")
if not rtl_fm_path.exists():
    rtl_fm_path = Path(r"C:\rtl-sdr\bin\x64\rtl_fm.exe")

mmsstv_path = Path(r"C:\Ham\MMSSTV\MMSSTV.EXE")

if not rtl_fm_path.exists():
    print("ERROR: rtl_fm.exe not found!")
    print("Place it in project root or C:\\rtl-sdr\\bin\\x64\\")
    exit(1)

print(f"Using: {rtl_fm_path}")
print()

# rtl_fm пишет сырые 16-bit signed int в stdout
# MMSSTV понимает WAV 22050 Hz 16-bit mono

cmd = [
    str(rtl_fm_path),
    "-f",
    f"{FREQUENCY}M",
    "-M",
    "fm",
    "-s",
    str(SAMPLE_RATE),
    "-r",
    str(SAMPLE_RATE),
    "-g",
    str(GAIN),
    "-d",
    "0",
    "-l",
    "0",  # squelch off
]

print(f"Command: {' '.join(cmd)} > {output_wav}")
print(f"Recording {DURATION} seconds...")
print("Press Ctrl+C to stop early")
print()

start_time = time.time()

with open(output_wav, "wb") as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)

    try:
        # Показываем прогресс
        for i in range(DURATION):
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            print(f"\r  [{elapsed}/{DURATION}s]", end="", flush=True)

        # Время вышло - останавливаем
        print(f"\n  Stopping...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
            process.wait()

    except KeyboardInterrupt:
        print(f"\n  User stopped!")
        process.kill()
        process.wait()

# Проверка файла
if output_wav.exists() and output_wav.stat().st_size > 0:
    size_mb = output_wav.stat().st_size / (1024 * 1024)
    print(f"\nOK! File: {output_wav}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"\nNEXT STEP - Decode with MMSSTV:")
    print(f"1. Open MMSSTV: {mmsstv_path}")
    print(f"2. Menu -> RxAudio -> Load File -> {output_wav}")
    print(f"3. MMSSTV will auto-detect SSTV mode and decode")

    # Автооткрытие MMSSTV (опционально)
    if mmsstv_path.exists():
        print(f"\nMMSSTV found! Opening...")
        subprocess.Popen([str(mmsstv_path), str(output_wav)])
else:
    print(f"\nERROR: File not created or empty!")
    print("Check if RTL-SDR is connected and not in use by another app")
