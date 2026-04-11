#!/usr/bin/env python3
"""
RTL_433 — 433 MHz ISM band decoder
Decodes weather stations, temperature/humidity sensors, and other devices

Использование:
    python rtl433_scanner.py                   # 60s scan на 433.92 MHz
    python rtl433_scanner.py --duration 120    # 120 секунд
    python rtl433_scanner.py --freq 868.3      # 868.3 MHz (EU ISM)
    python rtl433_scanner.py --gain 50         # Усиление 50 dB
    python rtl433_scanner.py --summary         # Показать сводку
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Пути к rtl_433 (Windows)
RTL_433_PATHS = [
    r"M:\GitHub\nanoprobe-sim-lab\tools\rtl_433\rtl_433.exe",
    r"C:\rtl_433\rtl_433.exe",
    r"C:\Program Files\rtl_433\rtl_433.exe",
    "rtl_433",  # Linux/Mac (в PATH)
]


def find_rtl433():
    """Найти исполняемый файл rtl_433"""
    for path in RTL_433_PATHS:
        if (
            os.path.isfile(path)
            or subprocess.run(
                ["where" if os.name == "nt" else "which", path],
                capture_output=True,
            ).returncode
            == 0
        ):
            return path
    return None


def scan_433_mhz(
    frequency_mhz: float = 433.92,
    gain: int = 40,
    duration: int = 60,
    output_dir: str = None,
):
    """
    Сканирование ISM band на частоте 433 MHz

    Args:
        frequency_mhz: Частота сканирования (MHz)
        gain: Усиление RTL-SDR (dB)
        duration: Длительность сканирования (секунды)
        output_dir: Директория для выходных файлов
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "data" / "rtl433")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"rtl433_{timestamp}.jsonl")
    log_file = os.path.join(output_dir, f"rtl433_{timestamp}.log")

    rtl433_exe = find_rtl433()
    if not rtl433_exe:
        print("❌ rtl_433 не найден!")
        print("\nУстановка:")
        print("  Windows: https://github.com/merbanan/rtl_433/releases")
        print("  Linux:   sudo apt install rtl-433")
        print("  Mac:     brew install rtl_433")
        sys.exit(1)

    frequency_hz = int(frequency_mhz * 1_000_000)

    print(f"RTL_433 — {frequency_mhz:.2f} MHz ISM Band Scanner")
    print(f"Date: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: RTL-SDR Blog V4")
    print(f"Frequency: {frequency_mhz:.2f} MHz ({frequency_hz} Hz)")
    print(f"Gain: {gain} dB")
    print(f"Duration: {duration}s")
    print(f"Output: {json_file}")
    print()

    cmd = [
        rtl433_exe,
        "-f",
        str(frequency_hz),
        "-g",
        str(gain),
        "-F",
        f"json:{json_file}",
        "-F",
        "log",
        "-C",
        "customary",
    ]

    print(f"[*] Запуск {rtl433_exe}...")
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        devices_found = []
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()

                # Parse JSON lines (device data)
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        if "model" in data:
                            devices_found.append(data)
                            model = data.get("model", "unknown")
                            devid = data.get("id", "?")
                            print(f"  [DEVICE] {model} (id={devid})")
                    except json.JSONDecodeError:
                        pass

            proc.wait(timeout=duration + 10)
        except subprocess.TimeoutExpired:
            print(f"\n[*] Timeout reached ({duration}s), stopping...")
            proc.kill()
            proc.wait()

    print(f"\n{'='*60}")
    print(f"[+] Scan complete!")
    print(f"    Devices found: {len(devices_found)}")
    print(f"    Log: {log_file}")
    print(f"    JSON: {json_file}")

    # Summary
    if devices_found:
        print(f"\n[*] Device Summary:")
        models = {}
        for d in devices_found:
            model = d.get("model", "unknown")
            models[model] = models.get(model, 0) + 1

        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
            print(f"    {model}: {count} messages")

        # Show last message details for each device
        print(f"\n[*] Latest Data:")
        shown = set()
        for d in devices_found:
            devid = f"{d.get('model')}_{d.get('id')}"
            if devid not in shown:
                shown.add(devid)
                print(f"\n  Model: {d.get('model')}")
                print(f"  ID: {d.get('id')}")
                if "temperature_C" in d:
                    print(f"  Temperature: {d['temperature_C']}°C")
                if "humidity" in d:
                    print(f"  Humidity: {d['humidity']}%")
                if "battery_ok" in d:
                    bat_status = "OK" if d["battery_ok"] else "LOW"
                    print(f"  Battery: {bat_status}")
                if "channel" in d:
                    print(f"  Channel: {d['channel']}")
    else:
        print(f"\n[-] No devices detected")
        print(f"    Possible reasons:")
        print(f"    - No {frequency_mhz:.1f} MHz devices in range")
        print(f"    - Try increasing gain (-g 50)")
        print(f"    - Try scanning multiple frequencies:")
        print(f"      rtl_433 -f 433920000 " f"-f 868300000 -f 915000000")

    return devices_found


def main():
    parser = argparse.ArgumentParser(
        description="RTL_433 ISM Band Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python rtl433_scanner.py                   # 60s на 433.92 MHz
  python rtl433_scanner.py --duration 120    # 120 секунд
  python rtl433_scanner.py --freq 868.3      # 868.3 MHz (EU)
  python rtl433_scanner.py --gain 50         # Усиление 50 dB
        """,
    )

    parser.add_argument(
        "--freq",
        type=float,
        default=433.92,
        help="Частота сканирования MHz (по умолч.: 433.92)",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=40,
        help="Усиление RTL-SDR dB (по умолч.: 40)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Длительность сек (по умолч.: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Директория вывода",
    )

    args = parser.parse_args()

    scan_433_mhz(
        frequency_mhz=args.freq,
        gain=args.gain,
        duration=args.duration,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
