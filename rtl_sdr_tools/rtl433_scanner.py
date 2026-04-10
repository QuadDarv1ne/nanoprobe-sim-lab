#!/usr/bin/env python3
"""
RTL_433 — 433 MHz ISM band decoder
Decodes weather stations, temperature/humidity sensors, and other devices
"""
import subprocess
import json
import os
from datetime import datetime

RTL_433 = r"M:\GitHub\nanoprobe-sim-lab\tools\rtl_433\rtl_433-rtlsdr.exe"
DURATION = 60  # seconds to listen
OUTPUT_DIR = r"M:\GitHub\nanoprobe-sim-lab\data\rtl433"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_file = os.path.join(OUTPUT_DIR, f"rtl433_{timestamp}.jsonl")
log_file = os.path.join(OUTPUT_DIR, f"rtl433_{timestamp}.log")

print(f"RTL_433 — 433 MHz ISM Band Decoder")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: RTL-SDR Blog V4")
print(f"Frequency: 433.92 MHz")
print(f"Duration: {DURATION}s")
print(f"Output: {json_file}")
print()

cmd = [
    RTL_433,
    "-f", "433920000",
    "-g", "40",
    "-F", f"json:{json_file}",
    "-F", "log",
]

print(f"[*] Starting rtl_433...")
with open(log_file, 'w', encoding='utf-8') as log:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    devices_found = []
    try:
        for line in proc.stdout:
            log.write(line)
            log.flush()

            # Parse JSON lines (device data)
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    if 'model' in data:
                        devices_found.append(data)
                        model = data.get('model', 'unknown')
                        devid = data.get('id', '?')
                        print(f"  [DEVICE] {model} (id={devid})")
                except json.JSONDecodeError:
                    pass

        proc.wait(timeout=DURATION + 10)
    except subprocess.TimeoutExpired:
        print(f"\n[*] Timeout reached ({DURATION}s), stopping...")
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
        model = d.get('model', 'unknown')
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
            if 'temperature_C' in d:
                print(f"  Temperature: {d['temperature_C']}°C")
            if 'humidity' in d:
                print(f"  Humidity: {d['humidity']}%")
            if 'battery_ok' in d:
                print(f"  Battery: {'OK' if d['battery_ok'] else 'LOW'}")
            if 'channel' in d:
                print(f"  Channel: {d['channel']}")
else:
    print(f"\n[-] No devices detected")
    print(f"    Possible reasons:")
    print(f"    - No 433 MHz devices in range")
    print(f"    - Try increasing gain (-g 50)")
    print(f"    - Try scanning multiple frequencies:")
    print(f"      rtl_433 -f 433920000 -f 868300000 -f 915000000")
