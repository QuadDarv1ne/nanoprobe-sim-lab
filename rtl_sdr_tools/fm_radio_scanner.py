#!/usr/bin/env python3
"""
FM Radio Scanner using rtl_fm (RTL-SDR Blog V4)
Scans FM band (88-108 MHz), finds strongest stations, and captures audio
"""

import os
import subprocess
from datetime import datetime

import numpy as np

RTL_FM_PATH = r"M:\GitHub\nanoprobe-sim-lab\tools\rtl-sdr-blog\x64\rtl_fm.exe"
DURATION_SEC = 10  # Duration to capture each station
GAIN = "40"  # dB
SAMPLE_RATE = "32k"  # Audio sample rate
OUTPUT_DIR = r"M:\GitHub\nanoprobe-sim-lab\fm_stations"

# FM band frequencies in Russia (common stations)
FM_STATIONS = {
    87.5: "Band Edge",
    88.3: "Radio Monte Carlo",
    89.1: "Radio Kultura",
    90.1: "Radio Mayak",
    91.2: "DFM",
    91.8: "Radio Dacha",
    92.4: "Retro FM",
    93.1: "Nashe Radio",
    93.6: "Keks FM",
    94.0: "Radio Record",
    94.5: "Studio 21",
    95.0: "Like FM",
    95.5: "Radio Energy",
    96.0: "Radio Relax FM",
    96.6: "Voyage FM",
    97.0: "Radio 7",
    97.5: "Romantika",
    98.0: "Average FM",
    98.8: "Radio Orpheus",
    99.2: "Radio Jazz",
    99.6: "Hit FM",
    100.1: "Avers",
    100.5: "Europa Plus",
    100.9: "Hit FM",
    101.1: "DFM",
    101.5: "Russian Radio",
    101.9: "Marusya FM",
    102.3: "Radio Maximum",
    102.7: "Radio Romantika",
    103.1: "Radio Monte Carlo",
    103.4: "Business FM",
    103.7: "Nashe Radio",
    104.2: "Ultra",
    104.7: "Radio Chanson",
    105.3: "Autorskoe Radio",
    105.7: "Radio Dacha",
    106.2: "Comedy Radio",
    106.6: "Pioneer FM",
    107.0: "Radio Kultura",
    107.4: "Radio Mayak",
    107.8: "Vesti FM",
    108.0: "Band Edge",
}


def scan_fm_with_rtl_power():
    """Use rtl_power to scan FM band and find strongest signals"""
    rtl_power = RTL_FM_PATH.replace("rtl_fm.exe", "rtl_power.exe")
    if not os.path.exists(rtl_power):
        print(f"[-] rtl_power.exe not found at {rtl_power}")
        return []

    print(f"\n{'='*60}")
    print(f"=== FM Band Scan (88-108 MHz) using rtl_power ===")
    print(f"{'='*60}")

    output_file = os.path.join(os.path.dirname(RTL_FM_PATH), "fm_scan.csv")

    # Scan FM band: 88-108 MHz, 100kHz steps, 1 second integration
    cmd = [
        rtl_power,
        "-f",
        "88M:108M:100k",
        "-g",
        GAIN,
        "-i",
        "5",  # 5 seconds integration
        "-1",  # Single sweep
        output_file,
    ]

    print(f"[*] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"[+] Scan complete")
            return parse_rtl_power_output(output_file)
        else:
            print(f"[-] rtl_power failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"[-] Scan timeout")
    except Exception as e:
        print(f"[-] Error: {e}")

    return []


def parse_rtl_power_output(csv_file):
    """Parse rtl_power CSV output and find strongest frequencies"""
    if not os.path.exists(csv_file):
        print(f"[-] CSV file not found: {csv_file}")
        return []

    results = []
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split(",")
            if len(parts) >= 7:
                try:
                    freq_start = float(parts[0])
                    freq_stop = float(parts[1])
                    freq_center = (freq_start + freq_stop) / 2
                    power_db = float(parts[6]) if len(parts) > 6 else -100

                    results.append((freq_center, power_db))
                except (ValueError, IndexError):
                    pass

    # Sort by power (strongest first)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n[*] Top 10 strongest frequencies:")
    for freq, power in results[:10]:
        # Find nearest station
        nearest = min(FM_STATIONS.keys(), key=lambda f: abs(f - freq / 1e6))
        station = FM_STATIONS[nearest]
        print(f"  {freq/1e6:6.2f} MHz: {'█' * int((power+60)*2):20s} {power:.1f} dB  ({station})")

    return results[:10]


def capture_fm_station(freq_mhz, station_name, duration=DURATION_SEC):
    """Capture audio from a specific FM station using rtl_fm"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    freq = f"{freq_mhz:.3f}M"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        OUTPUT_DIR, f"fm_{station_name.replace(' ', '_').lower()}_{timestamp}.wav"
    )

    print(f"\n{'='*60}")
    print(f"=== Capturing: {station_name} @ {freq_mhz:.1f} MHz ===")
    print(f"{'='*60}")

    # rtl_fm command for WBFM (Wideband FM for broadcast radio)
    cmd = [
        RTL_FM_PATH,
        "-f",
        freq,
        "-M",
        "wbfm",
        "-s",
        SAMPLE_RATE,
        "-g",
        GAIN,
        "-l",
        "0",  # No squelch
        "-E",
        "deemp",  # De-emphasis
        "-d",
        "0",  # Device index 0
    ]

    print(f"[*] Running: {' '.join(cmd)}")
    print(f"[*] Duration: {duration}s")
    print(f"[*] Output: {output_file}")

    try:
        # Run for specified duration
        result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)

        if result.stdout:
            # Save raw audio data
            with open(output_file.replace(".wav", ".raw"), "wb") as f:
                f.write(result.stdout)

            # Convert to WAV if possible (simple conversion)
            print(f"[+] Captured {len(result.stdout)} bytes of audio")

            # Analyze audio
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
            if len(audio_data) > 0:
                print(f"[*] Audio Analysis:")
                print(f"    Samples: {len(audio_data)}")
                print(f"    Duration: {len(audio_data)/32000:.2f}s")
                print(f"    Mean: {np.mean(audio_data):.1f}")
                print(f"    RMS: {np.sqrt(np.mean(audio_data.astype(float)**2)):.1f}")
                print(f"    Max: {np.max(np.abs(audio_data))}")
                print(f"    Clipping: {np.sum(np.abs(audio_data) >= 32767)} samples")

            return True
        else:
            print(f"[-] No audio captured")
            return False

    except subprocess.TimeoutExpired:
        print(f"[-] Capture timeout")
        return False
    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def capture_all_known_stations(max_stations=5):
    """Try to capture from known FM stations"""
    print(f"\n{'='*60}")
    print(f"=== FM Station Capture ===")
    print(f"{'='*60}")

    # Try top stations
    for freq_mhz, station_name in list(FM_STATIONS.items())[:max_stations]:
        try:
            print(f"\n[*] Trying {station_name} @ {freq_mhz:.1f} MHz...")
            success = capture_fm_station(freq_mhz, station_name, duration=DURATION_SEC)
            if success:
                print(f"[+] Successfully captured {station_name}")
            else:
                print(f"[-] Failed to capture {station_name}")
        except KeyboardInterrupt:
            print(f"\n[*] Interrupted by user")
            break
        except Exception as e:
            print(f"[-] Error with {station_name}: {e}")


def main():
    print(f"FM Radio Scanner using RTL-SDR Blog V4")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: RTL-SDR Blog V4 (R828D)")
    print(f"Gain: {GAIN} dB")

    # Option 1: Scan FM band with rtl_power
    try:
        scan_results = scan_fm_with_rtl_power()

        if scan_results:
            print(f"\n[*] Capturing strongest frequency...")
            best_freq = scan_results[0][0] / 1e6
            capture_fm_station(best_freq, f"FM_{best_freq:.1f}MHz")
    except KeyboardInterrupt:
        print(f"\n[*] Scan interrupted")

    # Option 2: Capture known stations
    try:
        capture_all_known_stations(max_stations=3)
    except KeyboardInterrupt:
        print(f"\n[*] Capture interrupted")

    print(f"\n{'='*60}")
    print(f"[+] FM Radio scan complete!")
    print(f"    Audio files saved to: {OUTPUT_DIR}")
    print(f"    To listen: Convert .raw to .wav with:")
    print(f"    sox -t raw -r 32k -e signed -b 16 -c 1 input.raw output.wav")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
