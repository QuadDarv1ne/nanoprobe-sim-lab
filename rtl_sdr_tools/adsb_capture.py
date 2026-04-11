#!/usr/bin/env python3
"""
ADS-B (1090 MHz) data capture using RTL-SDR V4
Decodes Mode-S transponder signals from aircraft
"""

from datetime import datetime, timezone

import numpy as np
from rtlsdr import RtlSdr

# ADS-B parameters
FREQUENCY = 1090e6  # 1090 MHz - ADS-B frequency
SAMPLE_RATE = 2.4e6  # 2.4 MS/s
GAIN = 40.0  # Manual gain for better pulse detection
DURATION_SEC = 10  # Capture duration

# ADS-B pulse detection thresholds
PULSE_THRESHOLD = 0.5  # Normalized amplitude
MIN_PULSE_WIDTH = int(SAMPLE_RATE * 0.5e-6)  # 0.5 µs
MAX_PULSE_WIDTH = int(SAMPLE_RATE * 1.5e-6)  # 1.5 µs


def detect_adsb_pulses(samples):
    """Detect ADS-B 1090ES pulses"""
    # Calculate envelope (magnitude)
    magnitude = np.abs(samples)

    # Normalize
    magnitude = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude

    # Simple threshold detection
    pulses = []
    in_pulse = False
    pulse_start = 0

    for i, amp in enumerate(magnitude):
        if amp > PULSE_THRESHOLD and not in_pulse:
            in_pulse = True
            pulse_start = i
        elif amp < PULSE_THRESHOLD * 0.7 and in_pulse:
            in_pulse = False
            pulse_width = i - pulse_start
            if MIN_PULSE_WIDTH <= pulse_width <= MAX_PULSE_WIDTH:
                pulses.append(
                    {
                        "start": pulse_start,
                        "width": pulse_width,
                        "amplitude": np.max(magnitude[pulse_start:i]),
                        "time_offset": pulse_start / SAMPLE_RATE,
                    }
                )

    return pulses


def decode_mode_s_address(pulses):
    """Extract ICAO address from Mode-S pulses (simplified)"""
    # Mode-S has 56 or 112 bits with 8 µs preamble
    # This is a simplified detection - full decoding requires df* library
    if len(pulses) < 10:
        return None

    # Look for preamble pattern (4 pulses: 0, 1, 3.5, 4.5 µs)
    # Simplified: just detect pulse train presence
    aircraft_count = len(pulses) // 14  # Mode-S has ~14 pulses per message
    return aircraft_count


def main():
    print(f"=== ADS-B Capture @ {FREQUENCY/1e6:.1f} MHz ===")
    print(f"Device: RTL-SDR V4")
    print(f"Sample Rate: {SAMPLE_RATE/1e6:.1f} MS/s")
    print(f"Gain: {GAIN:.1f} dB")
    print(f"Duration: {DURATION_SEC}s")
    print()

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = FREQUENCY
    sdr.gain = GAIN

    print(f"[*] Capturing {DURATION_SEC} seconds...")
    total_samples = int(SAMPLE_RATE * DURATION_SEC)
    samples = sdr.read_samples(total_samples)
    sdr.close()

    print(f"[+] Captured {len(samples)} I/Q samples ({len(samples)/SAMPLE_RATE:.2f}s)")
    print(f"[*] Signal strength: {np.mean(np.abs(samples)):.4f} ± {np.std(np.abs(samples)):.4f}")

    # Detect pulses
    print(f"\n[*] Detecting ADS-B pulses...")
    pulses = detect_adsb_pulses(samples)

    if not pulses:
        print("[-] No ADS-B pulses detected")
        print("    Possible reasons:")
        print("    - No aircraft in range")
        print("    - Gain needs adjustment")
        print("    - Try increasing capture duration")
        return

    print(f"[+] Found {len(pulses)} pulses")

    # Show first 10 pulses
    print(f"\n[*] First 10 pulses:")
    for i, pulse in enumerate(pulses[:10]):
        print(
            f"  #{i+1:2d}: t={pulse['time_offset']*1000:.1f}ms, "
            f"width={pulse['width']/SAMPLE_RATE*1e6:.2f}µs, "
            f"amp={pulse['amplitude']:.3f}"
        )

    # Estimate aircraft count
    aircraft_estimate = decode_mode_s_address(pulses)
    if aircraft_estimate and aircraft_estimate > 0:
        print(f"\n[~] Estimated aircraft: {aircraft_estimate}")
    else:
        print(f"\n[~] For full aircraft tracking, install dump1090:")
        print(f"    - Download: https://github.com/flightaware/dump1090")
        print(f"    - Or use: https://github.com/wiedehopf/readsb")

    # Save to file for further analysis
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    np.save(f"adsb_samples_{timestamp}.npy", samples)
    print(f"\n[+] Saved raw I/Q to: adsb_samples_{timestamp}.npy")

    # Pulse statistics
    pulse_widths = [p["width"] / SAMPLE_RATE * 1e6 for p in pulses]
    print(f"\n[*] Pulse Statistics:")
    print(f"    Width: {np.mean(pulse_widths):.2f} ± {np.std(pulse_widths):.2f} µs")
    amps = [p["amplitude"] for p in pulses]
    print(f"    Amplitude: {np.mean(amps):.3f} ± {np.std(amps):.3f}")


if __name__ == "__main__":
    main()
