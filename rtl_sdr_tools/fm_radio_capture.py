#!/usr/bin/env python3
"""
⚠️  DEPRECATED: Используйте fm_radio_unified.py capture
"""

import wave
from datetime import datetime, timezone

import numpy as np
from rtlsdr import RtlSdr

# FM Radio parameters
FREQUENCIES = {
    "Europa Plus": 100.5e6,
    "Russian Radio": 101.5e6,
    "DFM": 101.1e6,
    "Hit FM": 100.9e6,
    "Average": 98.0e6,  # Try this if specific stations unknown
}

SAMPLE_RATE = 2.4e6  # 2.4 MS/s (raw I/Q)
AUDIO_RATE = 44100  # Standard audio sample rate
GAIN = 30.0  # dB
DURATION_SEC = 5  # Capture duration
OUTPUT_FILE = "fm_radio_audio.wav"


def fm_demodulate(samples, audio_rate, sample_rate):
    """FM demodulation using quadrature detector"""
    # Calculate phase difference between consecutive samples
    phase = np.angle(samples)
    phase_diff = np.diff(phase)

    # Unwrap phase to avoid discontinuities
    phase_diff = np.unwrap(phase_diff)

    # Demodulated signal is proportional to frequency deviation
    audio = phase_diff * audio_rate / (2 * np.pi)

    # Decimate to audio sample rate
    decimation = int(sample_rate / audio_rate)
    audio_decimated = audio[::decimation]

    # Normalize to 16-bit range
    max_val = np.max(np.abs(audio_decimated))
    if max_val > 0:
        audio_decimated = audio_decimated / max_val * 0.9

    return audio_decimated


def save_wav(filename, audio_data, sample_rate):
    """Save audio to WAV file"""
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"[+] Audio saved to: {filename}")
    print(f"    Duration: {len(audio_data)/sample_rate:.2f}s")
    print(f"    Sample rate: {sample_rate} Hz")
    print(f"    Size: {len(audio_int16)} samples ({len(audio_int16)*2/1024:.1f} KB)")


def capture_fm(frequency, station_name="Unknown"):
    """Capture and demodulate FM radio station"""
    print(f"\n{'='*50}")
    print(f"=== FM Radio: {station_name} @ {frequency/1e6:.1f} MHz ===")
    print(f"{'='*50}")
    print(f"Device: RTL-SDR V4")
    print(f"Sample Rate: {SAMPLE_RATE/1e6:.1f} MS/s")
    print(f"Gain: {GAIN:.1f} dB")
    print(f"Duration: {DURATION_SEC}s")
    print()

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = frequency
    sdr.gain = GAIN

    print(f"[*] Capturing I/Q data...")
    total_samples = int(SAMPLE_RATE * DURATION_SEC)
    samples = sdr.read_samples(total_samples)
    sdr.close()

    signal_strength = np.mean(np.abs(samples))
    print(f"[+] Captured {len(samples)} samples ({len(samples)/SAMPLE_RATE:.2f}s)")
    print(f"[*] Signal strength: {signal_strength:.4f} ± {np.std(np.abs(samples)):.4f}")

    # Check if there's a strong signal
    if signal_strength < 0.01:
        print(f"[-] Weak signal at {frequency/1e6:.1f} MHz, trying different frequency...")
        return False

    print(f"[*] Demodulating FM...")
    audio = fm_demodulate(samples, AUDIO_RATE, SAMPLE_RATE)

    # Save audio
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"fm_{station_name.replace(' ', '_').lower()}_{timestamp}.wav"
    save_wav(filename, audio, AUDIO_RATE)

    # Signal analysis
    print(f"\n[*] Signal Analysis:")
    print(f"    Mean amplitude: {np.mean(np.abs(audio)):.4f}")
    print(f"    Max amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"    RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    return True


def scan_fm_band():
    """Quick scan of FM band to find strong signals"""
    print(f"\n{'='*50}")
    print(f"=== FM Band Scan (88-108 MHz) ===")
    print(f"{'='*50}")

    results = []

    # Scan in 2 MHz steps
    for freq_mhz in range(88, 109, 2):
        freq = freq_mhz * 1e6
        sdr = RtlSdr()
        sdr.sample_rate = SAMPLE_RATE
        sdr.center_freq = freq
        sdr.gain = GAIN

        # Quick capture (0.5s)
        samples = sdr.read_samples(int(SAMPLE_RATE * 0.5))
        sdr.close()

        strength = np.mean(np.abs(samples))
        results.append((freq_mhz, strength))
        print(f"  {freq_mhz:3d} MHz: {'█' * int(strength*50):30s} {strength:.4f}")

    # Find strongest signals
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n[*] Strongest signals:")
    for freq, strength in results[:3]:
        print(f"    {freq} MHz: {strength:.4f}")

    return results[0][0] * 1e6 if results else None


def main():
    print(f"FM Radio Capture using RTL-SDR V4")
    print(f"Date: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Option 1: Try to scan FM band first
    try:
        best_freq = scan_fm_band()

        if best_freq:
            print(f"\n[*] Trying strongest frequency: {best_freq/1e6:.1f} MHz")
            capture_fm(best_freq, f"FM_{best_freq/1e6:.1f}MHz")
    except Exception as e:
        print(f"[-] Scan failed: {e}")
        print(f"[*] Trying known frequencies...")

    # Option 2: Try specific stations
    for station_name, freq in FREQUENCIES.items():
        try:
            if capture_fm(freq, station_name):
                break
        except Exception as e:
            print(f"[-] Failed to capture {station_name}: {e}")
            continue

    print(f"\n{'='*50}")
    print(f"[+] FM Radio capture complete!")
    print(f"    To listen to the audio, open the .wav files")
    print(f"    with any media player (VLC, Windows Media Player)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
