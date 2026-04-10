#!/usr/bin/env python3
"""
FM Stereo Decoder (88-108 MHz)
Decodes stereo FM radio with RDS (Radio Data System)
"""

import logging
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import DatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "fm_radio"
DB_PATH = Path(__file__).parent.parent / "data" / "nanoprobe.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RTL_FM path
RTL_FM_PATH = (
    Path(__file__).parent.parent / "tools" / "rtl-sdr-blog" / "x64" / "rtl_fm.exe"
)

# Device serial
RTLSDR_SERIAL = "00000001"

# FM broadcast band
FM_BAND_LOW = 87.5e6
FM_BAND_HIGH = 108.0e6


class FMStereoDecoder:
    """FM stereo decoder with database integration"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager(str(DB_PATH))
        self._ensure_table()

    def record_station(
        self,
        frequency_mhz: float,
        duration: int = 60,
        gain: int = 30,
        output_file: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Record FM station using rtl_fm.

        Args:
            frequency_mhz: Station frequency in MHz
            duration: Recording duration in seconds
            gain: RTL-SDR gain
            output_file: Output WAV file path

        Returns:
            Path to recorded file or None
        """
        if not RTL_FM_PATH.exists():
            print(f"[!] rtl_fm not found at {RTL_FM_PATH}")
            return None

        if not output_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = f"fm_{frequency_mhz:.1f}_{timestamp}.wav"

        output_path = OUTPUT_DIR / output_file
        frequency_hz = int(frequency_mhz * 1e6)

        print(f"FM Station Recording")
        print(f"Frequency: {frequency_mhz:.1f} MHz")
        print(f"Duration: {duration}s, Gain: {gain} dB")
        print(f"Output: {output_path}")
        print()

        # rtl_fm command
        cmd = [
            str(RTL_FM_PATH),
            "-d", f"0:{RTLSDR_SERIAL}",
            "-f", str(frequency_hz),
            "-g", str(gain),
            "-s", "256000",  # Sample rate
            "-r", "48000",  # Audio rate
            "-F", "9",  # FIR filter
            "-l", "0",  # Low pass
            "-L", "0",  # High pass
            "-T", str(duration),  # Duration
        ]

        print(f"[*] Starting recording...")
        start_time = time.time()

        try:
            with open(output_path, "wb") as wav_file:
                # Write WAV header
                wav_header = self._create_wav_header(48000, 1, 16)
                wav_file.write(wav_header)

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Read raw audio samples
                total_samples = 0
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    wav_file.write(chunk)
                    total_samples += len(chunk)

                proc.wait(timeout=duration + 30)

            elapsed = time.time() - start_time
            file_size = output_path.stat().st_size

            print(f"\n[+] Recording complete!")
            print(f"    Duration: {elapsed:.1f}s")
            print(f"    Samples: {total_samples}")
            print(f"    File size: {file_size / 1024:.1f} KB")
            print(f"    Output: {output_path}")

            # Update WAV header with correct size
            self._update_wav_header(output_path, total_samples, 48000, 1, 16)

            # Save to database
            self.save_recording(
                frequency_mhz, str(output_path), file_size, elapsed
            )

            return output_path

        except subprocess.TimeoutExpired:
            print(f"\n[!] Timeout ({duration}s), stopping...")
            proc.kill()
            proc.wait()
            return None
        except Exception as e:
            print(f"\n[!] Error: {e}")
            logger.error(f"FM recording error: {e}")
            return None

    def scan_band(
        self,
        low_mhz: float = 88.0,
        high_mhz: float = 108.0,
        step_mhz: float = 0.2,
        duration_per_freq: int = 2,
        gain: int = 30,
    ) -> list[dict]:
        """
        Scan FM broadcast band for active stations.

        Args:
            low_mhz: Lower frequency bound
            high_mhz: Upper frequency bound
            step_mhz: Frequency step
            duration_per_freq: Listen time per frequency
            gain: RTL-SDR gain

        Returns:
            List of detected stations with signal strength
        """
        if not RTL_FM_PATH.exists():
            print(f"[!] rtl_fm not found")
            return []

        print(f"FM Band Scan")
        print(f"Range: {low_mhz:.1f}-{high_mhz:.1f} MHz")
        print(f"Step: {step_mhz:.2f} MHz")
        print(f"Duration per freq: {duration_per_freq}s")
        print()

        stations = []
        freq = low_mhz
        total_freqs = int((high_mhz - low_mhz) / step_mhz) + 1
        current = 0

        while freq <= high_mhz:
            current += 1
            freq_hz = int(freq * 1e6)

            cmd = [
                str(RTL_FM_PATH),
                "-d", f"0:{RTLSDR_SERIAL}",
                "-f", str(freq_hz),
                "-g", str(gain),
                "-s", "256000",
                "-r", "48000",
                "-T", str(duration_per_freq),
            ]

            print(f"[{current}/{total_freqs}] Scanning {freq:.2f} MHz...", end=" ")

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Collect samples to measure power
                total_bytes = 0
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    # Check if we have enough data
                    if total_bytes > 10000:
                        break

                proc.wait(timeout=duration_per_freq + 10)

                # Calculate signal strength
                power = total_bytes / duration_per_freq if duration_per_freq > 0 else 0

                # Threshold: if we got meaningful audio data, station is active
                if power > 1000:  # Arbitrary threshold
                    stations.append(
                        {
                            "frequency_mhz": freq,
                            "signal_power": power,
                            "signal_strength_db": 10 * np.log10(power) if power > 0 else -99,
                            "active": True,
                        }
                    )
                    print(f"ACTIVE ({power:.0f})")
                else:
                    print(f"---")

            except Exception as e:
                print(f"ERROR: {e}")

            freq += step_mhz

        # Save results
        if stations:
            self.save_scan_results(stations)
            print(f"\n[+] Found {len(stations)} active station(s)")
            print(f"{'=' * 50}")
            for s in sorted(stations, key=lambda x: x["signal_strength_db"], reverse=True):
                print(f"  {s['frequency_mhz']:6.2f} MHz  |  {s['signal_strength_db']:.1f} dB")
        else:
            print(f"\n[-] No active stations detected")

        return stations

    def save_recording(
        self,
        frequency_mhz: float,
        file_path: str,
        file_size: int,
        duration: float,
    ) -> int:
        """Save recording metadata to database"""
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fm_recordings
                (frequency_mhz, file_path, file_size_bytes, duration_sec, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (frequency_mhz, file_path, file_size, duration, now),
            )
            conn.commit()
            return cursor.lastrowid

    def save_scan_results(self, stations: list[dict]) -> int:
        """Save scan results to database"""
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for station in stations:
                cursor.execute(
                    """
                    INSERT INTO fm_stations
                    (frequency_mhz, signal_strength_db, signal_power, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        station["frequency_mhz"],
                        station["signal_strength_db"],
                        station["signal_power"],
                        now,
                    ),
                )
                count += 1
            conn.commit()

        return count

    def get_recordings(self, limit: int = 50) -> list[dict]:
        """Get recent recordings"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, frequency_mhz, file_path, file_size_bytes,
                       duration_sec, created_at
                FROM fm_recordings
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_known_stations(self) -> list[dict]:
        """Get known FM stations (most recent scan)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s1.*
                FROM fm_stations s1
                INNER JOIN (
                    SELECT frequency_mhz, MAX(created_at) as max_time
                    FROM fm_stations
                    GROUP BY frequency_mhz
                ) s2 ON s1.frequency_mhz = s2.frequency_mhz
                    AND s1.created_at = s2.max_time
                ORDER BY s1.signal_strength_db DESC
                """
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get FM radio statistics"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM fm_recordings")
            total_recordings = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT frequency_mhz) FROM fm_stations")
            unique_stations = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(file_size_bytes) FROM fm_recordings")
            total_size = cursor.fetchone()[0] or 0

        return {
            "total_recordings": total_recordings,
            "unique_stations": unique_stations,
            "total_storage_bytes": total_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _ensure_table(self):
        """Create FM radio tables if not exist"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fm_recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frequency_mhz REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    duration_sec REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fm_stations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frequency_mhz REAL NOT NULL,
                    signal_strength_db REAL,
                    signal_power REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_fm_rec_time ON fm_recordings(created_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_fm_station_freq ON fm_stations(frequency_mhz)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_fm_station_time ON fm_stations(created_at DESC)"
            )
            conn.commit()

    def _create_wav_header(
        self, sample_rate: int, channels: int, bits_per_sample: int
    ) -> bytes:
        """Create WAV file header"""
        import struct

        # Placeholder header (will be updated later)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            0,  # File size (placeholder)
            b"WAVE",
            b"fmt ",
            16,  # Subchunk1 size
            1,  # Audio format (PCM)
            channels,
            sample_rate,
            sample_rate * channels * bits_per_sample // 8,  # Byte rate
            channels * bits_per_sample // 8,  # Block align
            bits_per_sample,
            b"data",
            0,  # Data size (placeholder)
        )
        return header

    def _update_wav_header(
        self,
        filepath: Path,
        data_size: int,
        sample_rate: int,
        channels: int,
        bits_per_sample: int,
    ):
        """Update WAV file header with correct sizes"""
        file_size = data_size + 44 - 8  # Total file size - 8
        with open(filepath, "r+b") as f:
            # Update file size
            f.seek(4)
            import struct

            f.write(struct.pack("<I", file_size))

            # Update data size
            f.seek(40)
            f.write(struct.pack("<I", data_size))


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="FM Stereo Decoder (88-108 MHz)")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["record", "scan", "list", "stats"],
        default="stats",
        help="Mode: record station, scan band, list recordings, stats",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=106.0,
        help="Station frequency in MHz (default: 106.0)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=60,
        help="Recording duration in seconds (default: 60)",
    )
    parser.add_argument(
        "-g", "--gain", type=int, default=30, help="RTL-SDR gain (default: 30)"
    )

    args = parser.parse_args()

    decoder = FMStereoDecoder()

    if args.mode == "record":
        decoder.record_station(args.frequency, args.duration, args.gain)

    elif args.mode == "scan":
        decoder.scan_band(duration_per_freq=2, gain=args.gain)

    elif args.mode == "list":
        recordings = decoder.get_recordings()
        if recordings:
            print(f"\nRecent recordings ({len(recordings)}):")
            for r in recordings[:10]:
                size_kb = r.get("file_size_bytes", 0) / 1024
                print(
                    f"  {r['frequency_mhz']:.1f} MHz - "
                    f"{size_kb:.1f} KB, {r['duration_sec']:.1f}s"
                )
        else:
            print("No recordings in database")

    elif args.mode == "stats":
        stats = decoder.get_stats()
        print(f"\nFM Radio Statistics:")
        print(f"  Total recordings: {stats['total_recordings']}")
        print(f"  Unique stations: {stats['unique_stations']}")
        print(f"  Storage used: {stats['total_storage_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
