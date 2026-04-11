#!/usr/bin/env python3
"""
RTL_433 Multi-Frequency Scanner
Scans 433/868/915 MHz ISM bands for weather stations and sensors
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import DatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)

# Path to rtl_433 binary
RTL_433_PATH = Path(__file__).parent.parent / "tools" / "rtl_433" / "rtl_433-rtlsdr.exe"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "rtl433"
DB_PATH = Path(__file__).parent.parent / "data" / "nanoprobe.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common ISM frequencies
FREQUENCIES = {
    "433_MHz": 433920000,
    "868_MHz": 868300000,
    "915_MHz": 915000000,
}

# Device serial
RTLSDR_SERIAL = "00000001"


class RTL433Scanner:
    """RTL_433 multi-frequency scanner"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager(str(DB_PATH))
        self.devices_found = []

    def scan_frequency(
        self,
        frequency: int,
        duration: int = 30,
        gain: int = 40,
    ) -> list[dict]:
        """
        Scan a single frequency for RTL_433 devices.

        Args:
            frequency: Frequency in Hz
            duration: Scan duration in seconds
            gain: RTL-SDR gain (0-50)

        Returns:
            List of detected devices
        """
        if not RTL_433_PATH.exists():
            raise FileNotFoundError(f"rtl_433 not found at {RTL_433_PATH}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_file = OUTPUT_DIR / f"rtl433_{timestamp}.jsonl"
        log_file = OUTPUT_DIR / f"rtl433_{timestamp}.log"

        freq_mhz = frequency / 1_000_000
        print(f"\n{'=' * 60}")
        print(f"Scanning {freq_mhz:.3f} MHz for {duration}s")
        print(f"{'=' * 60}")

        cmd = [
            str(RTL_433_PATH),
            "-d",
            f"0:{RTLSDR_SERIAL}",
            "-f",
            str(frequency),
            "-g",
            str(gain),
            "-T",
            str(duration),
            "-F",
            f"json:{json_file}",
        ]

        devices = []
        try:
            with open(log_file, "w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                for line in proc.stdout:
                    log.write(line)
                    log.flush()

                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            if "model" in data:
                                devices.append(data)
                                self._print_device(data)
                        except json.JSONDecodeError:
                            pass

                proc.wait(timeout=duration + 10)

        except subprocess.TimeoutExpired:
            print(f"\n[*] Timeout ({duration}s), stopping...")
            proc.kill()
            proc.wait()
        except Exception as e:
            print(f"\n[!] Error: {e}")
            logger.error(f"RTL_433 scan error: {e}")

        print(f"\n[+] Found {len(devices)} device(s) at {freq_mhz:.3f} MHz")
        return devices

    def scan_all_frequencies(
        self,
        duration_per_freq: int = 30,
        gain: int = 40,
    ) -> dict:
        """
        Scan all ISM frequencies.

        Args:
            duration_per_freq: Duration per frequency in seconds
            gain: RTL-SDR gain

        Returns:
            Dict with frequency as key and devices as value
        """
        all_results = {}
        start_time = time.time()

        for freq_name, freq_hz in FREQUENCIES.items():
            devices = self.scan_frequency(freq_hz, duration_per_freq, gain)
            all_results[freq_name] = devices
            self.devices_found.extend(devices)

        elapsed = time.time() - start_time

        # Save results
        self._save_results(all_results, elapsed)

        return all_results

    def save_to_database(self, devices: list[dict]) -> int:
        """
        Save detected devices to database.

        Args:
            devices: List of device data dicts

        Returns:
            Number of records inserted
        """
        if not devices:
            return 0

        # Create rtl433_readings table if not exists
        self._ensure_table()

        count = 0
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            for device in devices:
                model = device.get("model", "unknown")
                device_id = device.get("id", "unknown")
                channel = device.get("channel", None)
                battery = device.get("battery_ok", None)
                temperature = device.get("temperature_C", None)
                humidity = device.get("humidity", None)
                pressure = device.get("pressure_hPa", None)
                wind_speed = device.get("wind_avg_km_h", None)
                rain = device.get("rain_mm", None)

                # Convert to JSON for raw data
                raw_data = json.dumps(device, default=str)

                cursor.execute(
                    """
                    INSERT INTO rtl433_readings
                    (model, device_id, channel, battery_ok, temperature_c,
                     humidity, pressure_hpa, wind_speed_kmh, rain_mm,
                     raw_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model,
                        str(device_id),
                        channel,
                        battery,
                        temperature,
                        humidity,
                        pressure,
                        wind_speed,
                        rain,
                        raw_data,
                        now,
                    ),
                )
                count += 1

            conn.commit()

        print(f"[*] Saved {count} reading(s) to database")
        return count

    def get_recent_readings(self, limit: int = 50) -> list[dict]:
        """
        Get recent readings from database.

        Args:
            limit: Maximum number of readings

        Returns:
            List of reading dicts
        """
        self._ensure_table()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, model, device_id, channel, battery_ok,
                       temperature_c, humidity, pressure_hpa,
                       wind_speed_kmh, rain_mm, raw_data, created_at
                FROM rtl433_readings
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_device_summary(self) -> dict:
        """
        Get summary of unique devices detected.

        Returns:
            Dict with device summaries
        """
        self._ensure_table()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model, device_id, channel,
                       COUNT(*) as reading_count,
                       MAX(created_at) as last_seen,
                       AVG(temperature_c) as avg_temp,
                       AVG(humidity) as avg_humidity
                FROM rtl433_readings
                GROUP BY model, device_id, channel
                ORDER BY last_seen DESC
                """
            )

            devices = {}
            for row in cursor.fetchall():
                key = f"{row[0]}_{row[1]}"
                devices[key] = {
                    "model": row[0],
                    "device_id": row[1],
                    "channel": row[2],
                    "reading_count": row[3],
                    "last_seen": row[4],
                    "avg_temperature_c": round(row[5], 2) if row[5] else None,
                    "avg_humidity": round(row[6], 2) if row[6] else None,
                }

            return devices

    def _print_device(self, data: dict):
        """Print device data in human-readable format"""
        model = data.get("model", "unknown")
        device_id = data.get("id", "?")
        parts = [f"  [DEVICE] {model} (id={device_id})"]

        if "temperature_C" in data:
            parts.append(f"Temp: {data['temperature_C']}°C")
        if "humidity" in data:
            parts.append(f"Humidity: {data['humidity']}%")
        if "battery_ok" in data:
            parts.append(f"Battery: {'OK' if data['battery_ok'] else 'LOW'}")
        if "channel" in data:
            parts.append(f"Ch: {data['channel']}")

        print("  |  ".join(parts))

    def _save_results(self, results: dict, elapsed: float):
        """Save scan results to JSON file"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_file = OUTPUT_DIR / f"rtl433_summary_{timestamp}.json"

        summary = {
            "scan_time": timestamp,
            "duration_seconds": elapsed,
            "frequencies_scanned": list(FREQUENCIES.keys()),
            "total_devices": len(self.devices_found),
            "results": results,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'=' * 60}")
        print(f"Scan Summary")
        print(f"{'=' * 60}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Total devices: {len(self.devices_found)}")
        print(f"Summary saved to: {summary_file}")

        # Per-frequency summary
        for freq_name, devices in results.items():
            print(f"\n  {freq_name}: {len(devices)} device(s)")

            if devices:
                models = {}
                for d in devices:
                    model = d.get("model", "unknown")
                    models[model] = models.get(model, 0) + 1

                for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {model}: {count} message(s)")

    def _ensure_table(self):
        """Create rtl433_readings table if not exists"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rtl433_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    channel INTEGER,
                    battery_ok INTEGER,
                    temperature_c REAL,
                    humidity REAL,
                    pressure_hpa REAL,
                    wind_speed_kmh REAL,
                    rain_mm REAL,
                    raw_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rtl433_model ON rtl433_readings(model)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_rtl433_device ON rtl433_readings(device_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_rtl433_time ON rtl433_readings(created_at DESC)"
            )
            conn.commit()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="RTL_433 Multi-Frequency Scanner")
    parser.add_argument(
        "-f",
        "--frequency",
        choices=["433", "868", "915", "all"],
        default="all",
        help="Frequency band to scan (default: all)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=30,
        help="Duration per frequency in seconds (default: 30)",
    )
    parser.add_argument("-g", "--gain", type=int, default=40, help="RTL-SDR gain (default: 40)")
    parser.add_argument("--list", action="store_true", help="List recent readings from database")
    parser.add_argument("--summary", action="store_true", help="Show device summary from database")

    args = parser.parse_args()

    scanner = RTL433Scanner()

    if args.list:
        readings = scanner.get_recent_readings()
        if readings:
            print(f"\nRecent readings ({len(readings)}):")
            for r in readings[:10]:
                print(
                    f"  {r['model']} (id={r['device_id']}) - "
                    f"Temp: {r.get('temperature_c', 'N/A')}°C, "
                    f"Humidity: {r.get('humidity', 'N/A')}%"
                )
        else:
            print("No readings in database")
        return

    if args.summary:
        devices = scanner.get_device_summary()
        if devices:
            print(f"\nDevice summary ({len(devices)} unique device(s)):")
            for key, dev in devices.items():
                print(f"  {dev['model']} (id={dev['device_id']})")
                print(f"    Readings: {dev['reading_count']}, " f"Last seen: {dev['last_seen']}")
                if dev["avg_temperature_c"]:
                    print(f"    Avg Temp: {dev['avg_temperature_c']}°C")
                if dev["avg_humidity"]:
                    print(f"    Avg Humidity: {dev['avg_humidity']}%")
        else:
            print("No devices in database")
        return

    # Scanning mode
    print(f"RTL_433 Multi-Frequency Scanner")
    print(f"Time: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: RTL-SDR Blog V4 (SN: {RTLSDR_SERIAL})")
    print()

    if args.frequency == "all":
        results = scanner.scan_all_frequencies(args.duration, args.gain)
    else:
        freq_map = {"433": 433920000, "868": 868300000, "915": 915000000}
        freq_hz = freq_map[args.frequency]
        devices = scanner.scan_frequency(freq_hz, args.duration, args.gain)
        scanner.devices_found = devices
        results = {f"{args.frequency}_MHz": devices}

    # Save to database
    all_devices = []
    for devices in results.values():
        all_devices.extend(devices)

    if all_devices:
        scanner.save_to_database(all_devices)
    else:
        print("\n[-] No devices detected")
        print("    Try:")
        print("    - Increasing gain (-g 50)")
        print("    - Using external antenna")
        print("    - Moving closer to sensors")


if __name__ == "__main__":
    asyncio.run(main())
