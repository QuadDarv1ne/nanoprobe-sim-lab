#!/usr/bin/env python3
"""
ADS-B Aircraft Tracker (1090 MHz)
Tracks aircraft using RTL-SDR and dump1090/readsb
"""

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

# Paths
DUMP1090_PATH = Path(__file__).parent.parent / "tools" / "dump1090"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "adsb"
DB_PATH = Path(__file__).parent.parent / "data" / "nanoprobe.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device serial
RTLSDR_SERIAL = "00000001"


class ADSBTracker:
    """ADS-B aircraft tracker with database integration"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager(str(DB_PATH))
        self.aircraft_seen = {}
        self._ensure_table()

    def capture_raw(
        self,
        duration: int = 10,
        gain: int = 40,
    ) -> Optional[Path]:
        """
        Capture raw ADS-B I/Q samples at 1090 MHz.

        Args:
            duration: Capture duration in seconds
            gain: RTL-SDR gain

        Returns:
            Path to saved .npy file or None
        """
        try:
            import numpy as np
            from rtlsdr import RtlSdr
        except ImportError:
            print("[!] pyrtlsdr or numpy not available")
            return None

        frequency = 1090e6
        sample_rate = 2.4e6

        print(f"ADS-B Raw Capture @ {frequency/1e6:.0f} MHz")
        print(f"Duration: {duration}s, Gain: {gain} dB")
        print()

        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.center_freq = frequency
        sdr.gain = gain

        try:
            print(f"[*] Capturing {duration} seconds...")
            total_samples = int(sample_rate * duration)
            samples = sdr.read_samples(total_samples)
            print(f"[+] Captured {len(samples)} I/Q samples")

            # Save
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = OUTPUT_DIR / f"adsb_raw_{timestamp}.npy"
            np.save(filepath, samples)

            print(f"[*] Saved to: {filepath}")
            return filepath

        finally:
            sdr.close()

    def decode_dump1090(
        self,
        duration: int = 30,
    ) -> list[dict]:
        """
        Decode ADS-B using dump1090/readsb.

        Supports two modes:
        1. Classic dump1090-mutability/fa: parses aircraft.json
        2. gvanem/dump1090-win: connects to HTTP API at port 8080

        Args:
            duration: Capture duration in seconds

        Returns:
            List of aircraft data dicts
        """
        # Check for dump1090 or readsb
        decoder = self._find_decoder()
        if not decoder:
            print("[!] dump1090/readsb not found")
            print("    Install: https://github.com/wiedehopf/readsb")
            return []

        print(f"Using decoder: {decoder}")

        # Detect decoder type
        decoder_type = self._detect_decoder_type(decoder)
        print(f"Decoder type: {decoder_type}")

        if decoder_type == "gvanem_win":
            return self._decode_gvanem_win(duration)
        else:
            return self._decode_classic(duration)

    def _detect_decoder_type(self, decoder_path: Path) -> str:
        """Detect decoder type based on path and behavior"""
        # gvanem/Dump1090 has web_root-Tar1090 and dump1090.cfg nearby
        cfg_file = decoder_path.parent / "dump1090.cfg"
        web_dir = decoder_path.parent / "web_root-Tar1090"
        if cfg_file.exists() or web_dir.exists():
            return "gvanem_win"
        return "classic"

    def _decode_gvanem_win(self, duration: int) -> list[dict]:
        """
        Start gvanem/dump1090-win and fetch aircraft data via HTTP API.
        This version runs as a background process with --net --interactive.
        """
        decoder = self._find_decoder()
        if not decoder:
            return []

        print(f"Starting gvanem/dump1090-win...")
        print(f"Capture duration: {duration}s")

        # Start dump1090 in background
        # gvanem/dump1090-win requires config for gain/device settings
        cmd = [
            str(decoder),
            "--config",
            str(decoder.parent / "adsb_tracking.cfg"),
            "--net",
            "--interactive",
            "--max-messages",
            "0",  # unlimited
        ]

        process = None
        aircraft_data = []

        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(decoder.parent),
            )

            # Wait for server to start
            print("Waiting for dump1090 to initialize...")
            time.sleep(3)

            # Fetch aircraft data via HTTP API
            import json as json_mod
            import urllib.request

            # Try to fetch from the HTTP API
            urls_to_try = [
                "http://localhost:8080/data/aircraft.json",
                "http://localhost:8080/data.json",
            ]

            start_time = time.time()
            while time.time() - start_time < duration:
                for url in urls_to_try:
                    try:
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            data = json_mod.loads(resp.read().decode())
                            if "aircraft" in data:
                                aircraft_data = data["aircraft"]
                                break
                    except Exception:
                        continue

                if aircraft_data:
                    break

                time.sleep(2)
                elapsed = int(time.time() - start_time)
                print(
                    f"\r  Waiting for aircraft data... [{elapsed}/{duration}s]", end="", flush=True
                )

            print()  # newline after progress

        except KeyboardInterrupt:
            print("\n[*] User interrupted")
        except Exception as e:
            print(f"[!] Error: {e}")
            logger.error(f"ADS-B decode error: {e}")
        finally:
            if process:
                print("Stopping dump1090...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        # Process and save
        if aircraft_data:
            print(f"\n[+] Found {len(aircraft_data)} aircraft")
            self._print_aircraft(aircraft_data)
            self.save_to_database(aircraft_data)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            json_file = OUTPUT_DIR / f"adsb_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump({"timestamp": timestamp, "aircraft": aircraft_data}, f, indent=2)
            print(f"[*] Saved to: {json_file}")
        else:
            print("\n[-] No aircraft detected")
            print("    Try:")
            print("    - Moving near window")
            print("    - Using external antenna")
            print("    - Increasing duration")

        return aircraft_data

    def _decode_classic(self, duration: int) -> list[dict]:
        """
        Decode ADS-B using classic dump1090-mutability/fa with aircraft.json output.
        """
        decoder = self._find_decoder()
        if not decoder:
            return []

        print(f"Using decoder: {decoder}")
        print(f"Capturing for {duration}s...")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_file = OUTPUT_DIR / f"adsb_{timestamp}.json"

        # Run dump1090 in JSON output mode
        cmd = [
            str(decoder),
            "--device",
            f"0:{RTLSDR_SERIAL}",
            "--gain",
            "max",
            "--net",
            "--net-beast",
            "--json-location-enabled",
            "--write-json",
            str(OUTPUT_DIR),
            "--max-range",
            "300",
            "--duration",
            str(duration),
        ]

        aircraft_data = []
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 30,
            )

            # Parse aircraft.json if created
            aircraft_json = OUTPUT_DIR / "aircraft.json"
            if aircraft_json.exists():
                with open(aircraft_json) as f:
                    data = json.load(f)
                    aircraft_data = data.get("aircraft", [])

        except subprocess.TimeoutExpired:
            print(f"\n[*] Timeout ({duration}s), stopping...")
        except Exception as e:
            print(f"[!] Error: {e}")
            logger.error(f"ADS-B decode error: {e}")

        # Process and save
        if aircraft_data:
            print(f"\n[+] Found {len(aircraft_data)} aircraft")
            self._print_aircraft(aircraft_data)
            self.save_to_database(aircraft_data)

            # Save JSON
            with open(json_file, "w") as f:
                json.dump({"timestamp": timestamp, "aircraft": aircraft_data}, f, indent=2)
            print(f"[*] Saved to: {json_file}")
        else:
            print("\n[-] No aircraft detected")
            print("    Try:")
            print("    - Moving near window")
            print("    - Using external antenna")
            print("    - Increasing duration")

        return aircraft_data

    def save_to_database(self, aircraft_list: list[dict]) -> int:
        """
        Save aircraft sightings to database.

        Args:
            aircraft_list: List of aircraft data from dump1090

        Returns:
            Number of records inserted
        """
        if not aircraft_list:
            return 0

        count = 0
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            for aircraft in aircraft_list:
                icao = aircraft.get("hex", "unknown")
                flight = aircraft.get("flight", "").strip() or None
                altitude = aircraft.get("alt_baro", None) or aircraft.get("altitude", None)
                speed = aircraft.get("speed", None)
                track = aircraft.get("track", None)
                lat = aircraft.get("lat", None)
                lon = aircraft.get("lon", None)
                vert_rate = aircraft.get("vert_rate", None)
                category = aircraft.get("category", None)
                squawk = aircraft.get("squawk", None)
                rssi = aircraft.get("rssi", None)
                messages = aircraft.get("messages", 0)
                seen = aircraft.get("seen", None)

                cursor.execute(
                    """
                    INSERT INTO adsb_sightings
                    (icao, flight, altitude_ft, speed_knots, heading,
                     latitude, longitude, vertical_rate, category, squawk,
                     rssi_db, message_count, seconds_ago, raw_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        icao,
                        flight,
                        altitude,
                        speed,
                        track,
                        lat,
                        lon,
                        vert_rate,
                        category,
                        squawk,
                        rssi,
                        messages,
                        seen,
                        json.dumps(aircraft, default=str),
                        now,
                    ),
                )
                count += 1

            conn.commit()

        print(f"[*] Saved {count} sighting(s) to database")
        return count

    def get_recent_sightings(self, limit: int = 50) -> list[dict]:
        """Get recent aircraft sightings"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, icao, flight, altitude_ft, speed_knots, heading,
                       latitude, longitude, category, squawk, rssi_db,
                       message_count, seconds_ago, created_at
                FROM adsb_sightings
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_active_aircraft(self, limit: int = 100) -> list[dict]:
        """Get currently active aircraft (most recent sighting per ICAO)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s1.*
                FROM adsb_sightings s1
                INNER JOIN (
                    SELECT icao, MAX(created_at) as max_time
                    FROM adsb_sightings
                    GROUP BY icao
                ) s2 ON s1.icao = s2.icao AND s1.created_at = s2.max_time
                ORDER BY s1.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get ADS-B statistics"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM adsb_sightings")
            total_sightings = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT icao) FROM adsb_sightings")
            unique_aircraft = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM adsb_sightings")
            row = cursor.fetchone()
            first_seen = row[0]
            last_seen = row[1]

        return {
            "total_sightings": total_sightings,
            "unique_aircraft": unique_aircraft,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _ensure_table(self):
        """Create adsb_sightings table if not exists"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS adsb_sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    icao TEXT NOT NULL,
                    flight TEXT,
                    altitude_ft REAL,
                    speed_knots REAL,
                    heading REAL,
                    latitude REAL,
                    longitude REAL,
                    vertical_rate REAL,
                    category TEXT,
                    squawk TEXT,
                    rssi_db REAL,
                    message_count INTEGER,
                    seconds_ago REAL,
                    raw_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_adsb_icao ON adsb_sightings(icao)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_adsb_time ON adsb_sightings(created_at DESC)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_adsb_flight ON adsb_sightings(flight)")
            conn.commit()

    def _find_decoder(self) -> Optional[Path]:
        """Find dump1090 or readsb binary"""
        import shutil

        # Check PATH first
        for name in ["dump1090", "readsb"]:
            found = shutil.which(name)
            if found:
                return Path(found)

        # Check project tools directory
        project_dump1090 = Path(__file__).parent.parent / "tools" / "Dump1090-main" / "dump1090.exe"
        if project_dump1090.exists():
            return project_dump1090

        # Check other common Windows locations
        candidates = [
            Path("C:/dump1090/dump1090.exe"),
            Path("C:/Program Files/dump1090/dump1090.exe"),
            Path("C:/Program Files (x86)/dump1090/dump1090.exe"),
            Path("C:/readsb/readsb.exe"),
            Path.home() / "dump1090" / "dump1090.exe",
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    def _print_aircraft(self, aircraft_list: list[dict]):
        """Print aircraft data in human-readable format"""
        print(f"\n{'=' * 70}")
        print(f"{'Callsign':<10} {'ICAO':<8} {'Alt(ft)':<8} {'Speed':<8} {'Heading':<8}")
        print(f"{'=' * 70}")

        for ac in aircraft_list:
            flight = (ac.get("flight") or "--------").strip()
            icao = ac.get("hex", "??????")
            alt = ac.get("alt_baro") or ac.get("altitude")
            speed = ac.get("speed")
            heading = ac.get("track")

            alt_str = f"{alt:,.0f}" if alt else "------"
            speed_str = f"{speed:.0f}" if speed else "---"
            heading_str = f"{heading:.0f}°" if heading else "---"

            print(f"{flight:<10} {icao:<8} {alt_str:<8} {speed_str:<8} {heading_str:<8}")

        print(f"{'=' * 70}")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ADS-B Aircraft Tracker (1090 MHz)")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["capture", "decode", "list", "stats"],
        default="decode",
        help="Mode: capture raw I/Q, decode via dump1090, list recent, stats",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=30,
        help="Capture duration in seconds (default: 30)",
    )
    parser.add_argument("-g", "--gain", type=int, default=40, help="RTL-SDR gain (default: 40)")

    args = parser.parse_args()

    tracker = ADSBTracker()

    if args.mode == "capture":
        tracker.capture_raw(args.duration, args.gain)

    elif args.mode == "decode":
        tracker.decode_dump1090(args.duration)

    elif args.mode == "list":
        sightings = tracker.get_recent_sightings()
        if sightings:
            print(f"\nRecent sightings ({len(sightings)}):")
            for s in sightings[:20]:
                alt = s.get("altitude_ft", "N/A")
                speed = s.get("speed_knots", "N/A")
                print(
                    f"  {s['flight'] or 'N/A':<10} {s['icao']} - "
                    f"Alt: {alt} ft, Speed: {speed} kts"
                )
        else:
            print("No sightings in database")

    elif args.mode == "stats":
        stats = tracker.get_stats()
        print(f"\nADS-B Statistics:")
        print(f"  Total sightings: {stats['total_sightings']}")
        print(f"  Unique aircraft: {stats['unique_aircraft']}")
        print(f"  First seen: {stats['first_seen'] or 'Never'}")
        print(f"  Last seen: {stats['last_seen'] or 'Never'}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
