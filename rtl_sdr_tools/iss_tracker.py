#!/usr/bin/env python3
"""
ISS/MKS Tracker — автоопределение координат + МСК время
Использует SGP4 + TLE из CelesTrak
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "components" / "py-sstv-groundstation" / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from satellite_tracker import SatelliteTracker

try:
    from utils.location_manager import MSK_TZ, get_location, get_location_info
except ImportError:
    try:
        from geolocation import TZInfo, get_location

        MSK_TZ = TZInfo("MSK", 3)

        def get_location_info():
            loc = get_location()
            return f"[LOC] {loc['city']}, {loc['country']}"

    except ImportError:

        def get_location():
            return {
                "lat": 55.7558,
                "lon": 37.6173,
                "city": "Moscow",
                "country": "Russia",
                "timezone": MSK_TZ,
            }

        def get_location_info():
            return "[LOC] Moscow, Russia (fallback)"


def main():
    loc = get_location()
    lat = loc["lat"]
    lon = loc["lon"]
    tz = loc.get("timezone", MSK_TZ)

    print("=" * 70)
    print("ISS TRACKER — Avto-opredelenie koordinat")
    print("=" * 70)
    print(get_location_info())
    print(f"Vremya: {tz.now_local().strftime('%Y-%m-%d %H:%M:%S')} {tz.name}")
    print("=" * 70)
    print()

    tracker = SatelliteTracker(ground_station_lat=lat, ground_station_lon=lon)

    # Обновление TLE из CelesTrak
    print("Zagruzka TLE dannykh iz CelesTrak...")
    try:
        updated = tracker.update_tle_from_celestrak()
        if updated:
            print(f"  OK: TLE obnovleno ({updated} sputnikov)")
        else:
            print("  WARNING: Ispolzuyutsya default TLE (CelesTrak nedostupen)")
    except Exception as e:
        print(f"  WARNING: {e}")
    print()

    # Текущая позиция МКС
    print("-" * 70)
    print("TEKUSCHEE POLOZHENIE MКС:")
    print("-" * 70)

    iss_pos = tracker.get_current_position("iss")
    if iss_pos:
        print(f"  Shirota:        {iss_pos.get('latitude', 0):.2f}°")
        print(f"  Dolgota:        {iss_pos.get('longitude', 0):.2f}°")
        print(f"  Vysota:         {iss_pos.get('altitude_km', 0):.1f} km")
        print(f"  Skorost:        {iss_pos.get('velocity_kmh', 0):.1f} km/h")
        print(f"  Podstyop:       {iss_pos.get('footprint_km', 0):.0f} km")

        # Проверяем видимость
        if "elevation" in iss_pos:
            elev = iss_pos["elevation"]
            if elev > 0:
                print(f"  Vysota nad gorizontom: {elev:.1f}° ✓ VIDNA!")
            else:
                print(f"  Vysota nad gorizontom: {elev:.1f}° (pod gorizontom)")
    else:
        print("  ERROR: Ne poluchilos opredelit pozitsiyu")

    print()

    # Ближайшие пролёты
    print("-" * 70)
    print("BLIZHAISHIE PROLYOTY MКС (24 chasa, min vysota 10°):")
    print("-" * 70)

    passes = tracker.get_pass_predictions(satellite_name="iss", hours_ahead=24, min_elevation=10.0)

    if passes:
        print(f"\n  Naydeno prolyotov: {len(passes)}\n")

        for i, p in enumerate(passes[:10], 1):  # Pokazyvaem pervye 10
            aos = p["aos"].strftime("%H:%M:%S")
            los = p["los"].strftime("%H:%M:%S")
            max_el = p["max_elevation"]
            duration = p["duration_minutes"]
            time_until = p.get("time_until_aos", "?")

            # Otsenka kachestva
            if max_el > 60:
                quality = "★★★★★ OTLICHNO"
            elif max_el > 40:
                quality = "★★★★  Xorosho"
            elif max_el > 20:
                quality = "★★★  Sredne"
            else:
                quality = "★★   Nizko"

            print(f"  {i}. {aos} - {los} ({duration:.0f} min)")
            print(f"     Maks. vysota: {max_el:.1f}° | {quality}")
            print(f"     Cherez: {time_until}")
            print(f"     Chastota SSTV: {p.get('frequency', 145.800):.3f} MHz")
            print()

            # Rekomendatsiya dlya zapisi
            if max_el > 30:
                print(f"  >>> REKOMENDUETSYA DLYA ZAPISI SSTV!")
                print(f"  >>> Zapusti za 2 minuty do {aos}:")
                print(f"  >>> python capture_sstv_mmsstv.py")
                print()
    else:
        print("\n  Net prolyotov v techenie 24 chasov s vysotoy > 10°")
        print("  Poprobuy snizit min_elevation do 5°")

    print("=" * 70)
    print()
    print("Dlya zapisi SSTV vo vremya prolyota:")
    print("  python capture_sstv_mmsstv.py")
    print()
    print("Ili s drugimi parametrami:")
    print("  python capture_sstv_mmsstv.py --duration 120 --gain 40")
    print()


if __name__ == "__main__":
    main()
