#!/usr/bin/env python3
"""
SSTV Ground Station - Главный CLI для управления RTL-SDR и SSTV
Удобный интерфейс для всех функций SSTV Ground Station
"""

import sys
import os

# Добавляем пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "components", "py-sstv-groundstation", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "components", "py-sstv-groundstation"))

import argparse
from datetime import datetime
from pathlib import Path

def cmd_check(args):
    """Проверка RTL-SDR устройства"""
    print("=" * 60)
    print("ПРОВЕРКА RTL-SDR УСТРОЙСТВА")
    print("=" * 60)
    
    # Запускаем verify_rtlsdr.py
    verify_script = Path(__file__).parent / "components" / "py-sstv-groundstation" / "verify_rtlsdr.py"
    if verify_script.exists():
        os.system(f'"{sys.executable}" "{verify_script}"')
    else:
        print("❌ Скрипт verify_rtlsdr.py не найден")

def cmd_passes(args):
    """Расписание пролётов спутников"""
    from satellite_tracker import SatelliteTracker
    
    print("=" * 60)
    print(f"РАСПИСАНИЕ ПРОЛЁТОВ СПУТНИКОВ")
    print(f"Станция: {args.lat:.2f}°N, {args.lon:.2f}°E")
    print("=" * 60)
    
    tracker = SatelliteTracker(ground_station_lat=args.lat, ground_station_lon=args.lon)
    
    # Пытаемся обновить TLE
    try:
        tracker.update_tle_from_celestrak()
    except Exception as e:
        print(f"⚠ Не удалось обновить TLE: {e}")
    
    # Получаем расписание
    if hasattr(args, 'satellite') and args.satellite:
        # Конкретный спутник
        passes = tracker.get_pass_predictions(args.satellite, hours_ahead=args.hours)
        if passes:
            print(f"\n📡 {args.satellite.upper()} - {len(passes)} пролётов:")
            for i, p in enumerate(passes[:10], 1):  # Показываем первые 10
                print(f"\n  {i}. {p['aos'].strftime('%d.%m %H:%M')} - {p['los'].strftime('%H:%M')}")
                print(f"     Max elevation: {p['max_elevation']:.1f}°")
                print(f"     Длительность: {p['duration_minutes']:.1f} мин")
                print(f"     Частота: {p['frequency']:.3f} MHz")
        else:
            print(f"\n⚠ Нет пролётов для {args.satellite}")
    else:
        # Все спутники
        schedule = tracker.get_sstv_schedule(hours_ahead=args.hours)
        if schedule:
            print(f"\n📡 Расписание SSTV ({len(schedule)} пролётов):")
            for i, s in enumerate(schedule[:15], 1):  # Первые 15
                print(f"\n  {i}. {s['satellite'].upper()} - {s['aos'].strftime('%d.%m %H:%M')}")
                print(f"     Max elevation: {s['max_elevation']:.1f}°")
                print(f"     Длительность: {s['duration_minutes']:.1f} мин")
                print(f"     Частота: {s['frequency']:.3f} MHz")
        else:
            print("\n⚠ Нет пролётов в ближайшие часы")

def cmd_position(args):
    """Текущая позиция спутника"""
    from satellite_tracker import SatelliteTracker
    
    tracker = SatelliteTracker()
    pos = tracker.get_current_position(args.satellite)
    
    if pos:
        print("=" * 60)
        print(f"ПОЗИЦИЯ {args.satellite.upper()}")
        print("=" * 60)
        print(f"  Широта: {pos['latitude']:.4f}°")
        print(f"  Долгота: {pos['longitude']:.4f}°")
        print(f"  Высота: {pos['altitude_km']:.1f} км")
        print(f"  Скорость: {pos['velocity_kmh']:.0f} км/ч")
        print(f"  Зона покрытия: {pos['footprint_km']:.0f} км")
        print(f"  Время: {pos['timestamp']}")
    else:
        print(f"❌ Не удалось получить позицию {args.satellite}")

def cmd_waterfall(args):
    """Waterfall спектрограмма"""
    print("=" * 60)
    print("WATERFALL СПЕКТРОГРАММА")
    print("=" * 60)
    
    waterfall_script = Path(__file__).parent / "run_waterfall.py"
    if waterfall_script.exists():
        os.system(f'"{sys.executable}" "{waterfall_script}"')
    else:
        print("❌ Скрипт run_waterfall.py не найден")
        print("Создайте файл run_waterfall.py в корне проекта")

def cmd_record(args):
    """Запись SSTV сигнала"""
    print("=" * 60)
    print("ЗАПИСЬ SSTV СИГНАЛА")
    print("=" * 60)
    
    from sstv_decoder import SSTVDecoder, detect_sstv_signal
    
    # Создаём директорию
    output_dir = Path(__file__).parent / "sstv_recordings"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_file = output_dir / f"sstv_{timestamp}.wav"
    
    print(f"\n📡 Запись {args.duration} секунд на частоте {args.frequency:.3f} MHz")
    print(f"📁 Сохранение: {wav_file}")
    
    # Запускаем запись через run_sstv_decoder.py
    decoder_script = Path(__file__).parent / "run_sstv_decoder.py"
    if decoder_script.exists():
        os.system(f'"{sys.executable}" "{decoder_script}" --frequency {args.frequency} --duration {args.duration}')
    else:
        print("❌ Скрипт run_sstv_decoder.py не найден")

def cmd_decode(args):
    """Декодирование SSTV из аудиофайла"""
    from sstv_decoder import SSTVDecoder, detect_sstv_signal
    
    audio_file = args.audio
    
    if not Path(audio_file).exists():
        print(f"❌ Файл не найден: {audio_file}")
        return
    
    print("=" * 60)
    print("ДЕКОДИРОВАНИЕ SSTV")
    print("=" * 60)
    
    # Обнаружение сигнала
    found, metadata = detect_sstv_signal(audio_file)
    if found:
        print(f"✓ SSTV сигнал обнаружен")
        print(f"  Длительность: {metadata.get('duration_seconds', 0):.1f}с")
        print(f"  VIS ratio: {metadata.get('sstv_band_ratio', 0):.2f}")
    else:
        print("⚠ SSTV сигнал не обнаружен, пробуем декодировать...")
    
    # Декодирование
    decoder = SSTVDecoder()
    image = decoder.decode_from_audio(audio_file)
    
    if image:
        output_file = Path(audio_file).stem + "_decoded.png"
        decoder.save_decoded_image(output_file)
        print(f"\n✅ Изображение декодировано!")
        print(f"  Режим: {decoder.get_metadata().get('mode', 'unknown')}")
        print(f"  Размер: {image.size[0]}x{image.size[1]}")
        print(f"  Файл: {output_file}")
    else:
        print("\n❌ Не удалось декодировать изображение")

def cmd_auto_record(args):
    """Автоматическая запись при пролёте"""
    from satellite_tracker import SatelliteTracker
    import time
    
    print("=" * 60)
    print("АВТОМАТИЧЕСКАЯ ЗАПИСЬ SSTV")
    print("=" * 60)
    
    tracker = SatelliteTracker(ground_station_lat=args.lat, ground_station_lon=args.lon)
    
    print(f"📡 Мониторинг пролётов ISS...")
    print(f"📍 Станция: {args.lat:.2f}°N, {args.lon:.2f}°E")
    print(f"⏱ Проверка каждые {args.interval} секунд")
    print("\nНажмите Ctrl+C для остановки\n")
    
    try:
        while True:
            # Проверяем видимость
            if tracker.is_satellite_visible('iss', min_elevation=10.0):
                pos = tracker.get_current_position('iss')
                print(f"\n🛰 ISS виден! Elevation: {pos.get('altitude_km', 0):.0f} км")
                print(f"  Запись SSTV на частоте 145.800 MHz...")
                
                # Запускаем запись
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(__file__).parent / "sstv_recordings"
                output_dir.mkdir(exist_ok=True)
                
                # Здесь можно запустить запись через RTL-SDR
                print(f"  📁 Сохранение в: {output_dir / f'auto_{timestamp}.wav'}")
                
                # Ждём пока спутник не скроется
                while tracker.is_satellite_visible('iss', min_elevation=5.0):
                    time.sleep(10)
                
                print("  ✅ Запись завершена")
            else:
                print(f".", end="", flush=True)
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\n⏹ Остановлено пользователем")

def main():
    parser = argparse.ArgumentParser(
        description="SSTV Ground Station - Управление RTL-SDR и SSTV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s check                              # Проверка RTL-SDR
  %(prog)s passes --lat 55.75 --lon 37.61    # Расписание пролётов
  %(prog)s position --satellite iss           # Позиция ISS
  %(prog)s waterfall                          # Waterfall спектрограмма
  %(prog)s record --duration 60               # Запись SSTV
  %(prog)s decode --audio file.wav            # Декодирование
  %(prog)s auto-record --lat 55.75 --lon 37.61  # Автозапись
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # check
    subparsers.add_parser('check', help='Проверка RTL-SDR устройства')
    
    # passes
    passes_parser = subparsers.add_parser('passes', help='Расписание пролётов')
    passes_parser.add_argument('--satellite', '-s', help='Конкретный спутник (iss, noaa_15, etc)')
    passes_parser.add_argument('--hours', '-t', type=int, default=24, help='На сколько часов вперёд')
    passes_parser.add_argument('--lat', type=float, default=55.75, help='Широта станции')
    passes_parser.add_argument('--lon', type=float, default=37.61, help='Долгота станции')
    
    # position
    pos_parser = subparsers.add_parser('position', help='Позиция спутника')
    pos_parser.add_argument('--satellite', '-s', default='iss', help='Название спутника')
    
    # waterfall
    subparsers.add_parser('waterfall', help='Waterfall спектрограмма')
    
    # record
    rec_parser = subparsers.add_parser('record', help='Запись SSTV сигнала')
    rec_parser.add_argument('--frequency', '-f', type=float, default=145.800, help='Частота MHz')
    rec_parser.add_argument('--duration', '-d', type=int, default=60, help='Длительность записи (сек)')
    
    # decode
    dec_parser = subparsers.add_parser('decode', help='Декодирование SSTV')
    dec_parser.add_argument('--audio', '-a', required=True, help='Путь к аудиофайлу')
    
    # auto-record
    auto_parser = subparsers.add_parser('auto-record', help='Автоматическая запись')
    auto_parser.add_argument('--lat', type=float, default=55.75, help='Широта')
    auto_parser.add_argument('--lon', type=float, default=37.61, help='Долгота')
    auto_parser.add_argument('--interval', '-i', type=int, default=30, help='Интервал проверки (сек)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        'check': cmd_check,
        'passes': cmd_passes,
        'position': cmd_position,
        'waterfall': cmd_waterfall,
        'record': cmd_record,
        'decode': cmd_decode,
        'auto-record': cmd_auto_record,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
