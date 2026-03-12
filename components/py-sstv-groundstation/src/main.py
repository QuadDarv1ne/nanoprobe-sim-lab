# -*- coding: utf-8 -*-
"""Наземная станция SSTV для приема и декодирования сигналов."""

import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sstv_decoder import SSTVDecoder, detect_sstv_signal
from sdr_interface import SDRInterface, SDRScanner, create_sdr


def show_banner():
    """Отображает баннер программы."""
    print("=" * 60)
    print("       НАЗЕМНАЯ СТАНЦИЯ SSTV")
    print("       SSTV Ground Station")
    print("=" * 60)


def mode_decode_audio(args):
    """Режим декодирования аудиофайла."""
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Ошибка: Файл '{audio_path}' не найден")
        return False

    print(f"Декодирование аудио: {audio_path}")
    decoder = SSTVDecoder(mode=args.mode)

    # Обнаружение сигнала
    if args.detect:
        found, metadata = detect_sstv_signal(str(audio_path))
        if found:
            print(f"✓ SSTV сигнал обнаружен")
            print(f"  Длительность: {metadata.get('duration_seconds', 0):.1f}с")
            print(f"  Частота дискретизации: {metadata.get('sample_rate', 0)} Гц")
        else:
            print("? SSTV сигнал не обнаружен или сомнительный")

    # Декодирование
    image = decoder.decode_from_audio(str(audio_path))

    if image:
        output = args.output or "decoded_sstv_image.png"
        decoder.save_decoded_image(output)
        print(f"Изображение сохранено: {output}")

        # Показываем метаданные
        metadata = decoder.get_metadata()
        print(f"Режим: {metadata.get('mode', 'unknown')}")
        print(f"Размер: {image.size[0]}x{image.size[1]}")
        return True
    else:
        print("Не удалось декодировать изображение")
        return False


def mode_receive_sdr(args):
    """Режим приема с SDR."""
    print(f"Прием с SDR: {args.frequency} МГц")

    sdr = create_sdr(
        device_index=args.device,
        frequency=args.frequency,
        sample_rate=args.sample_rate,
        gain=args.gain,
        bias_tee=args.bias_tee,
        agc=args.agc
    )
    if not sdr:
        print("Ошибка инициализации SDR")
        return False

    try:
        # Запись сигнала
        duration = args.duration or 60
        output_audio = args.output_audio or "sdr_recording.wav"

        print(f"Запись {duration}с... Нажмите Ctrl+C для остановки")
        sdr.start_recording(duration_seconds=duration, output_file=output_audio)

        # Ожидаем завершения или прерывания
        try:
            time.sleep(duration + 2)
        except KeyboardInterrupt:
            print("\nОстановка записи...")
            sdr.stop_recording()

        # Декодирование записанного
        if args.auto_decode:
            print("Декодирование записанного сигнала...")
            decoder = SSTVDecoder()
            
            # Пробуем декодировать из сэмплов если есть
            if hasattr(sdr, 'recorded_samples') and sdr.recorded_samples:
                import numpy as np
                all_samples = np.concatenate(sdr.recorded_samples)
                image = decoder.decode_from_samples(all_samples, sample_rate=sdr.sample_rate)
            else:
                image = decoder.decode_from_audio(output_audio)

            if image:
                output_image = args.output_image or "decoded_sstv.png"
                decoder.save_decoded_image(output_image)
                print(f"Изображение сохранено: {output_image}")
                
                # Показываем метаданные
                metadata = decoder.metadata
                if metadata:
                    print(f"Режим: {metadata.get('mode', 'unknown')}")
                    print(f"Размер: {image.size[0]}x{image.size[1]}")

        return True

    finally:
        sdr.close()


def mode_scan(args):
    """Режим сканирования частот."""
    freq_min = args.freq_min or 137
    freq_max = args.freq_max or 146
    step = args.step or 0.1

    print(f"Сканирование диапазона: {freq_min}-{freq_max} МГц, шаг {step} МГц")

    sdr = SDRInterface()
    if not sdr.initialize():
        print("Ошибка инициализации SDR")
        return False

    try:
        scanner = SDRScanner(sdr)
        signals = scanner.scan_frequencies(
            freq_range=(freq_min, freq_max),
            step_mhz=step,
            threshold_db=args.threshold or -80
        )

        if signals:
            print("\nНайденные сигналы:")
            for sig in signals:
                print(f"  {sig['frequency']:.3f} МГц - {sig['strength_db']:.1f} dB")

            # Самый сильный сигнал
            strongest = scanner.get_strongest_signal()
            if strongest:
                print(f"\nСамый сильный: {strongest['frequency']:.3f} МГц")

        # Сохраняем спектр
        if args.save_spectrum:
            scanner.plot_spectrum("spectrum.png")

        return True

    finally:
        sdr.close()


def mode_list_frequencies(args):
    """Показывает список предустановленных частот."""
    print("\nПредустановленные частоты:")
    print("-" * 40)

    frequencies = SDRInterface.FREQUENCIES
    for name, freq in frequencies.items():
        print(f"  {name:15} {freq:7.4f} МГц")

    print("\nИспользование:")
    print("  python main.py --frequency iss      # МКС")
    print("  python main.py --frequency noaa_15  # NOAA 15")
    print("  python main.py -f 145.800           # Своё значение")


def mode_demo(args):
    """Демонстрационный режим."""
    print("Доступные режимы:")
    print()
    print("  Декодирование аудио:")
    print("    python main.py --audio file.wav")
    print("    python main.py --audio file.wav --mode 'Martin 1'")
    print()
    print("  Обнаружение сигнала:")
    print("    python main.py --audio file.wav --detect")
    print()
    print("  Прием с SDR:")
    print("    python main.py --sdr --frequency 145.800")
    print("    python main.py --sdr -f iss --duration 30")
    print()
    print("  Сканирование частот:")
    print("    python main.py --scan")
    print("    python main.py --scan --freq-min 137 --freq-max 146")
    print()
    print("  Список частот:")
    print("    python main.py --list-freq")
    print()
    print("  Проверка устройства:")
    print("    python main.py --check")


def mode_check_device(args):
    """Проверка подключения RTL-SDR устройства."""
    print("\nПРОВЕРКА RTL-SDR УСТРОЙСТВА")
    print("-" * 40)
    
    # Проверка импорта
    print("1. Проверка rtlsdr...")
    try:
        from rtlsdr import RtlSdr
        print("   ✓ rtlsdr установлен")
    except ImportError:
        print("   ✗ rtlsdr не найден")
        print("   Установите: pip install rtlsdr pyrtlsdr")
        return False
    
    # Поиск устройств
    print("\n2. Поиск устройств...")
    try:
        num_devices = RtlSdr.get_device_count()
        print(f"   Найдено: {num_devices}")
        if num_devices == 0:
            print("   ⚠ Устройства не подключены")
            return False
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Информация об устройстве
    print("\n3. Информация об устройстве:")
    for i in range(num_devices):
        try:
            sdr = RtlSdr(device_index=i)
            device_name = sdr.get_device_name() if hasattr(sdr, 'get_device_name') else 'Unknown'
            serial = sdr.get_serial_number() if hasattr(sdr, 'get_serial_number') else 'Unknown'
            manufacturer = sdr.get_manufacturer() if hasattr(sdr, 'get_manufacturer') else 'Unknown'
            
            print(f"   Устройство #{i}:")
            print(f"      Название: {device_name}")
            print(f"      Серийный: {serial}")
            print(f"      Производитель: {manufacturer}")
            
            # Определение V4
            if 'R828D' in device_name.upper() or 'V4' in device_name.upper():
                print(f"      ✓ RTL-SDR V4 обнаружен")
            
            sdr.close()
        except Exception as e:
            print(f"   Устройство #{i}: Ошибка - {e}")
    
    print("\n✓ RTL-SDR готов к работе")
    return True


def main():
    """Основная функция SSTV станции."""
    parser = argparse.ArgumentParser(
        description="Наземная станция SSTV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Основные опции
    parser.add_argument("--audio", "-a", type=str, help="Путь к аудиофайлу")
    parser.add_argument("--output", "-o", type=str, help="Путь для сохранения изображения")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="auto",
        help="Режим SSTV (auto, 'Martin 1', 'Scottie 1', etc.)"
    )

    # SDR опции
    parser.add_argument("--sdr", action="store_true", help="Режим приема с SDR")
    parser.add_argument(
        "--frequency", "-f",
        type=str,
        default="iss",
        help="Частота (МГц) или название (iss, noaa_15, etc.)"
    )
    parser.add_argument("--device", "-d", type=int, default=0, help="Индекс SDR устройства")
    parser.add_argument("--sample-rate", type=int, default=2400000, help="Частота дискретизации (по умолчанию 2.4 MSPS)")
    parser.add_argument("--gain", type=int, default=30, help="Усиление в dB (0-50)")
    parser.add_argument("--bias-tee", action="store_true", help="Включить Bias-T для питания антенны")
    parser.add_argument("--agc", action="store_true", help="Включить автоматическую регулировку усиления")
    parser.add_argument("--duration", type=int, default=60, help="Длительность записи (с)")
    parser.add_argument("--output-audio", type=str, help="Файл для аудио записи")
    parser.add_argument("--output-image", type=str, help="Файл для изображения")
    parser.add_argument("--auto-decode", action="store_true", help="Авто декодирование после записи")

    # Сканирование
    parser.add_argument("--scan", action="store_true", help="Режим сканирования частот")
    parser.add_argument("--freq-min", type=float, help="Мин. частота сканирования")
    parser.add_argument("--freq-max", type=float, help="Макс. частота сканирования")
    parser.add_argument("--step", type=float, default=0.1, help="Шаг сканирования (МГц)")
    parser.add_argument("--threshold", type=float, default=-80, help="Порог обнаружения (dB)")
    parser.add_argument("--save-spectrum", action="store_true", help="Сохранить график спектра")

    # Другие опции
    parser.add_argument("--detect", action="store_true", help="Обнаружить SSTV сигнал")
    parser.add_argument("--list-freq", action="store_true", help="Список частот")
    parser.add_argument("--demo", action="store_true", help="Демонстрационный режим")
    parser.add_argument("--check", action="store_true", help="Проверка подключения RTL-SDR")

    args = parser.parse_args()

    show_banner()

    # Определяем режим работы
    if args.check:
        mode_check_device(args)
    elif args.list_freq:
        mode_list_frequencies(args)
    elif args.demo:
        mode_demo(args)
    elif args.scan:
        mode_scan(args)
    elif args.sdr:
        mode_receive_sdr(args)
    elif args.audio:
        mode_decode_audio(args)
    else:
        mode_demo(args)


if __name__ == "__main__":
    main()
