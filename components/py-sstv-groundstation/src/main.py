# -*- coding: utf-8 -*-
"""
Наземная станция SSTV
Скрипт для приема и декодирования SSTV-сигналов
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sstv_decoder import SSTVDecoder, detect_sstv_signal
from sdr_interface import SDRInterface, list_devices


def main():
    """Основная функция SSTV станции."""
    parser = argparse.ArgumentParser(
        description="Наземная станция SSTV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --audio recording.wav                 Декодировать аудиофайл
  %(prog)s --audio recording.wav --detect        Обнаружить SSTV сигнал
  %(prog)s --sdr --frequency 145.800             Прием с RTL-SDR (МКС)
  %(prog)s --sdr --scan                          Сканирование диапазона
  %(prog)s --list-devices                        Список RTL-SDR устройств
        """,
    )

    # Режимы работы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--audio", "-a", type=str, help="Путь к аудиофайлу")
    mode_group.add_argument("--sdr", "-s", action="store_true", help="Режим приема с RTL-SDR")
    mode_group.add_argument("--scan", action="store_true", help="Сканирование диапазона")
    mode_group.add_argument("--list-devices", action="store_true", help="Список RTL-SDR устройств")

    # Параметры SDR
    parser.add_argument(
        "--device-index", "-d", type=int, default=0, help="Индекс RTL-SDR устройства (0 по умолчанию)"
    )
    parser.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=145.800,
        help="Частота приема в МГц (по умолчанию: 145.800 - МКС)",
    )
    parser.add_argument(
        "--gain", "-g", type=int, default=30, help="Усиление RTL-SDR в дБ (0-50, по умолчанию: 30)"
    )
    parser.add_argument(
        "--duration",
        "-t",
        type=float,
        default=30.0,
        help="Длительность записи в секундах (по умолчанию: 30)",
    )

    # Параметры сканирования
    parser.add_argument("--scan-start", type=float, default=145.0, help="Начало сканирования в МГц")
    parser.add_argument("--scan-end", type=float, default=146.0, help="Конец сканирования в МГц")
    parser.add_argument("--scan-step", type=float, default=0.001, help="Шаг сканирования в МГц")

    # Общие параметры
    parser.add_argument("--output", "-o", type=str, help="Путь для сохранения (изображение/аудио)")
    parser.add_argument("--detect", action="store_true", help="Обнаружить SSTV сигнал")
    parser.add_argument("--auto-decode", action="store_true", help="Автоматически декодировать после записи")

    args = parser.parse_args()

    print("=" * 50)
    print("       НАЗЕМНАЯ СТАНЦИЯ SSTV")
    print("       Поддержка RTL-SDR V4")
    print("=" * 50)

    # Список устройств
    if args.list_devices:
        devices = list_devices()
        if not devices:
            print("\nУстройства не найдены. Проверьте подключение RTL-SDR.")
            print("Убедитесь, что драйверы установлены (Zadig для Windows).")
        sys.exit(0)

    decoder = SSTVDecoder()

    # Режим декодирования аудио
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Ошибка: Файл '{audio_path}' не найден")
            sys.exit(1)

        if args.detect:
            print(f"\nОбнаружение SSTV сигнала в: {audio_path}")
            result = detect_sstv_signal(str(audio_path))
            if result:
                print(f"SSTV сигнал обнаружен: {result}")
            else:
                print("SSTV сигнал не обнаружен")
        else:
            print(f"\nДекодирование аудио: {audio_path}")
            image = decoder.decode_from_audio(str(audio_path))

            if image:
                output = args.output or "decoded_sstv_image.png"
                decoder.save_decoded_image(output)
                print(f"Изображение сохранено в: {output}")
            else:
                print("Не удалось декодировать изображение")

    # Режим приема с RTL-SDR
    elif args.sdr:
        print(f"\nРежим приема с RTL-SDR")
        print(f"Частота: {args.frequency:.3f} МГц")
        print(f"Усиление: {args.gain} дБ")
        print(f"Длительность: {args.duration} сек")

        sdr = SDRInterface(device_index=args.device_index)

        if not sdr.initialize():
            print("\nНе удалось инициализировать RTL-SDR")
            print("Проверьте:")
            print("  1. Подключено ли устройство")
            print("  2. Установлены ли драйверы (pip install rtlsdr pyrtlsdr)")
            print("  3. Для Windows: используйте Zadig для установки WinUSB драйвера")
            sys.exit(1)

        try:
            # Настройка параметров
            sdr.set_frequency(args.frequency)
            sdr.set_gain(args.gain)

            # Запись сигнала
            output_file = args.output or f"sstv_recording_{args.frequency:.3f}MHz.wav"
            success = sdr.record_audio(duration_sec=args.duration, output_file=output_file)

            if success and args.auto_decode:
                print("\nАвтоматическое декодирование...")
                image = decoder.decode_from_audio(output_file)
                if image:
                    img_output = output_file.replace(".wav", "_decoded.png")
                    decoder.save_decoded_image(img_output)
                    print(f"Декодированное изображение: {img_output}")

        except KeyboardInterrupt:
            print("\nПрием прерван")
        finally:
            sdr.close()

    # Режим сканирования
    elif args.scan:
        print(f"\nРежим сканирования диапазона")
        print(f"Диапазон: {args.scan_start:.3f} - {args.scan_end:.3f} МГц")
        print(f"Шаг: {args.scan_step:.3f} МГц")

        sdr = SDRInterface(device_index=args.device_index)

        if not sdr.initialize():
            print("Не удалось инициализировать RTL-SDR")
            sys.exit(1)

        try:
            signals = sdr.scan_frequencies(
                start_mhz=args.scan_start, end_mhz=args.scan_end, step_mhz=args.scan_step
            )

            if signals:
                print(f"\nНайдено сигналов: {len(signals)}")
                for freq, power in signals:
                    print(f"  {freq:.3f} МГц - {power:.1f} дБ")
            else:
                print("\nСигналы не обнаружены")

        finally:
            sdr.close()

    # Режим по умолчанию - справка
    else:
        print("\nРежимы работы:")
        print("  --audio <файл>           Декодировать аудиофайл")
        print("  --audio <файл> --detect  Обнаружить SSTV сигнал в аудио")
        print("  --sdr --frequency <МГц>  Прием с RTL-SDR")
        print("  --scan                   Сканирование диапазона")
        print("  --list-devices           Показать доступные RTL-SDR устройства")
        print("\nБыстрый старт с RTL-SDR:")
        print("  python main.py --sdr --frequency 145.800 --duration 60")
        print("  python main.py --sdr --frequency 145.800 --auto-decode")


if __name__ == "__main__":
    main()
