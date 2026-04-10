#!/usr/bin/env python3
"""
SSTV Ground Station — Захват и декодирование через MMSSTV

Автоматизирует полный цикл:
1. Захват сигнала с МКС (145.800 MHz) через rtl_fm
2. Сохранение в WAV формате
3. Автоматическое открытие MMSSTV для декодирования

Использование:
    python sstv_mmsstv_capture.py              # Захват 60 секунд
    python sstv_mmsstv_capture.py -d 120       # Захват 120 секунд
    python sstv_mmsstv_capture.py -f 145.800   # Другая частота
    python sstv_mmsstv_capture.py --no-decode  # Без автооткрытия MMSSTV
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Конфигурация ─────────────────────────────────────────────────────────────

FREQUENCY = 145.800  # МКС SSTV (MHz)
DURATION = 60        # секунд
GAIN = 40            # dB
SAMPLE_RATE = 22050  # Hz (стандарт для SSTV)

RTL_FM_PATHS = [
    Path("rtl_fm.exe"),
    Path(r"C:\rtl-sdr\bin\x64\rtl_fm.exe"),
    Path(r"C:\Program Files\rtl-sdr\rtl_fm.exe"),
    Path(r"C:\rtlsdr\rtl_fm.exe"),
]

MMSSTV_PATH = Path(r"C:\Ham\MMSSTV\MMSSTV.EXE")

OUTPUT_DIR = Path("data/sstv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Функции ──────────────────────────────────────────────────────────────────


def find_rtl_fm() -> Path:
    """Найти rtl_fm.exe"""
    for path in RTL_FM_PATHS:
        if path.exists():
            return path
    
    # Проверить PATH
    import shutil
    found = shutil.which("rtl_fm") or shutil.which("rtl_fm.exe")
    if found:
        return Path(found)
    
    return None


def capture_sstv(
    frequency: float,
    duration: int,
    gain: int,
    sample_rate: int,
    output_file: Path,
) -> bool:
    """
    Захват SSTV сигнала через rtl_fm.
    
    Returns:
        True если успешно
    """
    rtl_fm = find_rtl_fm()
    if not rtl_fm:
        print("[!] rtl_fm.exe не найден!")
        print("    Установите RTL-SDR драйверы:")
        print("    https://github.com/osmocom/rtl-sdr/releases")
        return False

    print(f"[*] Используем: {rtl_fm}")
    print(f"[*] Частота: {frequency} MHz")
    print(f"[*] Длительность: {duration} секунд")
    print(f"[*] Gain: {gain} dB")
    print(f"[*] Выход: {output_file}")
    print()

    # rtl_fm команда
    cmd = [
        str(rtl_fm),
        "-f", f"{frequency}M",
        "-M", "fm",
        "-s", str(sample_rate),
        "-r", str(sample_rate),
        "-g", str(gain),
        "-d", "0",
        "-l", "0",  # squelch off
    ]

    print(f"[*] Команда: {' '.join(cmd)} > {output_file.name}")
    print(f"[*] Запись {duration} секунд...")
    print(f"[*] Нажмите Ctrl+C для остановки")
    print()

    start_time = time.time()

    try:
        with open(output_file, "wb") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
            )

            # Прогресс-бар
            for i in range(duration):
                time.sleep(1)
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                print(
                    f"\r  [{'=' * i}{' ' * (duration - i)}] "
                    f"{elapsed}/{duration}s (осталось {remaining}s)",
                    end="",
                    flush=True,
                )
                # Проверка что процесс жив
                if process.poll() is not None:
                    print(f"\n[!] rtl_fm завершился неожиданно!")
                    stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                    print(f"    stderr: {stderr_output[:500]}")
                    return False

            # Время вышло
            print(f"\n\n[*] Остановка записи...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    except KeyboardInterrupt:
        print(f"\n\n[*] Пользователь остановил запись!")
        process.kill()
        process.wait()

    # Проверка результата
    if output_file.exists() and output_file.stat().st_size > 0:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        duration_actual = size_mb * 8 / (sample_rate * 2 * 16 / 8)  # приблизительная длительность
        print(f"\n✅ OK!")
        print(f"    Файл: {output_file}")
        print(f"    Размер: {size_mb:.1f} MB")
        print(f"    Длительность: ~{duration_actual:.0f} секунд")
        return True
    else:
        print(f"\n❌ ОШИБКА: Файл не создан или пустой!")
        print("    Проверьте:")
        print("    - RTL-SDR подключён")
        print("    - Не используется другим приложением")
        print("    - Антенна подключена")
        return False


def open_mmsstv(wav_file: Path) -> bool:
    """
    Открыть WAV файл в MMSSTV для декодирования.
    
    MMSSTV автоматически распознаёт SSTV режим и декодирует изображение.
    """
    if not MMSSTV_PATH.exists():
        print(f"\n[!] MMSSTV не найден: {MMSSTV_PATH}")
        print("    Установите MMSSTV:")
        print("    http://hamsoft.ca/pages/mmsstv.php")
        print()
        print("    Ручное декодирование:")
        print(f"    1. Откройте MMSSTV")
        print(f"    2. Menu -> RxAudio -> Load File -> {wav_file}")
        print(f"    3. MMSSTV автоматически распознает SSTV режим")
        return False

    print(f"\n[*] Открываю MMSSTV...")
    print(f"    {MMSSTV_PATH}")
    print(f"    Файл: {wav_file}")
    print()
    print("[*] В MMSSTV:")
    print("    - Меню -> RxAudio -> Load File -> выберите файл")
    print("    - Или перетащите файл в окно MMSSTV")
    print()

    # Запуск MMSSTV
    # MMSSTV не принимает файл как аргумент, поэтому просто откроем программу
    subprocess.Popen([str(MMSSTV_PATH)])
    
    print(f"[*] MMSSTV запущен!")
    print(f"    Теперь загрузите файл: {wav_file}")
    
    return True


def list_recordings(limit: int = 10):
    """Показать последние записи"""
    if not OUTPUT_DIR.exists():
        print("Нет записей")
        return
    
    files = sorted(OUTPUT_DIR.glob("iss_sstv_*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not files:
        print("Нет записей SSTV")
        return
    
    print(f"Последние {min(len(files), limit)} записей SSTV:")
    print()
    print(f"{'Файл':<50} {'Размер':>10} {'Дата':>20}")
    print("-" * 85)
    
    for f in files[:limit]:
        size_mb = f.stat().st_size / (1024 * 1024)
        date_str = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{f.name:<50} {size_mb:>8.1f} MB  {date_str:>20}")


# ── Главная ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SSTV Ground Station — Захват и декодирование через MMSSTV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Захват 60 секунд с автооткрытием MMSSTV
  python sstv_mmsstv_capture.py

  # Захват 120 секунд
  python sstv_mmsstv_capture.py -d 120

  # Другая частота (NOAA)
  python sstv_mmsstv_capture.py -f 137.100

  # Только запись, без MMSSTV
  python sstv_mmsstv_capture.py --no-decode

  # Показать последние записи
  python sstv_mmsstv_capture.py --list
        """,
    )

    parser.add_argument(
        "-f", "--frequency",
        type=float,
        default=FREQUENCY,
        help=f"Частота MHz (по умолчанию: {FREQUENCY})",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=DURATION,
        help=f"Длительность секунд (по умолчанию: {DURATION})",
    )
    parser.add_argument(
        "-g", "--gain",
        type=int,
        default=GAIN,
        help=f"Gain dB (по умолчанию: {GAIN})",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Не открывать MMSSTV автоматически",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать последние записи",
    )

    args = parser.parse_args()

    # Режим списка
    if args.list:
        list_recordings()
        return

    # Захват
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"iss_sstv_{timestamp}.wav"

    print("=" * 70)
    print("SSTV Ground Station — Захват и декодирование")
    print("=" * 70)
    print()

    success = capture_sstv(
        frequency=args.frequency,
        duration=args.duration,
        gain=args.gain,
        sample_rate=SAMPLE_RATE,
        output_file=output_file,
    )

    if success and not args.no_decode:
        open_mmsstv(output_file)


if __name__ == "__main__":
    main()
