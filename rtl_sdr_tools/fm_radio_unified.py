#!/usr/bin/env python3
"""
FM Радио — Универсальный модуль (RTL-SDR)

Режимы:
    1. Приёмник — прослушивание FM радио в реальном времени
    2. Захват — запись аудио в WAV файл
    3. Сканирование — поиск станций в диапазоне 88-108 МГц
    4. Мультизахват — запись нескольких станций последовательно

Использование:
    python fm_radio_unified.py listen --freq 106.0           # Слушать
    python fm_radio_unified.py capture --freq 106.0 -d 10    # Записать 10 сек
    python fm_radio_unified.py scan                          # Сканировать
    python fm_radio_unified.py multi --stations 106.0 100.5  # Мультизахват
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# RTL-SDR путь (Windows)
RTL_FM_PATH = r"M:\GitHub\nanoprobe-sim-lab\tools\rtl-sdr-blog\x64\rtl_fm.exe"

try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    print("⚠️  pyrtlsdr не установлен: pip install pyrtlsdr")

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Известные Moscow FM станции
MOSCOW_FM_STATIONS: Dict[str, float] = {
    "Europa Plus": 100.5,
    "Russian Radio": 101.5,
    "DFM": 101.1,
    "Hit FM": 100.9,
    "Radio Record": 94.0,
    "Radio Energy": 95.5,
    "Retro FM": 92.4,
    "Nashe Radio": 93.1,
    "Radio Maximum": 102.3,
    "Radio Monte Carlo": 103.1,
    "Radio Kultura": 89.1,
    "Radio Mayak": 90.1,
    "Radio 7": 97.0,
    "Like FM": 95.0,
    "Radio Jazz": 99.2,
    "Marusya FM": 101.9,
    "Radio Romantika": 102.7,
    "Voyage FM": 96.6,
    "Studio 21": 94.5,
    "Radio Relax FM": 96.0,
}


# ============================================================
# РЕЖИМ 1: Приёмник реального времени
# ============================================================


class FMRadioReceiver:
    """FM радио приёмник с RTL-SDR для прослушивания в реальном времени"""

    def __init__(
        self,
        frequency_mhz: float = 106.0,
        sample_rate: int = 2400000,
        audio_rate: int = 48000,
        gain: int = 30,
        device_index: int = 0,
    ):
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.gain = gain
        self.device_index = device_index
        self.sdr: Optional[RtlSdr] = None
        self.audio_stream = None
        self._running = False
        self._deemphasis_alpha = 75e-6 / (75e-6 + 1.0 / self.audio_rate)

    def _fm_demodulate(self, samples: np.ndarray) -> np.ndarray:
        """FM демодуляция через разность фаз"""
        phase = np.angle(samples)
        audio = np.diff(phase)
        return (audio / (np.max(np.abs(audio)) + 1e-10)).astype(np.float32)

    def _deemphasize(self, audio: np.ndarray) -> np.ndarray:
        """De-emphasis фильтр 75 μs"""
        alpha = self._deemphasis_alpha
        out = np.zeros_like(audio)
        out[0] = audio[0]
        for i in range(1, len(audio)):
            out[i] = alpha * audio[i] + (1 - alpha) * out[i - 1]
        return out

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Ресэмплинг к audio_rate"""
        from scipy import signal as sig

        sos = sig.butter(8, 20000, "lowpass", fs=self.sample_rate, output="sos")
        filtered = sig.sosfiltfilt(sos, audio)
        return sig.resample_poly(filtered, self.audio_rate, self.sample_rate).astype(np.float32)

    def start(self):
        """Запуск приёмника"""
        if not RTLSDR_AVAILABLE or not SOUNDDEVICE_AVAILABLE:
            logger.error("RTL-SDR или sounddevice не доступен")
            return

        logger.info(f"📻 FM Receiver — {self.frequency_mhz:.1f} MHz")
        try:
            self.sdr = RtlSdr(device_index=self.device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency_hz
            self.sdr.gain = self.gain

            self.audio_stream = sd.OutputStream(
                samplerate=self.audio_rate, channels=1, dtype="float32"
            )
            self.audio_stream.start()

            signal.signal(signal.SIGINT, lambda s, f: self.stop())
            signal.signal(signal.SIGTERM, lambda s, f: self.stop())

            logger.info("🎵 Воспроизведение... (Ctrl+C для остановки)")
            self._running = True
            self._receive_loop()
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
        finally:
            self.stop()

    def _receive_loop(self):
        """Основной цикл приёма"""
        audio_per_buffer = int(self.audio_rate * 0.1)
        iq_per_buffer = (int(audio_per_buffer * self.sample_rate / self.audio_rate) // 2) * 2

        while self._running:
            try:
                samples = self.sdr.read_samples(iq_per_buffer)
                audio = self._fm_demodulate(samples)
                audio = self._deemphasize(audio)
                audio = self._resample(audio)
                if self.audio_stream:
                    self.audio_stream.write(audio)
            except Exception as e:
                if self._running:
                    logger.error(f"Ошибка: {e}")
                break

    def stop(self):
        """Остановка"""
        self._running = False
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass
        logger.info("👋 Остановлен")


# ============================================================
# РЕЖИМ 2: Захват в WAV файл
# ============================================================


def capture_fm_to_wav(
    frequency_mhz: float,
    duration_sec: int = 10,
    gain: int = 30,
    output_file: Optional[str] = None,
    device_index: int = 0,
) -> str:
    """Захват FM радио в WAV файл"""
    if not RTLSDR_AVAILABLE:
        logger.error("RTL-SDR не доступен")
        return ""

    if output_file is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        station_name = next(
            (n for n, f in MOSCOW_FM_STATIONS.items() if abs(f - frequency_mhz) < 0.1),
            f"{frequency_mhz:.1f}",
        )
        output_file = f"data/fm_radio/{station_name}_{ts}.wav"

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"📡 Захват: {frequency_mhz:.1f} MHz, {duration_sec} сек")
    sdr = RtlSdr(device_index=device_index)
    sdr.sample_rate = 2400000
    sdr.center_freq = frequency_mhz * 1e6
    sdr.gain = gain

    audio_rate = 44100
    total_samples = int(audio_rate * duration_sec)
    audio_buffer = np.zeros(total_samples, dtype=np.float32)
    idx = 0

    try:
        while idx < total_samples:
            samples = sdr.read_samples(240000)
            phase = np.angle(samples)
            audio = np.diff(phase)
            audio = audio / (np.max(np.abs(audio)) + 1e-10)

            # De-emphasis
            alpha = 75e-6 / (75e-6 + 1.0 / audio_rate)
            for i in range(1, len(audio)):
                audio[i] = alpha * audio[i] + (1 - alpha) * audio[i - 1]

            chunk_size = min(len(audio), total_samples - idx)
            audio_buffer[idx : idx + chunk_size] = audio[:chunk_size]
            idx += chunk_size

            progress = min(100, int(idx / total_samples * 100))
            print(f"\r📊 Захват: {progress}%", end="", flush=True)
    finally:
        sdr.close()

    # Нормализация и сохранение
    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0:
        audio_buffer = (audio_buffer / max_val * 0.9 * 32767).astype(np.int16)

    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio_rate)
        wf.writeframes(audio_buffer.tobytes())

    print(f"\n✅ Сохранено: {output_file}")
    return output_file


# ============================================================
# РЕЖИМ 3: Сканирование FM диапазона
# ============================================================


def scan_fm_band(
    start_mhz: float = 88.0,
    end_mhz: float = 108.0,
    step_mhz: float = 0.1,
    gain: int = 30,
    device_index: int = 0,
) -> List[Tuple[float, float]]:
    """Сканирование FM диапазона с поиском станций"""
    if not RTLSDR_AVAILABLE:
        logger.error("RTL-SDR не доступен")
        return []

    logger.info(f"📡 Сканирование: {start_mhz:.1f} - {end_mhz:.1f} MHz")
    sdr = RtlSdr(device_index=device_index)
    sdr.sample_rate = 2400000
    sdr.gain = gain

    stations = []
    freq = start_mhz

    while freq <= end_mhz:
        sdr.center_freq = freq * 1e6
        samples = sdr.read_samples(1024 * 256)
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-10)

        # Определяем имя станции если известна
        station_name = next((n for n, f in MOSCOW_FM_STATIONS.items() if abs(f - freq) < 0.05), "")
        marker = f" — {station_name}" if station_name else ""

        if power_db > -30:
            stations.append((freq, power_db))
            logger.info(f"  📻 {freq:.1f} MHz: {power_db:.1f} dB ** СТАНЦИЯ **{marker}")

        freq += step_mhz

    sdr.close()

    if stations:
        logger.info(f"\n✅ Найдено {len(stations)} станций:")
        for freq, power_db in sorted(stations, key=lambda x: x[1], reverse=True):
            name = next(
                (n for n, f in MOSCOW_FM_STATIONS.items() if abs(f - freq) < 0.05), "Unknown"
            )
            logger.info(f"  {freq:.1f} MHz ({power_db:.1f} dB) — {name}")
    else:
        logger.info("\n⚠️  Станций не найдено. Попробуйте изменить порог или усиление.")

    return stations


# ============================================================
# РЕЖИМ 4: Мультизахват (несколько станций)
# ============================================================


def capture_multiple_stations(
    frequencies: List[float],
    duration_sec: int = 8,
    gain: int = 30,
) -> List[str]:
    """Последовательный захват нескольких станций"""
    output_files = []

    for freq in frequencies:
        station_name = next(
            (n for n, f in MOSCOW_FM_STATIONS.items() if abs(f - freq) < 0.1), f"{freq:.1f}"
        )
        logger.info(f"\n{'='*50}")
        logger.info(f"📡 Захват: {station_name} @ {freq:.1f} MHz")
        logger.info(f"{'='*50}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"data/fm_radio/{station_name}_{ts}.wav"

        result = capture_fm_to_wav(
            frequency_mhz=freq,
            duration_sec=duration_sec,
            gain=gain,
            output_file=output_file,
        )
        if result:
            output_files.append(result)

    logger.info(f"\n✅ Захвачено {len(output_files)}/{len(frequencies)} станций")
    return output_files


# ============================================================
# CLI интерфейс
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="FM Радио — Универсальный модуль (RTL-SDR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Слушать Europa Plus
  python fm_radio_unified.py listen --freq 100.5

  # Записать 30 секунд
  python fm_radio_unified.py capture --freq 106.0 -d 30

  # Сканировать диапазон
  python fm_radio_unified.py scan

  # Записать несколько станций
  python fm_radio_unified.py multi --freqs 106.0 100.5 101.5
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Режим работы")

    # listen
    p_listen = subparsers.add_parser("listen", help="Слушать FM радио")
    p_listen.add_argument("--freq", type=float, default=106.0, help="Частота МГц")
    p_listen.add_argument("--gain", type=int, default=30, help="Усиление dB")
    p_listen.add_argument("--device", type=int, default=0, help="Индекс устройства")

    # capture
    p_capture = subparsers.add_parser("capture", help="Записать в WAV")
    p_capture.add_argument("--freq", type=float, default=106.0, help="Частота МГц")
    p_capture.add_argument("--gain", type=int, default=30, help="Усиление dB")
    p_capture.add_argument("-d", "--duration", type=int, default=10, help="Длительность сек")
    p_capture.add_argument("-o", "--output", type=str, help="Имя файла")
    p_capture.add_argument("--device", type=int, default=0, help="Индекс устройства")

    # scan
    p_scan = subparsers.add_parser("scan", help="Сканировать FM диапазон")
    p_scan.add_argument("--start", type=float, default=88.0, help="Начало МГц")
    p_scan.add_argument("--end", type=float, default=108.0, help="Конец МГц")
    p_scan.add_argument("--step", type=float, default=0.1, help="Шаг МГц")
    p_scan.add_argument("--gain", type=int, default=30, help="Усиление dB")
    p_scan.add_argument("--device", type=int, default=0, help="Индекс устройства")

    # multi
    p_multi = subparsers.add_parser("multi", help="Мультизахват станций")
    p_multi.add_argument("--freqs", type=float, nargs="+", help="Частоты МГц")
    p_multi.add_argument("--stations", type=str, nargs="+", help="Имена станций (Europa Plus, ...)")
    p_multi.add_argument("-d", "--duration", type=int, default=8, help="Длительность сек")
    p_multi.add_argument("--gain", type=int, default=30, help="Усиление dB")

    args = parser.parse_args()

    if args.mode == "listen":
        receiver = FMRadioReceiver(
            frequency_mhz=args.freq,
            gain=args.gain,
            device_index=args.device,
        )
        receiver.start()

    elif args.mode == "capture":
        capture_fm_to_wav(
            frequency_mhz=args.freq,
            duration_sec=args.duration,
            gain=args.gain,
            output_file=args.output,
            device_index=args.device,
        )

    elif args.mode == "scan":
        scan_fm_band(
            start_mhz=args.start,
            end_mhz=args.end,
            step_mhz=args.step,
            gain=args.gain,
            device_index=args.device,
        )

    elif args.mode == "multi":
        frequencies = list(args.freqs) if args.freqs else []
        if args.stations:
            for name in args.stations:
                if name in MOSCOW_FM_STATIONS:
                    frequencies.append(MOSCOW_FM_STATIONS[name])
                else:
                    logger.warning(f"⚠️  Станция '{name}' не найдена")
        if not frequencies:
            logger.error("❌ Укажите --freqs или --stations")
            sys.exit(1)
        capture_multiple_stations(frequencies, duration_sec=args.duration, gain=args.gain)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
