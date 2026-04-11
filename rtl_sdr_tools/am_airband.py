#!/usr/bin/env python3
"""
Авиадиапазон AM приёмник через RTL-SDR (118-137 МГц)

Позволяет слушать:
- Диспетчеров аэропортов
- Пилотов
- ATIS (автоматическая информация)
- Ground control

Использование:
    python am_airband.py --freq 127.5    # Прослушивание
    python am_airband.py --scan           # Сканирование
"""

import argparse
import logging
import signal
from typing import Optional

import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  sounddevice не установлен: pip install sounddevice")

try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    print("⚠️  pyrtlsdr не установлен: pip install pyrtlsdr")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AMAirbandReceiver:
    """
    AM авиадиапазон приёмник.

    Частоты: 118.000 - 136.975 МГц
    Шаг: 8.33 кГц (Европа) или 25 кГц (США)
    Модуляция: AM
    """

    def __init__(
        self,
        frequency_mhz: float = 127.5,
        sample_rate: int = 2400000,
        audio_rate: int = 48000,
        gain: int = 30,
        device_index: int = 0,
    ):
        """
        Инициализация AM приёмника.

        Args:
            frequency_mhz: Частота в МГц (118-137)
            sample_rate: Частота дискретизации RTL-SDR
            audio_rate: Частота аудио вывода
            gain: Усиление RTL-SDR
            device_index: Индекс устройства
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.gain = gain
        self.device_index = device_index

        self.sdr: Optional[RtlSdr] = None
        self.audio_stream = None
        self._running = False

    def _am_demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        AM демодуляция.

        AM: аудио = огибающая I/Q сигнала
        envelope = sqrt(I² + Q²) = |samples|

        Args:
            samples: Комплексные I/Q сэмплы

        Returns:
            Аудио сигнал (float32)
        """
        # Огибающая = амплитуда
        audio = np.abs(samples)

        # DC removal (убираем постоянную составляющую)
        audio = audio - np.mean(audio)

        # Нормализация к [-1, 1]
        max_val = np.max(np.abs(audio)) + 1e-10
        audio = audio / max_val

        return audio.astype(np.float32)

    def _apply_audio_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Полосовой фильтр для голоса (300-3000 Hz).

        Авиасвязь использует узкую полосу для голоса.
        """
        from scipy import signal as scipy_signal

        # Bandpass 300-3000 Hz
        lowcut = 300
        highcut = 3000
        nyquist = self.audio_rate / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist

        # Butterworth 4-го порядка
        sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, audio)

        return filtered

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Ресэмплинг к аудио частоте"""
        from scipy import signal as scipy_signal

        resampled = scipy_signal.resample_poly(audio, self.audio_rate, self.sample_rate)
        return resampled.astype(np.float32)

    def start(self):
        """Запуск приёмника"""
        if not RTLSDR_AVAILABLE:
            logger.error("RTL-SDR не доступен")
            return

        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice не доступен")
            return

        logger.info(f"🛩️  AM Airband Receiver — {self.frequency_mhz:.3f} MHz")
        logger.info(f"📡 Инициализация RTL-SDR...")

        try:
            # Инициализация RTL-SDR
            self.sdr = RtlSdr(device_index=self.device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency_hz
            self.sdr.gain = self.gain

            device_name = self.sdr.get_device_name()
            serial = self.sdr.get_serial_number()

            logger.info(f"✅ Устройство: {device_name} (SN: {serial})")
            logger.info(f"📡 Частота: {self.frequency_mhz:.3f} MHz")
            logger.info(f"📊 Sample rate: {self.sample_rate / 1e6:.1f} MSPS")
            logger.info(f"🔊 Gain: {self.gain} dB")
            logger.info(f"🎧 Audio: {self.audio_rate} Hz")
            logger.info("")
            logger.info("🎙️  Воспроизведение... (Ctrl+C для остановки)")

            # Аудио поток
            self.audio_stream = sd.OutputStream(
                samplerate=self.audio_rate,
                channels=1,
                dtype="float32",
            )
            self.audio_stream.start()

            # Сигналы
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self._running = True
            self._receive_loop()

        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def _receive_loop(self):
        """Основной цикл приёма"""
        # ~100ms аудио
        audio_samples = int(self.audio_rate * 0.1)
        iq_samples = int(audio_samples * self.sample_rate / self.audio_rate)
        iq_samples = (iq_samples // 2) * 2  # Чётное

        logger.info(f"📦 Буфер: {iq_samples} I/Q → {audio_samples} аудио")

        while self._running:
            try:
                # Чтение I/Q
                samples = self.sdr.read_samples(iq_samples)

                # AM демодуляция
                audio = self._am_demodulate(samples)

                # Фильтрация голоса
                audio = self._apply_audio_filter(audio)

                # Ресэмплинг
                audio = self._resample(audio)

                # Воспроизведение
                if self.audio_stream and self._running:
                    self.audio_stream.write(audio)

            except Exception as e:
                if self._running:
                    logger.error(f"Ошибка в цикле: {e}")
                break

    def _signal_handler(self, signum, frame):
        logger.info("\n🛑 Остановка...")
        self._running = False

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

        logger.info("👋 AM приёмник остановлен")


def scan_airband(
    start_mhz: float = 118.0,
    end_mhz: float = 137.0,
    step_mhz: float = 0.025,
    gain: int = 30,
    device_index: int = 0,
):
    """
    Сканирование авиадиапазона.

    Args:
        start_mhz: Начальная частота
        end_mhz: Конечная частота
        step_mhz: Шаг (25 кГц или 8.33 кГц)
        gain: Усиление
        device_index: Индекс устройства
    """
    if not RTLSDR_AVAILABLE:
        logger.error("RTL-SDR не доступен")
        return

    logger.info(f"🛩️  Сканирование авиадиапазона: {start_mhz:.3f} - {end_mhz:.3f} MHz")
    logger.info(f"📊 Шаг: {step_mhz * 1000:.2f} кГц")

    sdr = RtlSdr(device_index=device_index)
    sdr.sample_rate = 2400000
    sdr.gain = gain

    channels = []

    freq = start_mhz
    while freq <= end_mhz:
        sdr.center_freq = freq * 1e6

        # Чтение сэмплов
        samples = sdr.read_samples(1024 * 256)

        # Мощность сигнала
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-10)

        # Детектор активности
        if power_db > -35:  # Порог
            channels.append((freq, power_db))
            logger.info(f"  📻 {freq:.3f} MHz: {power_db:.1f} dB ** АКТИВНЫЙ КАНАЛ **")

        freq += step_mhz

    sdr.close()

    if channels:
        logger.info(f"\n✅ Найдено {len(channels)} активных каналов:")
        for freq, power_db in channels:
            logger.info(f"  {freq:.3f} MHz ({power_db:.1f} dB)")
    else:
        logger.info("\n⚠️  Активных каналов не найдено")


def main():
    parser = argparse.ArgumentParser(
        description="AM Авиадиапазон через RTL-SDR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Слушать 127.5 MHz
  python am_airband.py --freq 127.5

  # Сканирование всего диапазона
  python am_airband.py --scan

  # С усилением
  python am_airband.py --freq 121.5 --gain 40
        """,
    )

    parser.add_argument(
        "--freq",
        type=float,
        default=127.5,
        help="Частота в МГц (118-137, по умолчанию: 127.5)",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=30,
        help="Усиление RTL-SDR dB (0-50, по умолчанию: 30)",
    )
    parser.add_argument("--device", type=int, default=0, help="Индекс RTL-SDR устройства")
    parser.add_argument("--scan", action="store_true", help="Сканирование авиадиапазона")

    args = parser.parse_args()

    if args.scan:
        scan_airband(gain=args.gain, device_index=args.device)
    else:
        # Проверка диапазона
        if not (118.0 <= args.freq <= 137.0):
            logger.error(f"❌ Частота {args.freq} вне диапазона 118-137 МГц")
            return

        receiver = AMAirbandReceiver(
            frequency_mhz=args.freq,
            gain=args.gain,
            device_index=args.device,
        )
        receiver.start()


if __name__ == "__main__":
    main()
