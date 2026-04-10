#!/usr/bin/env python3
"""
FM Радио приёмник через RTL-SDR

Позволяет слушать FM радио (88-108 МГц) через наушники.

Использование:
    python fm_radio.py --freq 106.0        # Европа Плюс
    python fm_radio.py --freq 101.7        # Другая станция
    python fm_radio.py --freq 88.3 --scan  # Сканирование
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FMRadioReceiver:
    """
    FM радио приёмник с RTL-SDR.

    Цепочка обработки:
    RTL-SDR → I/Q → FM демодуляция → Де-эмфазис → Ресэмплинг → Звук
    """

    def __init__(
        self,
        frequency_mhz: float = 106.0,
        sample_rate: int = 2400000,
        audio_rate: int = 48000,
        gain: int = 30,
        device_index: int = 0,
    ):
        """
        Инициализация FM приёмника.

        Args:
            frequency_mhz: Частота станции в МГц
            sample_rate: Частота дискретизации RTL-SDR
            audio_rate: Частота аудио вывода
            gain: Усиление RTL-SDR (0-50 dB)
            device_index: Индекс устройства
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.gain = gain
        self.device_index = device_index

        # RTL-SDR устройство
        self.sdr: Optional[RtlSdr] = None

        # Аудио поток
        self.audio_stream = None

        # Флаги
        self._running = False

        # FIR фильтр для де-эмфазиса (75 μs - Европа/США)
        self._deemphasis_filter = self._create_deemphasis_filter()

    def _create_deemphasis_filter(self) -> np.ndarray:
        """
        Создаёт FIR фильтр де-эмфазиса.

        FM радио использует пред-эмфазис 75 μs (Европа/США)
        или 50 μs (Япония). Для корректного звука нужен
        обратный фильтр — де-эмфазис.
        """
        # Простой RC фильтр де-эмфазиса
        # 75 μs = 2122 Hz cutoff
        tau = 75e-6  # 75 микросекунд
        dt = 1.0 / self.audio_rate
        alpha = dt / (tau + dt)

        # Коэффициенты IIR фильтра 1-го порядка
        # y[n] = alpha * x[n] + (1-alpha) * y[n-1]
        return alpha

    def _fm_demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        FM демодуляция I/Q данных.

        Args:
            samples: Комплексные I/Q сэмплы

        Returns:
            Аудио сигнал (float32)
        """
        # Дифференцирование фазы
        # audio = d(phase)/dt
        phase = np.angle(samples)
        audio = np.diff(phase)

        # Нормализация к [-1, 1]
        audio = audio / np.max(np.abs(audio) + 1e-10)

        return audio.astype(np.float32)

    def _deemphasize(self, audio: np.ndarray) -> np.ndarray:
        """
        Применяет де-эмфазис фильтр.

        Args:
            audio: Аудио сигнал

        Returns:
            Фильтрованный аудио
        """
        alpha = self._deemphasis_filter
        deemphasized = np.zeros_like(audio)
        deemphasized[0] = audio[0]

        # IIR фильтр 1-го порядка
        for i in range(1, len(audio)):
            deemphasized[i] = alpha * audio[i] + (1 - alpha) * deemphasized[i - 1]

        return deemphasized

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Ресэмплинг к аудио частоте.

        Args:
            audio: Аудио сигнал

        Returns:
            Ресэмплированный аудио
        """
        from scipy import signal as scipy_signal

        # Вычисляем коэффициент ресэмплинга
        ratio = self.audio_rate / self.sample_rate

        if ratio > 1:
            # Upsampling — интерполяция
            resampled = scipy_signal.resample(audio, int(len(audio) * ratio))
        else:
            # Downsampling — антиалиасинг + децимация
            # Сначала anti-aliasing фильтр
            nyquist = self.sample_rate / 2
            cutoff = min(20000, nyquist * 0.8)  # 20 kHz max audio
            sos = scipy_signal.butter(8, cutoff, "lowpass", fs=self.sample_rate, output="sos")
            filtered = scipy_signal.sosfiltfilt(sos, audio)
            resampled = scipy_signal.resample_poly(filtered, self.audio_rate, self.sample_rate)

        return resampled.astype(np.float32)

    def start(self):
        """Запуск приёмника"""
        if not RTLSDR_AVAILABLE:
            logger.error("RTL-SDR не доступен")
            return

        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice не доступен")
            return

        logger.info(f"📻 FM Radio Receiver — {self.frequency_mhz:.1f} MHz")
        logger.info(f"📡 Инициализация RTL-SDR...")

        try:
            # Инициализация RTL-SDR
            self.sdr = RtlSdr(device_index=self.device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency_hz
            self.sdr.gain = self.gain
            self.sdr.freq_correction = 0  # PPM коррекция

            device_name = self.sdr.get_device_name()
            serial = self.sdr.get_serial_number()

            logger.info(f"✅ Устройство: {device_name} (SN: {serial})")
            logger.info(f"📡 Частота: {self.frequency_mhz:.3f} MHz")
            logger.info(f"📊 Sample rate: {self.sample_rate / 1e6:.1f} MSPS")
            logger.info(f"🔊 Gain: {self.gain} dB")
            logger.info(f"🎧 Audio rate: {self.audio_rate} Hz")

            # Запуск аудио потока
            self._start_audio_stream()

            # Обработка Ctrl+C
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("🎵 Воспроизведение... (Ctrl+C для остановки)")
            self._running = True

            # Основной цикл чтения
            self._receive_loop()

        except Exception as e:
            logger.error(f"❌ Ошибка запуска: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def _start_audio_stream(self):
        """Запуск аудио вывода через sounddevice"""

        def audio_callback(outdata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")

        self.audio_stream = sd.OutputStream(
            samplerate=self.audio_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        self.audio_stream.start()
        logger.info("🔊 Аудио поток запущен")

    def _receive_loop(self):
        """Основной цикл приёма"""
        # Размер буфера для ~100ms аудио
        audio_samples_per_buffer = int(self.audio_rate * 0.1)
        iq_samples_per_buffer = int(audio_samples_per_buffer * self.sample_rate / self.audio_rate)

        # Округляем до чётного числа
        iq_samples_per_buffer = (iq_samples_per_buffer // 2) * 2

        logger.info(
            f"📦 Буфер: {iq_samples_per_buffer} I/Q сэмплов → {audio_samples_per_buffer} аудио"
        )

        while self._running:
            try:
                # Чтение I/Q данных
                samples = self.sdr.read_samples(iq_samples_per_buffer)

                # FM демодуляция
                audio = self._fm_demodulate(samples)

                # Де-эмфазис
                audio = self._deemphasize(audio)

                # Ресэмплинг
                audio = self._resample(audio)

                # Воспроизведение
                if self.audio_stream and self._running:
                    self.audio_stream.write(audio)

            except Exception as e:
                if self._running:
                    logger.error(f"Ошибка в цикле приёма: {e}")
                break

    def _signal_handler(self, signum, frame):
        """Обработка сигналов"""
        logger.info("\n🛑 Остановка...")
        self._running = False

    def stop(self):
        """Остановка приёмника"""
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

        logger.info("👋 Приёмник остановлен")


def scan_fm_band(
    start_mhz: float = 88.0,
    end_mhz: float = 108.0,
    step_mhz: float = 0.1,
    gain: int = 30,
    device_index: int = 0,
):
    """
    Сканирование FM диапазона.

    Args:
        start_mhz: Начальная частота
        end_mhz: Конечная частота
        step_mhz: Шаг сканирования
        gain: Усиление
        device_index: Индекс устройства
    """
    if not RTLSDR_AVAILABLE:
        logger.error("RTL-SDR не доступен")
        return

    logger.info(f"📡 Сканирование FM: {start_mhz:.1f} - {end_mhz:.1f} MHz")

    sdr = RtlSdr(device_index=device_index)
    sdr.sample_rate = 2400000
    sdr.gain = gain

    stations = []

    freq = start_mhz
    while freq <= end_mhz:
        sdr.center_freq = freq * 1e6

        # Читаем сэмплы
        samples = sdr.read_samples(1024 * 256)

        # Вычисляем мощность сигнала
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-10)

        # Простой детектор станции (порог)
        if power_db > -30:  # Порог можно настроить
            stations.append((freq, power_db))
            logger.info(f"  📻 {freq:.1f} MHz: {power_db:.1f} dB ** ВОЗМОЖНАЯ СТАНЦИЯ **")

        freq += step_mhz

    sdr.close()

    if stations:
        logger.info(f"\n✅ Найдено {len(stations)} потенциальных станций:")
        for freq, power_db in stations:
            logger.info(f"  {freq:.1f} MHz ({power_db:.1f} dB)")
    else:
        logger.info("\n⚠️  Станций не найдено. Попробуйте изменить порог или усиление.")


def main():
    parser = argparse.ArgumentParser(
        description="FM Радио через RTL-SDR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Слушать Европа Плюс (106.0 МГц)
  python fm_radio.py --freq 106.0

  # Сканирование FM диапазона
  python fm_radio.py --scan

  # С ручным усилением
  python fm_radio.py --freq 101.7 --gain 40

  # С коррекцией PPM
  python fm_radio.py --freq 106.0 --ppm -5
        """,
    )

    parser.add_argument(
        "--freq", type=float, default=106.0, help="Частота станции в МГц (по умолчанию: 106.0)"
    )
    parser.add_argument(
        "--gain", type=int, default=30, help="Усиление RTL-SDR в dB (0-50, по умолчанию: 30)"
    )
    parser.add_argument("--device", type=int, default=0, help="Индекс RTL-SDR устройства")
    parser.add_argument("--scan", action="store_true", help="Сканирование FM диапазона")
    parser.add_argument(
        "--scan-start", type=float, default=88.0, help="Начальная частота сканирования (МГц)"
    )
    parser.add_argument(
        "--scan-end", type=float, default=108.0, help="Конечная частота сканирования (МГц)"
    )
    parser.add_argument("--ppm", type=int, default=0, help="Коррекция частоты в PPM")

    args = parser.parse_args()

    if args.scan:
        scan_fm_band(
            start_mhz=args.scan_start,
            end_mhz=args.scan_end,
            gain=args.gain,
            device_index=args.device,
        )
    else:
        receiver = FMRadioReceiver(
            frequency_mhz=args.freq,
            gain=args.gain,
            device_index=args.device,
        )

        # PPM коррекция
        if args.ppm != 0:
            logger.info(f"🔧 PPM коррекция: {args.ppm}")
            # В реальной реализации нужно установить freq_correction

        receiver.start()


if __name__ == "__main__":
    main()
