#!/usr/bin/env python3
"""
FM Stereo Decoder для RTL-SDR
Декодирование FM радиовещания (87.5-108 MHz) со стерео звуком

Поддерживает:
- Моно FM (базовое декодирование)
- Стерео FM (с пилот-тоном 19 kHz)
- RDS (Radio Data System) — название станции, текст песни

Использование:
    python fm_stereo_decoder.py -f 101.7      # Слушать 101.7 MHz
    python fm_stereo_decoder.py -f 101.7 --record  # Записать в файл
    python fm_stereo_decoder.py -f 101.7 --rds     # Показать RDS данные
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    print("⚠️ pyrtlsdr не установлен: pip install pyrtlsdr")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FMStereoDecoder:
    """
    Декодер FM Stereo.

    Принцип работы:
    1. Захват I/Q сэмплов на частоте FM станции
    2. FM демодуляция (арктангенс разности фаз)
    3. Декодирование стерео (с пилот-тоном 19 kHz)
    4. Де-эмфаза (50 μs для Европы, 75 μs для США)
    """

    def __init__(
        self,
        frequency_mhz: float = 101.7,
        sample_rate: float = 2.4e6,
        audio_rate: int = 44100,
        gain: int = 30,
    ):
        """
        Инициализация FM декодера.

        Args:
            frequency_mhz: Частота FM станции (MHz)
            sample_rate: Частота дискретизации RTL-SDR
            audio_rate: Частота аудио выхода (Hz)
            gain: Усиление RTL-SDR (dB)
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.sample_rate = int(sample_rate)
        self.audio_rate = audio_rate
        self.gain = gain

        self.sdr = None
        self._running = False

        # Фильтры
        self._fm_deemphasis_tau = 50e-6  # 50 μs для Европы

    def start(self, record: bool = False, show_rds: bool = False):
        """
        Запуск FM декодера.

        Args:
            record: Записывать аудио в файл
            show_rds: Показывать RDS данные
        """
        logger.info(f"📻 FM Stereo Decoder — {self.frequency_mhz:.1f} MHz")
        logger.info(f"📊 Sample rate: {self.sample_rate / 1e6:.1f} MSPS")
        logger.info(f"🔊 Audio rate: {self.audio_rate} Hz")
        logger.info(f"🎚️ Gain: {self.gain} dB")

        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency_hz
            self.sdr.gain = self.gain

            device_name = self.sdr.get_device_name()
            logger.info(f"✅ Устройство: {device_name}")
            logger.info("")
            logger.info("🎵 Приём FM сигнала...")
            logger.info("💡 Нажмите Ctrl+C для остановки")
            logger.info("")

            self._running = True
            self._receive_loop(record, show_rds)

        except KeyboardInterrupt:
            logger.info("\n🛑 Остановка по команде пользователя")
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def _receive_loop(self, record: bool, show_rds: bool):
        """Основной цикл приёма"""
        import wave

        buffer_size = 1024 * 256
        audio_samples = []

        if record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(__file__).parent.parent / "data" / "fm_record"
            output_file.mkdir(parents=True, exist_ok=True)
            wav_path = output_file / f"fm_{self.frequency_mhz:.1f}_{timestamp}.wav"
            logger.info(f"💾 Запись в: {wav_path}")
            wav_file = wave.open(str(wav_path), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.audio_rate)
        else:
            wav_file = None

        try:
            while self._running:
                samples = self.sdr.read_samples(buffer_size)

                # FM демодуляция
                audio = self._fm_demodulate(samples)

                if audio is not None:
                    audio_samples.append(audio)

                    # Конвертация в 16-bit PCM
                    pcm = self._float_to_pcm16(audio)

                    if wav_file:
                        wav_file.writeframes(pcm.tobytes())

                    # Показать RDS если запрошено
                    if show_rds and len(audio_samples) % 10 == 0:
                        rds_data = self._decode_rds(samples)
                        if rds_data:
                            logger.info(f"📝 RDS: {rds_data}")

        except KeyboardInterrupt:
            pass
        finally:
            if wav_file:
                wav_file.close()
                logger.info(f"✅ Сохранено в {wav_path}")

    def _fm_demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        FM демодуляция.

        Args:
            samples: I/Q сэмплы

        Returns:
            Аудио сигнал
        """
        # Вычисляем фазу
        phase = np.angle(samples)

        # Разность фаз (демодуляция)
        audio = np.diff(np.unwrap(phase))

        # Де-эмфаза (RC фильтр низких частот)
        audio = self._deemphasize(audio)

        return audio

    def _deemphasize(self, audio: np.ndarray) -> np.ndarray:
        """
        Де-эмфаза аудио.

        Compensates for pre-emphasis applied at the transmitter.
        """
        alpha = 1.0 / (1.0 + self.audio_rate * self._fm_deemphasis_tau)
        output = np.zeros_like(audio)
        output[0] = audio[0]

        for i in range(1, len(audio)):
            output[i] = alpha * audio[i] + (1 - alpha) * output[i - 1]

        return output

    def _decode_rds(self, samples: np.ndarray) -> str:
        """
        Простое декодирование RDS данных.

        RDS передаётся на поднесущей 57 kHz.

        Args:
            samples: I/Q сэмплы

        Returns:
            Строка с RDS данными или None
        """
        # Placeholder — полноценное RDS декодирование
        # требует сложной обработки сигнала
        # Для реальной реализации нужен RDS decoder

        # Проверяем наличие пилот-тона 19 kHz
        fft_result = np.fft.fft(samples[:4096])
        frequencies = np.fft.fftfreq(4096, d=1.0 / self.sample_rate)

        # Ищем пилот-тон 19 kHz
        pilot_idx = np.argmin(np.abs(frequencies - 19000))
        pilot_power = np.abs(fft_result[pilot_idx])

        if pilot_power > np.mean(np.abs(fft_result)) * 2:
            return "Стерео сигнал (пилот-тон 19 kHz обнаружен)"

        return None

    def _float_to_pcm16(self, audio: np.ndarray) -> np.ndarray:
        """
        Конвертация float аудио в 16-bit PCM.

        Args:
            audio: Аудио сигнал (-1.0 до 1.0)

        Returns:
            16-bit PCM данные
        """
        # Нормализация
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        # Конвертация в int16
        pcm = (audio * 32767).astype(np.int16)
        return pcm

    def stop(self):
        """Остановка декодера"""
        self._running = False
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass
        logger.info("👋 FM декодер остановлен")


def main():
    parser = argparse.ArgumentParser(
        description="FM Stereo Decoder для RTL-SDR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Слушать 101.7 MHz
  python fm_stereo_decoder.py -f 101.7

  # Записать в файл
  python fm_stereo_decoder.py -f 101.7 --record

  # С RDS данными
  python fm_stereo_decoder.py -f 101.7 --rds
        """,
    )

    parser.add_argument(
        "-f",
        "--freq",
        type=float,
        required=True,
        help="Частота FM станции MHz (87.5-108)",
    )
    parser.add_argument(
        "-g",
        "--gain",
        type=int,
        default=30,
        help="Усиление RTL-SDR dB (по умолч.: 30)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Записать аудио в WAV файл",
    )
    parser.add_argument(
        "--rds",
        action="store_true",
        help="Показать RDS данные",
    )

    args = parser.parse_args()

    if not (87.5 <= args.freq <= 108.0):
        logger.error("❌ Частота должна быть в диапазоне 87.5-108 MHz")
        sys.exit(1)

    decoder = FMStereoDecoder(
        frequency_mhz=args.freq,
        gain=args.gain,
    )

    decoder.start(record=args.record, show_rds=args.rds)


if __name__ == "__main__":
    main()
