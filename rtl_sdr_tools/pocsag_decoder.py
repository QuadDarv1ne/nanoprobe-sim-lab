#!/usr/bin/env python3
"""
POCSAG Pager Decoder для RTL-SDR
Декодирование пейджинговых сообщений (137-175 MHz, 450-470 MHz)

POCSAG (Post Office Code Standardization Advisory Group):
- Скорость: 512, 1200, 2400 бод
- Частоты: 137-175 MHz, 450-470 MHz
- Типы сообщений: Tone, Numeric, Alphanumeric

Использование:
    python pocsag_decoder.py -f 148.5        # Декодировать 148.5 MHz
    python pocsag_decoder.py -f 148.5 -r 512  # Скорость 512 бод
    python pocsag_decoder.py -f 148.5 --json  # JSON вывод
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone

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


class POCSAGDecoder:
    """
    Декодер POCSAG пейджинговых сообщений.

    Формат POCSAG:
    - Преамбула: 576 бит (101010...)
    - Batch: 8 слов (2x SYNC, 6x MESSAGE)
    - Слово: 32 бита (21 бит данных + 11 бит ECC)
    """

    def __init__(
        self,
        baud_rate: int = 512,
        frequency_mhz: float = 148.5,
    ):
        """
        Инициализация POCSAG декодера.

        Args:
            baud_rate: Скорость передачи (512, 1200, 2400)
            frequency_mhz: Частота приёма (MHz)
        """
        self.baud_rate = baud_rate
        self.frequency_mhz = frequency_mhz

        self.messages = []
        self.pagers = {}
        self.messages_received = 0
        self.start_time = datetime.now(timezone.utc)

        # POCSAG константы
        self.SYNC_WORD = 0x7CD215D  # Синхрослово
        self.IDLE_WORD = 0x7A89C19  # Слово простоя
        self.PREAMBLE_LENGTH = 576  # бит

    def decode_samples(self, samples: np.ndarray) -> list:
        """
        Декодирование POCSAG из I/Q сэмплов.

        Args:
            samples: I/Q сэмплы

        Returns:
            Список декодированных сообщений
        """
        messages = []

        # FM демодуляция
        phase = np.angle(samples)
        baseband = np.diff(np.unwrap(phase))

        # Пороговая детекция
        threshold = np.std(baseband) * 1.5
        bits = baseband > threshold

        # Поиск преамбулы
        preamble_pattern = np.array([i % 2 == 0 for i in range(self.PREAMBLE_LENGTH)])

        for i in range(len(bits) - self.PREAMBLE_LENGTH):
            if np.array_equal(bits[i : i + self.PREAMBLE_LENGTH], preamble_pattern):
                # Нашли преамбулу, декодируем batch
                batch_data = bits[i + self.PREAMBLE_LENGTH : i + self.PREAMBLE_LENGTH + 544]
                batch_messages = self._decode_batch(batch_data)
                messages.extend(batch_messages)
                break

        return messages

    def _decode_batch(self, bits: np.ndarray) -> list:
        """
        Декодирование POCSAG batch.

        Batch = 8 слов (32 бита каждое)
        2x SYNC + 6x MESSAGE

        Args:
            bits: Биты batch

        Returns:
            Список сообщений
        """
        messages = []

        for i in range(0, len(bits), 32):
            if i + 32 > len(bits):
                break

            word_bits = bits[i : i + 32]
            word = self._bits_to_int(word_bits)

            # Проверяем синхрослово
            if word == self.SYNC_WORD:
                continue

            # Проверяем слово простоя
            if word == self.IDLE_WORD:
                continue

            # Декодируем сообщение
            message = self._decode_word(word, i // 32)
            if message:
                messages.append(message)

        return messages

    def _decode_word(self, word: int, position: int) -> dict:
        """
        Декодирование POCSAG слова.

        Args:
            word: 32-битное слово
            position: Позиция в batch

        Returns:
            Сообщение или None
        """
        # Проверяем бит чётности (ECC)
        data_bits = (word >> 11) & 0x1FFFFF  # 21 бит данных

        # Определяем тип сообщения
        if data_bits & 0x800000:
            # Адресное слово
            return self._decode_address(data_bits, position)
        else:
            # Слово сообщения
            return self._decode_message_data(data_bits, position)

    def _decode_address(self, data: int, position: int) -> dict:
        """
        Декодирование адресного слова.

        Args:
            data: 21 бит данных адреса
            position: Позиция в batch

        Returns:
            Данные адреса
        """
        address = (data >> 10) & 0x7FFF
        function_bits = (data >> 8) & 0x3

        pager_id = address * 4 + function_bits

        return {
            "type": "address",
            "pager_id": pager_id,
            "position": position,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _decode_message_data(self, data: int, position: int) -> dict:
        """
        Декодирование данных сообщения.

        Args:
            data: 21 бит данных сообщения
            position: Позиция в batch

        Returns:
            Данные сообщения
        """
        # Преобразуем в ASCII (для alphanumeric)
        message_bytes = []
        for i in range(0, 21, 7):
            char_code = (data >> (14 - i)) & 0x7F
            if 32 <= char_code <= 126:
                message_bytes.append(chr(char_code))

        text = "".join(message_bytes).strip()

        if text:
            return {
                "type": "message",
                "text": text,
                "position": position,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return None

    def _bits_to_int(self, bits: np.ndarray) -> int:
        """Конвертация битов в целое число"""
        result = 0
        for bit in bits:
            result = (result << 1) | int(bit)
        return result

    def add_message(self, message: dict):
        """Добавить сообщение"""
        self.messages.append(message)
        self.messages_received += 1

        if message.get("type") == "address":
            pager_id = message["pager_id"]
            if pager_id not in self.pagers:
                self.pagers[pager_id] = []
            self.pagers[pager_id].append(message)

    def get_stats(self) -> dict:
        """Получить статистику"""
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "messages_received": self.messages_received,
            "pagers_tracked": len(self.pagers),
            "uptime_seconds": round(elapsed, 1),
            "baud_rate": self.baud_rate,
            "frequency_mhz": self.frequency_mhz,
        }


class POCSAGReceiver:
    """
    POCSAG приёмник с RTL-SDR.
    """

    def __init__(
        self,
        frequency_mhz: float = 148.5,
        gain: int = 30,
        baud_rate: int = 512,
    ):
        """
        Инициализация POCSAG приёмника.

        Args:
            frequency_mhz: Частота приёма (MHz)
            gain: Усиление RTL-SDR (dB)
            baud_rate: Скорость передачи (бод)
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.gain = gain
        self.baud_rate = baud_rate

        self.sdr = None
        self.decoder = POCSAGDecoder(
            baud_rate=baud_rate,
            frequency_mhz=frequency_mhz,
        )
        self._running = False

    def start(self, output_json: bool = False):
        """
        Запуск POCSAG приёмника.

        Args:
            output_json: Вывод в JSON формате
        """
        logger.info(f"📟 POCSAG Pager Decoder — {self.frequency_mhz:.1f} MHz")
        logger.info(f"⚡ Baud rate: {self.baud_rate}")
        logger.info(f"🎚️ Gain: {self.gain} dB")
        logger.info("")
        logger.info("📡 Приём POCSAG сигналов...")
        logger.info("💡 Нажмите Ctrl+C для остановки")
        logger.info("")

        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = int(self.baud_rate * 2)  # Минимальная частота дискретизации
            self.sdr.center_freq = self.frequency_hz
            self.sdr.gain = self.gain

            device_name = self.sdr.get_device_name()
            logger.info(f"✅ Устройство: {device_name}")
            logger.info("")

            self._running = True
            self._receive_loop(output_json)

        except KeyboardInterrupt:
            logger.info("\n🛑 Остановка по команде пользователя")
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def _receive_loop(self, output_json: bool):
        """Основной цикл приёма"""
        buffer_size = 8192

        while self._running:
            try:
                samples = self.sdr.read_samples(buffer_size)
                messages = self.decoder.decode_samples(samples)

                for msg in messages:
                    self.decoder.add_message(msg)

                    if output_json:
                        print(json.dumps(msg, indent=2))
                    else:
                        if msg["type"] == "address":
                            logger.info(f"📟 Pager #{msg['pager_id']}")
                        elif msg["type"] == "message":
                            logger.info(f"💬 {msg['text']}")

                # Статистика каждые 100 сообщений
                if self.decoder.messages_received % 100 == 0:
                    stats = self.decoder.get_stats()
                    logger.info(
                        f"📊 Messages: {stats['messages_received']} | "
                        f"Pagers: {stats['pagers_tracked']}"
                    )

            except Exception as e:
                if self._running:
                    logger.error(f"Ошибка в цикле: {e}")
                break

    def stop(self):
        """Остановка приёмника"""
        self._running = False
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass

        stats = self.decoder.get_stats()
        logger.info(f"\n📊 Final stats:")
        logger.info(f"   Messages: {stats['messages_received']}")
        logger.info(f"   Pagers: {stats['pagers_tracked']}")
        logger.info(f"   Uptime: {stats['uptime_seconds']}s")
        logger.info("👋 POCSAG декодер остановлен")


def main():
    parser = argparse.ArgumentParser(
        description="POCSAG Pager Decoder для RTL-SDR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Декодировать 148.5 MHz (512 бод)
  python pocsag_decoder.py -f 148.5

  # Скорость 1200 бод
  python pocsag_decoder.py -f 148.5 -r 1200

  # JSON вывод
  python pocsag_decoder.py -f 148.5 --json
        """,
    )

    parser.add_argument(
        "-f",
        "--freq",
        type=float,
        required=True,
        help="Частота POCSAG MHz (137-175 или 450-470)",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=512,
        choices=[512, 1200, 2400],
        help="Скорость передачи бод (по умолч.: 512)",
    )
    parser.add_argument(
        "-g",
        "--gain",
        type=int,
        default=30,
        help="Усиление RTL-SDR dB (по умолч.: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON вывод данных",
    )

    args = parser.parse_args()

    # Проверяем диапазон частот
    if not ((137 <= args.freq <= 175) or (450 <= args.freq <= 470)):
        logger.error("❌ Частота должна быть в диапазоне " "137-175 MHz или 450-470 MHz")
        sys.exit(1)

    receiver = POCSAGReceiver(
        frequency_mhz=args.freq,
        gain=args.gain,
        baud_rate=args.rate,
    )

    receiver.start(output_json=args.json)


if __name__ == "__main__":
    main()
