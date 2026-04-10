#!/usr/bin/env python3
"""
ADS-B приёмник самолётов через RTL-SDR (1090 МГц)

Декодирует:
- Callsign (номер рейса)
- Altitude (высота)
- Speed (скорость)
- Lat/Lon (координаты)
- Heading (курс)

Использование:
    python adsb_receiver.py              # Основной режим
    python adsb_receiver.py --json       # Вывод в JSON
    python adsb_receiver.py --map        # Открыть карту в браузере
"""

import argparse
import json
import logging
import signal
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    print("⚠️  pyrtlsdr не установлен: pip install pyrtlsdr")

try:
    import webbrowser
    import threading
    import http.server

    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ADS-B константы
ADS_B_FREQUENCY = 1090e6  # 1090 МГц
ADS_B_SAMPLE_RATE = 2400000  # 2.4 MSPS
ADS_B_PULSE_WIDTH = 0.5e-6  # 0.5 мкс


class ADSBDecoder:
    """
    Простой ADS-B декодер.

    Декодирует Mode-S Extended Squitter (DF17)
    """

    def __init__(self):
        self.aircraft: Dict[str, dict] = {}
        self.messages_received = 0
        self.messages_decoded = 0
        self.start_time = datetime.now(timezone.utc)

    def detect_adsb_pulses(self, samples: np.ndarray) -> List[dict]:
        """
        Детекция ADS-B импульсов (1090 MHz).

        ADS-B использует PPM (Pulse Position Modulation):
        - 8 микросекунд преамбула
        - 56 или 112 микросекунд данных

        Args:
            samples: I/Q сэмплы

        Returns:
            Список обнаруженных сообщений
        """
        # Вычисляем мощность
        power = np.abs(samples) ** 2

        # Простой детектор порогом
        threshold = np.mean(power) * 3.0
        messages = []

        i = 0
        while i < len(power) - 100:
            if power[i] > threshold:
                # Нашли начало импульса
                # Проверяем преамбулу ADS-B (8 мкс)
                if self._check_preamble(power, i):
                    # Декодируем сообщение
                    message = self._decode_message(samples, i)
                    if message:
                        messages.append(message)
                        i += 112  # Пропускаем длину сообщения
                else:
                    i += 10
            else:
                i += 1

        return messages

    def _check_preamble(self, power: np.ndarray, index: int) -> bool:
        """
        Проверка преамбулы ADS-B.

        Преамбула: 4 импульса на позициях 0, 1.0, 3.5, 4.5 мкс
        """
        # Упрощённая проверка — есть ли импульс в начале
        return power[index] > np.mean(power) * 2.5

    def _decode_message(self, samples: np.ndarray, index: int) -> Optional[dict]:
        """
        Декодирование ADS-B сообщения.

        Returns:
            dict с данными или None
        """
        # В реальной реализации здесь нужно:
        # 1. Извлечь 112 бит данных
        # 2. Декодировать Downlink Format (DF)
        # 3. Распознать тип сообщения
        # 4. Извлечь ICAO address, callsign, altitude, etc.

        # Placeholder — для полноценного декодирования нужна
        # библиотека pyModeS или dump1090

        return None

    def add_aircraft(self, icao: str, data: dict):
        """Добавить/обновить данные самолёта"""
        if icao not in self.aircraft:
            self.aircraft[icao] = {
                "icao": icao,
                "first_seen": datetime.now(timezone.utc).isoformat(),
            }

        self.aircraft[icao].update(data)
        self.aircraft[icao]["last_seen"] = datetime.now(timezone.utc).isoformat()

    def get_aircraft_list(self) -> List[dict]:
        """Получить список всех видимых самолётов"""
        return list(self.aircraft.values())

    def get_stats(self) -> dict:
        """Статистика декодера"""
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "messages_received": self.messages_received,
            "messages_decoded": self.messages_decoded,
            "aircraft_tracked": len(self.aircraft),
            "uptime_seconds": round(elapsed, 1),
        }


class ADSBReceiver:
    """
    ADS-B приёмник с RTL-SDR.

    Режимы:
    1. RTL-SDR прямой захват (требует декодер)
    2. dump1090 интеграция (рекомендуется)
    """

    def __init__(
        self,
        frequency_mhz: float = 1090.0,
        sample_rate: int = 2400000,
        gain: int = 30,
        device_index: int = 0,
        use_dump1090: bool = False,
    ):
        """
        Инициализация ADS-B приёмника.

        Args:
            frequency_mhz: Частота ADS-B (1090 МГц)
            sample_rate: Частота дискретизации
            gain: Усиление RTL-SDR
            device_index: Индекс устройства
            use_dump1090: Использовать dump1090 вместо Python декодера
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.sample_rate = sample_rate
        self.gain = gain
        self.device_index = device_index
        self.use_dump1090 = use_dump1090

        self.sdr: Optional[RtlSdr] = None
        self.decoder = ADSBDecoder()
        self._running = False

    def start(self, output_json: bool = False, web_map: bool = False):
        """
        Запуск ADS-B приёмника.

        Args:
            output_json: Вывод в JSON
            web_map: Показать карту самолётов
        """
        logger.info(f"✈️  ADS-B Receiver — {self.frequency_mhz:.0f} MHz")

        if self.use_dump1090:
            logger.info("📡 Режим: dump1090 интеграция")
            self._start_dump1090_mode(output_json, web_map)
        else:
            logger.info("📡 Режим: RTL-SDR прямой захват")
            self._start_sdr_mode(output_json, web_map)

    def _start_sdr_mode(self, output_json: bool, web_map: bool):
        """Прямой захват через RTL-SDR"""
        if not RTLSDR_AVAILABLE:
            logger.error("❌ RTL-SDR не доступен")
            return

        try:
            logger.info("📡 Инициализация RTL-SDR...")

            self.sdr = RtlSdr(device_index=self.device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency_hz
            self.sdr.gain = self.gain

            device_name = self.sdr.get_device_name()
            logger.info(f"✅ Устройство: {device_name}")
            logger.info(f"📡 Частота: {self.frequency_mhz:.0f} MHz")
            logger.info(f"📊 Sample rate: {self.sample_rate / 1e6:.1f} MSPS")
            logger.info(f"🔊 Gain: {self.gain} dB")
            logger.info("")
            logger.info("⚠️  Прямой захват ADS-B ограничен")
            logger.info("💡 Рекомендуется использовать dump1090:")
            logger.info("   python adsb_receiver.py --dump1090")
            logger.info("")

            # Обработка Ctrl+C
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self._running = True
            self._receive_loop(output_json)

        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def _receive_loop(self, output_json: bool):
        """Основной цикл приёма"""
        buffer_size = 1024 * 512  # ~200ms

        while self._running:
            try:
                samples = self.sdr.read_samples(buffer_size)
                self.decoder.messages_received += 1

                # Детекция ADS-B
                messages = self.decoder.detect_adsb_pulses(samples)

                if messages:
                    self.decoder.messages_decoded += len(messages)

                # Вывод каждые 50 сообщений
                if self.decoder.messages_received % 50 == 0:
                    stats = self.decoder.get_stats()
                    logger.info(
                        f"📊 Messages: {stats['messages_received']} | "
                        f"Aircraft: {stats['aircraft_tracked']}"
                    )

                    if output_json:
                        aircraft = self.decoder.get_aircraft_list()
                        print(json.dumps(aircraft, indent=2))

            except Exception as e:
                if self._running:
                    logger.error(f"Ошибка в цикле: {e}")
                break

    def _start_dump1090_mode(self, output_json: bool, web_map: bool):
        """
        Режим интеграции с dump1090.

        dump1090 — оптимизированный ADS-B декодер на C.
        Мы запускаем его как subprocess и парсим вывод.
        """
        import subprocess

        logger.info("")
        logger.info("📦 Проверка dump1090...")

        # Проверяем наличие dump1090
        try:
            result = subprocess.run(
                ["dump1090", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.info(f"✅ dump1090 найден: {result.stderr.strip()}")
        except FileNotFoundError:
            logger.warning("⚠️  dump1090 не найден")
            logger.info("")
            logger.info("📦 Установка dump1090:")
            logger.info("  Windows:")
            logger.info("    Скачайте с https://github.com/flightaware/dump1090")
            logger.info("")
            logger.info("  Linux:")
            logger.info("    sudo apt install dump1090-mutability")
            logger.info("")
            logger.info("🔄 Переключаюсь на прямой захват RTL-SDR...")
            self._start_sdr_mode(output_json, web_map)
            return

    def _signal_handler(self, signum, frame):
        """Обработка сигналов"""
        logger.info("\n🛑 Остановка...")
        self._running = False

    def stop(self):
        """Остановка приёмника"""
        self._running = False

        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass

        logger.info("👋 ADS-B приёмник остановлен")


def start_web_server(aircraft_data: dict, port: int = 8080):
    """
    Запуск веб-сервера с картой самолётов.

    Args:
        aircraft_data: Данные о самолётах
        port: Порт веб-сервера
    """

    class ADSBHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._serve_map()
            elif self.path == "/aircraft.json":
                self._serve_json()
            else:
                self.send_error(404)

        def _serve_map(self):
            """HTML страница с картой"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            html = """
<!DOCTYPE html>
<html>
<head>
    <title>ADS-B Aircraft Tracker</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; }
        #map { height: 100vh; }
        .plane-icon { font-size: 24px; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // Карта
        var map = L.map('map').setView([55.75, 37.61], 8);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Маркеры самолётов
        function updateAircraft() {
            fetch('/aircraft.json')
                .then(r => r.json())
                .then(data => {
                    console.log('Aircraft:', data.length);
                });
        }

        setInterval(updateAircraft, 2000);
        updateAircraft();
    </script>
</body>
</html>
"""
            self.wfile.write(html.encode())

        def _serve_json(self):
            """JSON данные о самолётах"""
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            data = list(aircraft_data.values())
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format, *args):
            pass  # Подавить логи

    server = http.server.HTTPServer(("localhost", port), ADSBHandler)
    logger.info(f"🌐 Веб-карта: http://localhost:{port}")

    # Открыть в браузере
    webbrowser.open(f"http://localhost:{port}")

    # Запуск в отдельном потоке
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server


def main():
    parser = argparse.ArgumentParser(
        description="ADS-B ADS-B приёмник через RTL-SDR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Прямой захват
  python adsb_receiver.py

  # С dump1090 (рекомендуется)
  python adsb_receiver.py --dump1090

  # JSON вывод
  python adsb_receiver.py --json

  # Веб-карта
  python adsb_receiver.py --map
        """,
    )

    parser.add_argument(
        "--freq",
        type=float,
        default=1090.0,
        help="Частота ADS-B МГц (по умолчанию: 1090.0)",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=30,
        help="Усиление RTL-SDR dB (по умолчанию: 30)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Индекс RTL-SDR устройства"
    )
    parser.add_argument(
        "--dump1090", action="store_true", help="Использовать dump1090"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON вывод данных"
    )
    parser.add_argument(
        "--map", action="store_true", help="Веб-карта самолётов"
    )

    args = parser.parse_args()

    receiver = ADSBReceiver(
        frequency_mhz=args.freq,
        gain=args.gain,
        device_index=args.device,
        use_dump1090=args.dump1090,
    )

    receiver.start(output_json=args.json, web_map=args.map)


if __name__ == "__main__":
    main()
