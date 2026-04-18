"""
Автоматический захват сигналов спутников.

Планировщик предсказывает пролёты спутников и автоматически включает запись.

Поддерживаемые спутники:
- NOAA-15, NOAA-18, NOAA-19: APT режим (137 MHz)
- METEOR-M2: LRPT режим (137.9 MHz)

Требуемые зависимости:
    pip install skyfield requests
"""

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SatellitePass:
    """Информация о пролёте спутника."""

    satellite: str
    aos: datetime  # Acquisition of Signal (начало пролёта)
    los: datetime  # Loss of Signal (конец пролёта)
    max_elevation: float  # Максимальная элевация в градусах
    frequency_mhz: float  # Частота в МГц
    mode: str = "APT"  # Режим передачи (APT или LRPT)
    azimuth_aos: float = 0.0  # Азимут в начале пролёта
    azimuth_los: float = 0.0  # Азимут в конце пролёта
    predicted: bool = True  # Предсказанный или фактический пролёт

    def duration_seconds(self) -> int:
        """Длительность пролёта в секундах."""
        return int((self.los - self.aos).total_seconds())

    def time_to_aos(self) -> float:
        """Время до начала пролёта в секундах."""
        return (self.aos - datetime.now()).total_seconds()

    def is_active(self) -> bool:
        """Проверка: активен ли пролёт сейчас."""
        now = datetime.now()
        return self.aos <= now <= self.los

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "satellite": self.satellite,
            "aos": self.aos.isoformat(),
            "los": self.los.isoformat(),
            "max_elevation": self.max_elevation,
            "frequency_mhz": self.frequency_mhz,
            "mode": self.mode,
            "duration_seconds": self.duration_seconds(),
            "azimuth_aos": self.azimuth_aos,
            "azimuth_los": self.azimuth_los,
            "predicted": self.predicted,
        }


@dataclass
class TLEData:
    """TLE данные спутника."""

    name: str
    line1: str
    line2: str
    updated: datetime = field(default_factory=datetime.now)


class SatelliteAutoCapture:
    """
    Автоматический захват спутниковых сигналов.

    Основные возможности:
    - Предсказание пролётов спутников на основе TLE данных
    - Автоматический запуск записи перед началом пролёта
    - Поддержка NOAA (APT) и METEOR (LRPT) спутников
    - Расширяемая система обработчиков событий
    """

    # Конфигурация поддерживаемых спутников
    SATELLITES = {
        "NOAA-15": {"freq": 137.620, "mode": "APT", "tle_name": "NOAA 15"},
        "NOAA-18": {"freq": 137.9125, "mode": "APT", "tle_name": "NOAA 18"},
        "NOAA-19": {"freq": 137.100, "mode": "APT", "tle_name": "NOAA 19"},
        "METEOR-M2": {"freq": 137.900, "mode": "LRPT", "tle_name": "METEOR M 2"},
        "METEOR-M2-2": {"freq": 137.850, "mode": "LRPT", "tle_name": "METEOR M 2-2"},
        "METEOR-M2-3": {"freq": 137.700, "mode": "LRPT", "tle_name": "METEOR M 2-3"},
    }

    # Минимальная элевация для начала записи (градусы)
    MIN_ELEVATION = 5.0

    # Время предварительного запуска записи перед AOS (секунды)
    PRE_RECORD_OFFSET = 120

    # Время продолжения записи после LOS (секунды)
    POST_RECORD_OFFSET = 60

    # Интервал обновления TLE данных (часы)
    TLE_REFRESH_HOURS = 6

    # Интервал проверки планировщика (секунды)
    CHECK_INTERVAL = 30

    def __init__(
        self,
        location_lat: float = 55.75,
        location_lon: float = 37.61,
        output_dir: str = "data/satellite_captures",
        device_index: int = 0,
        sample_rate: int = 2400000,
        tle_refresh_hours: int = 6,
    ):
        """
        Инициализация автозахвата спутников.

        Args:
            location_lat: Широта местоположения (градусы)
            location_lon: Долгота местоположения (градусы)
            output_dir: Директория для сохранения записей
            device_index: Индекс RTL-SDR устройства
            sample_rate: Частота дискретизации
            tle_refresh_hours: Интервал обновления TLE данных
        """
        self.location = (location_lat, location_lon)
        self.output_dir = Path(output_dir)
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.tle_refresh_hours = tle_refresh_hours

        self.passes: List[SatellitePass] = []
        self.tle_data: Dict[str, TLEData] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._capture_handlers: List[Callable[[SatellitePass], None]] = []
        self._pass_start_handlers: List[Callable[[SatellitePass], None]] = []
        self._pass_end_handlers: List[Callable[[SatellitePass], None]] = []

        # Текущий активный пролёт
        self._active_pass: Optional[SatellitePass] = None

        # Создаём директорию для записей
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Инициализация автозахвата спутников: " "location=(%.2f, %.2f), output=%s",
            location_lat,
            location_lon,
            self.output_dir,
        )

    def add_capture_handler(self, handler: Callable[[SatellitePass], None]) -> None:
        """
        Добавить обработчик события захвата.

        Вызывается при начале захвата спутника.

        Args:
            handler: Функция обратного вызова, принимающая SatellitePass
        """
        self._capture_handlers.append(handler)
        logger.debug("Добавлен обработчик захвата: %s", handler.__name__)

    def add_pass_start_handler(self, handler: Callable[[SatellitePass], None]) -> None:
        """
        Добавить обработчик начала пролёта.

        Args:
            handler: Функция обратного вызова
        """
        self._pass_start_handlers.append(handler)

    def add_pass_end_handler(self, handler: Callable[[SatellitePass], None]) -> None:
        """
        Добавить обработчик окончания пролёта.

        Args:
            handler: Функция обратного вызова
        """
        self._pass_end_handlers.append(handler)

    def predict_passes(self, hours_ahead: int = 48) -> List[SatellitePass]:
        """
        Предсказание пролётов спутников на ближайшие N часов.

        Args:
            hours_ahead: Количество часов вперёд для предсказания

        Returns:
            Список предсказанных пролётов, отсортированный по времени AOS
        """
        try:
            from skyfield.api import EarthSatellite, Topos, load
        except ImportError:
            logger.error("Модуль skyfield не найден. Установите: pip install skyfield")
            return []

        # Загружаем TLE данные
        if not self._refresh_tle_data():
            logger.warning("Не удалось обновить TLE данные, используем кэш")

        ts = load.timescale()
        station = Topos(self.location[0], self.location[1])

        passes = []

        for sat_name, sat_config in self.SATELLITES.items():
            tle_data = self.tle_data.get(sat_name)
            if not tle_data:
                logger.warning("TLE данные для %s не найдены", sat_name)
                continue

            try:
                satellite = EarthSatellite(tle_data.line1, tle_data.line2, sat_name, ts)

                # Время начала и конца предсказания
                t0 = ts.now()
                t1 = ts.utc(datetime.now() + timedelta(hours=hours_ahead))

                # Ищем пролёты с минимальной элевацией MIN_ELEVATION
                times, events = satellite.find_events(
                    station, t0, t1, altitude_degrees=self.MIN_ELEVATION
                )

                # Группируем события в пролёты
                # events: 0=aos, 1=max elevation, 2=los
                i = 0
                while i < len(events):
                    if i + 2 < len(events):
                        aos_time = times[i]
                        max_el_time = times[i + 1]
                        los_time = times[i + 2]

                        aos = aos_time.utc_datetime()
                        los = los_time.utc_datetime()

                        # Вычисляем максимальную элевацию
                        max_el = self._calc_max_elevation(
                            satellite, station, aos_time, max_el_time, los_time
                        )

                        # Вычисляем азимуты
                        az_aos, az_los = self._calc_azimuths(satellite, station, aos_time, los_time)

                        pass_duration = (los - aos).total_seconds()

                        # Пропускаем слишком короткие пролёты
                        if pass_duration < 300:  # < 5 минут
                            i += 3
                            continue

                        passes.append(
                            SatellitePass(
                                satellite=sat_name,
                                aos=aos,
                                los=los,
                                max_elevation=max_el,
                                frequency_mhz=sat_config["freq"],
                                mode=sat_config["mode"],
                                azimuth_aos=az_aos,
                                azimuth_los=az_los,
                                predicted=True,
                            )
                        )

                        i += 3
                    else:
                        break

            except Exception as e:
                logger.error("Ошибка предсказания для %s: %s", sat_name, e)

        # Сортируем по времени AOS
        passes.sort(key=lambda p: p.aos)
        self.passes = passes

        logger.info("Предсказано %d пролётов на ближайшие %d часов", len(passes), hours_ahead)

        # Логируем ближайшие пролёты
        for pass_ in passes[:3]:
            time_to_aos = pass_.time_to_aos()
            if time_to_aos > 0:
                logger.info(
                    "Ближайший пролёт: %s через %.1f мин, " "макс. элевация: %.1f°",
                    pass_.satellite,
                    time_to_aos / 60,
                    pass_.max_elevation,
                )

        return passes

    def start_scheduler(self) -> None:
        """Запуск планировщика в отдельном потоке."""
        if self._running:
            logger.warning("Планировщик уже запущен")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="SatelliteScheduler"
        )
        self._thread.start()
        logger.info("Планировщик автозахвата запущен")

    def stop_scheduler(self) -> None:
        """Остановка планировщика."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Планировщик автозахвата остановлен")

    def _scheduler_loop(self) -> None:
        """Основной цикл планировщика."""
        logger.info("Запуск цикла планировщика")

        while self._running:
            try:
                now = datetime.now()

                # Обновляем предсказания каждые tle_refresh_hours
                if not self.passes or (
                    self.passes and (now - self.passes[0].aos).total_seconds() > 3600
                ):
                    self.predict_passes(hours_ahead=24)

                # Проверяем ближайшие пролёты
                self._check_upcoming_passes()

                # Проверяем активный пролёт
                self._check_active_pass()

                # Удаляем устаревшие пролёты
                self.passes = [p for p in self.passes if p.los > now - timedelta(minutes=10)]

            except Exception as e:
                logger.error("Ошибка в цикле планировщика: %s", e)

            # Ждём до следующей проверки
            time.sleep(self.CHECK_INTERVAL)

    def _check_upcoming_passes(self) -> None:
        """Проверка предстоящих пролётов."""
        _ = datetime.now()  # noqa: F841

        for pass_ in self.passes:
            time_to_aos = pass_.time_to_aos()

            # За PRE_RECORD_OFFSET минут до пролёта начинаем подготовку
            if 0 < time_to_aos < self.PRE_RECORD_OFFSET:
                logger.info(
                    "Приближается пролёт %s через %.1f мин, " "элевация: %.1f°",
                    pass_.satellite,
                    time_to_aos / 60,
                    pass_.max_elevation,
                )

                # Ждём начала пролёта
                time.sleep(max(time_to_aos, 1))

                # Выполняем захват
                self._execute_capture(pass_)
                break

    def _check_active_pass(self) -> None:
        """Проверка активного пролёта."""
        if not self._active_pass:
            return

        now = datetime.now()

        # Проверяем закончился ли пролёт
        if now > self._active_pass.los + timedelta(seconds=self.POST_RECORD_OFFSET):
            logger.info(
                "Завершение захвата %s (длительность: %ds)",
                self._active_pass.satellite,
                self._active_pass.duration_seconds(),
            )

            # Вызываем обработчики окончания
            for handler in self._pass_end_handlers:
                try:
                    handler(self._active_pass)
                except Exception as e:
                    logger.error("Ошибка обработчика окончания: %s", e)

            self._active_pass = None

    def _execute_capture(self, pass_: SatellitePass) -> None:
        """
        Выполнение захвата спутника.

        Args:
            pass_: Информация о пролёте
        """
        self._active_pass = pass_

        # Вычисляем длительность записи
        duration = pass_.duration_seconds() + self.PRE_RECORD_OFFSET + self.POST_RECORD_OFFSET

        # Формируем имя файла
        timestamp = pass_.aos.strftime("%Y%m%d_%H%M%S")
        filename = f"{pass_.satellite}_{timestamp}_" f"{pass_.mode}_{pass_.frequency_mhz:.3f}MHz"
        output_path = self.output_dir / filename

        logger.info(
            "Начало захвата %s на частоте %.3f MHz, " "длительность: %ds, файл: %s",
            pass_.satellite,
            pass_.frequency_mhz,
            duration,
            output_path,
        )

        # Вызываем обработчики начала
        for handler in self._pass_start_handlers:
            try:
                handler(pass_)
            except Exception as e:
                logger.error("Ошибка обработчика начала: %s", e)

        # Вызываем общие обработчики захвата
        for handler in self._capture_handlers:
            try:
                handler(pass_)
            except Exception as e:
                logger.error("Ошибка обработчика захвата: %s", e)

        # Запускаем запись
        try:
            freq_hz = int(pass_.frequency_mhz * 1e6)

            cmd = [
                "rtl_sdr",
                "-f",
                str(freq_hz),
                "-s",
                str(self.sample_rate),
                "-g",
                "48",  # Усиление в дБ * 10
                "-d",
                str(self.device_index),
                str(output_path.with_suffix(".raw")),
            ]

            # Запускаем запись с таймаутом
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Ждём окончания записи
            start_time = time.time()
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > duration:
                    # Таймаут, принудительно завершаем
                    process.terminate()
                    break
                time.sleep(1)

                # Логируем прогресс
                remaining = duration - elapsed
                if int(remaining) % 60 == 0 and remaining > 0:
                    logger.info("Запись %s: осталось %.0f сек", pass_.satellite, remaining)

            # Получаем результат
            stdout, stderr = process.communicate()

            if process.returncode == 0 or os.path.exists(str(output_path.with_suffix(".raw"))):
                logger.info(
                    "Запись %s завершена успешно: %s",
                    pass_.satellite,
                    output_path.with_suffix(".raw"),
                )
            else:
                logger.error("Ошибка записи %s: %s", pass_.satellite, stderr.decode())

        except FileNotFoundError:
            logger.error("rtl_sdr не найден. Установите rtl-sdr-tools.")
        except Exception as e:
            logger.error("Ошибка захвата %s: %s", pass_.satellite, e)

    def _refresh_tle_data(self) -> bool:
        """
        Обновление TLE данных спутников.

        Returns:
            True если обновление успешно
        """
        try:
            import requests

            # URL для загрузки TLE данных (CelesTrak)
            tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle"

            logger.info("Загрузка TLE данных с Celestrak...")

            response = requests.get(tle_url, timeout=30)
            response.raise_for_status()

            # Парсим TLE данные
            lines = response.text.strip().split("\n")

            current_name = None
            line1 = None

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # Первая строка - название спутника
                if not line.startswith("1 ") and not line.startswith("2 "):
                    current_name = line
                    continue

                # Первая строка TLE
                if line.startswith("1 "):
                    line1 = line
                # Вторая строка TLE
                elif line.startswith("2 ") and current_name and line1:
                    # Находим соответствующий спутник
                    for sat_name, sat_config in self.SATELLITES.items():
                        if sat_config["tle_name"] in current_name:
                            self.tle_data[sat_name] = TLEData(
                                name=current_name, line1=line1, line2=line, updated=datetime.now()
                            )
                            logger.debug("Обновлены TLE данные для %s", sat_name)
                            break

                    current_name = None
                    line1 = None

            logger.info("Обновлены TLE данные для %d спутников", len(self.tle_data))

            return len(self.tle_data) > 0

        except ImportError:
            logger.error("Модуль requests не найден")
            return False
        except Exception as e:
            logger.error("Ошибка загрузки TLE данных: %s", e)
            return False

    def _calc_max_elevation(
        self,
        satellite: Any,
        station: Any,
        aos_time,
        max_el_time,
        los_time,
    ) -> float:
        """
        Вычисление максимальной элевации пролёта.

        Args:
            satellite: Объект спутника из skyfield
            station: Объект станции (наблюдателя)
            aos_time: Время начала пролёта
            max_el_time: Время максимальной элевации
            los_time: Время конца пролёта

        Returns:
            Максимальная элевация в градусах
        """
        try:
            # Вычисляем элевацию в момент максимальной высоты
            geocentric = satellite.at(max_el_time)
            altaz = geocentric.observe(station).apparent().altaz()
            elevation = altaz.degrees

            return elevation
        except Exception as e:
            logger.error("Ошибка вычисления элевации: %s", e)
            return 0.0

    def _calc_azimuths(
        self,
        satellite: Any,
        station: Any,
        aos_time,
        los_time,
    ) -> Tuple[float, float]:
        """
        Вычисление азимутов начала и конца пролёта.

        Args:
            satellite: Объект спутника
            station: Объект станции
            aos_time: Время начала пролёта
            los_time: Время конца пролёта

        Returns:
            Tuple (азимут AOS, азимут LOS) в градусах
        """
        try:
            # Азимут в начале пролёта
            aos_geocentric = satellite.at(aos_time)
            aos_altaz = aos_geocentric.observe(station).apparent().altaz()
            az_aos = aos_altaz.azimuth.degrees

            # Азимут в конце пролёта
            los_geocentric = satellite.at(los_time)
            los_altaz = los_geocentric.observe(station).apparent().altaz()
            az_los = los_altaz.azimuth.degrees

            return az_aos, az_los
        except Exception as e:
            logger.error("Ошибка вычисления азимутов: %s", e)
            return 0.0, 0.0

    def get_next_pass(self, satellite: Optional[str] = None) -> Optional[SatellitePass]:
        """
        Получить следующий пролёт.

        Args:
            satellite: Имя спутника (опционально, если None - любой)

        Returns:
            Следующий пролёт или None
        """
        now = datetime.now()

        future_passes = [p for p in self.passes if p.aos > now]

        if satellite:
            future_passes = [p for p in future_passes if p.satellite == satellite]

        if future_passes:
            return min(future_passes, key=lambda p: p.aos)

        return None

    def get_active_pass(self) -> Optional[SatellitePass]:
        """Получить активный пролёт."""
        return self._active_pass

    def get_passes_summary(self) -> Dict[str, Any]:
        """
        Получить сводку о пролётах.

        Returns:
            Словарь со статистикой
        """
        now = datetime.now()

        upcoming = [p for p in self.passes if p.aos > now]
        active = [p for p in self.passes if p.is_active()]

        return {
            "total_predicted": len(self.passes),
            "upcoming": len(upcoming),
            "active": len(active),
            "next_pass": (upcoming[0].to_dict() if upcoming else None),
            "active_pass": (self._active_pass.to_dict() if self._active_pass else None),
            "satellites": list(self.SATELLITES.keys()),
            "location": self.location,
        }


def main():
    """CLI интерфейс для автозахвата спутников."""
    import argparse

    parser = argparse.ArgumentParser(description="Автоматический захват спутников NOAA/METEOR")
    parser.add_argument(
        "--lat",
        type=float,
        default=55.75,
        help="Широта местоположения",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=37.61,
        help="Долгота местоположения",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/satellite_captures",
        help="Директория для записей",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Индекс RTL-SDR устройства",
    )
    parser.add_argument(
        "--predict",
        type=int,
        default=48,
        help="Количество часов для предсказания",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Запуск планировщика в фоновом режиме",
    )

    args = parser.parse_args()

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    capture = SatelliteAutoCapture(
        location_lat=args.lat,
        location_lon=args.lon,
        output_dir=args.output,
        device_index=args.device,
    )

    # Предсказываем пролёты
    print(f"Предсказание пролётов на {args.predict} часов...")
    passes = capture.predict_passes(hours_ahead=args.predict)

    if passes:
        print(f"\nНайдено {len(passes)} пролётов:")
        for p in passes[:5]:
            print(
                f"  {p.satellite}: {p.aos.strftime('%Y-%m-%d %H:%M')} - "
                f"{p.los.strftime('%H:%M')}, "
                f"макс. элевация: {p.max_elevation:.1f}°, "
                f"{p.mode}"
            )
    else:
        print("Пролёты не найдены")

    if args.daemon:
        print("\nЗапуск планировщика...")
        capture.start_scheduler()

        try:
            while True:
                time.sleep(60)
                summary = capture.get_passes_summary()
                print(
                    f"\rАктивных: {summary['active']}, " f"ожидается: {summary['upcoming']}",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("\nОстановка планировщика...")
            capture.stop_scheduler()
    else:
        print("\nИспользуйте --daemon для запуска планировщика")


if __name__ == "__main__":
    main()
