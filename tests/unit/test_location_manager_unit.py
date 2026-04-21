"""Unit-тесты для модуля менеджера местоположения."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.location_manager import (
    CACHE_FILE,
    CACHE_TTL_HOURS,
    DEFAULT_LAT,
    DEFAULT_LON,
    MSK_TZ,
    TZInfo,
    force_detect_and_save,
    get_location,
    get_location_info,
    load_location_cache,
    now_msk,
    now_utc,
    refresh_msk_data,
    save_location_cache,
    utc_to_msk,
)


class TestTZInfo(unittest.TestCase):
    """Тесты для класса TZInfo"""

    def test_init(self):
        """Тест инициализации TZInfo"""
        tz = TZInfo("MSK", 3)
        self.assertEqual(tz.name, "MSK")
        self.assertEqual(tz.utc_offset, 3)

    def test_to_local(self):
        """Тест конвертации UTC в локальное время"""
        tz = TZInfo("MSK", 3)
        dt_utc = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt_local = tz.to_local(dt_utc)
        self.assertEqual(dt_local, datetime(2023, 1, 1, 15, 0, 0))

    def test_to_local_naive(self):
        """Тест конвертации наивного UTC времени (предполагается, что оно UTC)"""
        tz = TZInfo("MSK", 3)
        dt_utc = datetime(2023, 1, 1, 12, 0, 0)  # naive
        dt_local = tz.to_local(dt_utc)
        self.assertEqual(dt_local, datetime(2023, 1, 1, 15, 0, 0))

    def test_now_local(self):
        """Тест получения текущего локального времени"""
        tz = TZInfo("MSK", 3)
        # Мы не можем точно проверить время, но можем проверить, что оно возвращает datetime
        now_local = tz.now_local()
        self.assertIsInstance(now_local, datetime)

    def test_repr(self):
        """Тест строкового представления"""
        tz = TZInfo("MSK", 3)
        self.assertEqual(repr(tz), "TZInfo(MSK, UTC+3)")


class TestLocationManager(unittest.TestCase):
    """Тесты для функций менеджера местоположения"""

    def setUp(self):
        """Подготовка тестового окружения"""
        # Создаем временную директорию для кэша
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "location_cache.json"
        # Подменяем глобальный CACHE_FILE на временный
        self.original_cache_file = CACHE_FILE
        # Мы не можем легко подменять глобальную константу, поэтому будем использовать временную директорию
        # и устанавливать переменную окружения для кэша? Вместо этого, мы будем мокировать путь к кэшу в функциях.
        # Для простоты, мы будем использовать временную директорию и подменять CACHE_FILE в модуле.
        # Но так как мы импортировали константу, мы не можем легко её изменить.
        # Вместо этого, мы будем тестировать функции, которые используют кэш, и мокировать путь к кэшу.
        # Однако, в данном модуле кэш используется через глобальную переменную CACHE_FILE.
        # Мы будем использовать patch для подмены CACHE_FILE в модуле utils.location_manager.
        self.cache_patch = patch("utils.location_manager.CACHE_FILE", self.cache_file)
        self.cache_patch.start()

        # Также очистим переменные окружения, если они были установлены
        self.env_backup = dict(os.environ)
        if "GROUND_STATION_LAT" in os.environ:
            del os.environ["GROUND_STATION_LAT"]
        if "GROUND_STATION_LON" in os.environ:
            del os.environ["GROUND_STATION_LON"]
        if "GROUND_STATION_CITY" in os.environ:
            del os.environ["GROUND_STATION_CITY"]
        if "GROUND_STATION_COUNTRY" in os.environ:
            del os.environ["GROUND_STATION_COUNTRY"]

    def tearDown(self):
        """Очистка после тестов"""
        self.cache_patch.stop()
        os.environ.clear()
        os.environ.update(self.env_backup)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_now_msk(self):
        """Тест получения текущего времени в МСК"""
        msk_time = now_msk()
        self.assertIsInstance(msk_time, datetime)
        # Проверяем, что время соответствует МСК (UTC+3)
        utc_time = datetime.now(timezone.utc)
        expected_msk = utc_time + timedelta(hours=3)
        # Допускаем погрешность в несколько секунд
        self.assertAlmostEqual(msk_time.timestamp(), expected_msk.timestamp(), delta=5)

    def test_utc_to_msk(self):
        """Тест конвертации UTC времени в МСК"""
        dt_utc = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msk_time = utc_to_msk(dt_utc)
        self.assertEqual(msk_time, datetime(2023, 1, 1, 15, 0, 0))

    def test_now_utc(self):
        """Тест получения текущего времени UTC (наивное)"""
        utc_time = now_utc()
        self.assertIsInstance(utc_time, datetime)
        # Проверяем, что время наивное (без timezone)
        self.assertIsNone(utc_time.tzinfo)
        # Проверяем, что оно близко к текущему UTC времени
        now_utc_aware = datetime.now(timezone.utc).replace(tzinfo=None)
        self.assertAlmostEqual(utc_time.timestamp(), now_utc_aware.timestamp(), delta=5)

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.load_location_cache")
    def test_get_location_from_cache(self, mock_load_cache, mock_detect_ip):
        """Тест получения местоположения из кэша"""
        # Настраиваем мок для кэша
        cached_location = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Москва",
            "country": "Россия",
            "timezone": MSK_TZ,
            "timestamp": (
                datetime.now(timezone.utc) - timedelta(hours=12)
            ).isoformat(),  # Не устарел
        }
        mock_load_cache.return_value = cached_location
        mock_detect_ip.return_value = None  # Не должно быть вызвано

        location = get_location(force_detect=False, use_env=False, auto_refresh=False)
        self.assertEqual(location["lat"], 55.75)
        self.assertEqual(location["lon"], 37.61)
        self.assertEqual(location["city"], "Москва")
        self.assertEqual(location["country"], "Россия")
        self.assertEqual(location["timezone"], MSK_TZ)
        mock_detect_ip.assert_not_called()  # Убеждаемся, что IP не использовался

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.load_location_cache")
    def test_get_location_from_ip(self, mock_load_cache, mock_detect_ip):
        """Тест получения местоположения по IP"""
        # Настраиваем мок для кэша (пустой или устаревший)
        mock_load_cache.return_value = None  # Нет кэша
        # Настраиваем мок для IP
        ip_location = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Москва",
            "country": "Россия",
            "timezone": MSK_TZ,
        }
        mock_detect_ip.return_value = ip_location

        location = get_location(force_detect=False, use_env=False, auto_refresh=False)
        self.assertEqual(location["lat"], 55.75)
        self.assertEqual(location["lon"], 37.61)
        self.assertEqual(location["city"], "Москва")
        self.assertEqual(location["country"], "Россия")
        self.assertEqual(location["timezone"], MSK_TZ)
        mock_detect_ip.assert_called_once()

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.load_location_cache")
    def test_get_location_from_env(self, mock_load_cache, mock_detect_ip):
        """Тест получения местоположения из переменных окружения"""
        os.environ["GROUND_STATION_LAT"] = "55.75"
        os.environ["GROUND_STATION_LON"] = "37.61"
        os.environ["GROUND_STATION_CITY"] = "Тестовый город"
        os.environ["GROUND_STATION_COUNTRY"] = "Тестовая страна"

        # Кэш и IP не должны использоваться
        mock_load_cache.return_value = None
        mock_detect_ip.return_value = None

        location = get_location(force_detect=False, use_env=True, auto_refresh=False)
        self.assertEqual(location["lat"], 55.75)
        self.assertEqual(location["lon"], 37.61)
        self.assertEqual(location["city"], "Тестовый город")
        self.assertEqual(location["country"], "Тестовая страна")
        self.assertEqual(location["timezone"], MSK_TZ)
        mock_load_cache.assert_not_called()
        mock_detect_ip.assert_not_called()

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.save_location_cache")
    def test_force_detect_and_save(self, mock_save_cache, mock_detect_ip):
        """Тест принудительного определения и сохранения местоположения"""
        ip_location = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Москва",
            "country": "Россия",
            "timezone": MSK_TZ,
        }
        mock_detect_ip.return_value = ip_location

        location = force_detect_and_save()
        self.assertEqual(location["lat"], 55.75)
        self.assertEqual(location["lon"], 37.61)
        self.assertEqual(location["city"], "Москва")
        self.assertEqual(location["country"], "Россия")
        self.assertEqual(location["timezone"], MSK_TZ)
        mock_detect_ip.assert_called_once()
        mock_save_cache.assert_called_once_with(ip_location)

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.save_location_cache")
    def test_force_detect_and_save_failure(self, mock_save_cache, mock_detect_ip):
        """Тест неудачного принудительного определения местоположения"""
        mock_detect_ip.return_value = None  # Не удалось определить

        location = force_detect_and_save()
        self.assertIsNone(location)
        mock_detect_ip.assert_called_once()
        mock_save_cache.assert_not_called()

    def test_get_location_info(self):
        """Тест получения читаемой строки с информацией о местоположении"""
        # Мокируем get_location, чтобы вернуть предсказуемое значение
        with patch("utils.location_manager.get_location") as mock_get_location:
            mock_get_location.return_value = {
                "lat": 55.7558,
                "lon": 37.6173,
                "city": "Москва",
                "country": "Россия",
                "timezone": MSK_TZ,
            }
            info = get_location_info()
            self.assertIn("Москва, Россия", info)
            self.assertIn("Coords: 55.7558N, 37.6173E", info)
            self.assertIn("Timezone: MSK (UTC+3)", info)

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.save_location_cache")
    def test_refresh_msk_data(self, mock_save_cache, mock_detect_ip):
        """Тест принудительного обновления данных МСК"""
        location = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Москва",
            "country": "Россия",
            "timezone": MSK_TZ,
        }
        mock_detect_ip.return_value = location

        result = refresh_msk_data()
        self.assertEqual(result, location)
        mock_detect_ip.assert_called_once()
        mock_save_cache.assert_called_once_with(location)

    @patch("utils.location_manager.detect_location_by_ip")
    @patch("utils.location_manager.save_location_cache")
    def test_refresh_msk_data_failure(self, mock_save_cache, mock_detect_ip):
        """Тест неудачного принудительного обновления данных МСК"""
        mock_detect_ip.return_value = None

        result = refresh_msk_data()
        self.assertIsNone(result)
        mock_detect_ip.assert_called_once()
        mock_save_cache.assert_not_called()

    def test_load_location_cache(self):
        """Тест загрузки кэша местоположения"""
        # Создаем файл кэша
        cache_data = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Тест",
            "country": "Тестовая страна",
            "timezone_name": "MSK",
            "timezone_offset": 3,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        loaded = load_location_cache()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["lat"], 55.75)
        self.assertEqual(loaded["lon"], 37.61)
        self.assertEqual(loaded["city"], "Тест")
        self.assertEqual(loaded["country"], "Тестовая страна")

    def test_load_location_cache_expired(self):
        """Тест загрузки просроченного кэша"""
        # Создаем файл кэша с устаревшим timestamp
        cache_data = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Тест",
            "country": "Тестовая страна",
            "timezone_name": "MSK",
            "timezone_offset": 3,
            "timestamp": (
                datetime.now(timezone.utc) - timedelta(hours=25)
            ).isoformat(),  # Устарел на 25 часов
        }
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        loaded = load_location_cache()
        self.assertIsNone(loaded)  # Должен вернуть None из-за устаревания

    def test_save_location_cache(self):
        """Тест сохранения кэша местоположения"""
        location = {
            "lat": 55.75,
            "lon": 37.61,
            "city": "Тест",
            "country": "Тестовая страна",
            "timezone": MSK_TZ,
        }
        save_location_cache(location)
        self.assertTrue(self.cache_file.exists())
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        self.assertEqual(cached["lat"], 55.75)
        self.assertEqual(cached["lon"], 37.61)
        self.assertEqual(cached["city"], "Тест")
        self.assertEqual(cached["country"], "Тестовая страна")
        self.assertEqual(cached["timezone_name"], "MSK")
        self.assertEqual(cached["timezone_offset"], 3)
        self.assertIn("timestamp", cached)


if __name__ == "__main__":
    unittest.main()
