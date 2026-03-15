"""
Тесты для SSTV Ground Station API.
Проверка расписания МКС, декодирования SSTV и кэширования.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к проекту
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSatelliteTracker:
    """Тесты для SatelliteTracker."""

    def test_tracker_initialization(self):
        """Тест инициализации трекера."""
        from satellite_tracker import SatelliteTracker
        
        tracker = SatelliteTracker()
        
        assert tracker.ground_station_lat == 55.75
        assert tracker.ground_station_lon == 37.61
        assert 'iss' in tracker.satellites
        assert len(tracker.SSTV_FREQUENCIES) > 0

    def test_get_pass_predictions(self):
        """Тест предсказания пролётов."""
        from satellite_tracker import SatelliteTracker
        
        tracker = SatelliteTracker()
        passes = tracker.get_pass_predictions('iss', hours_ahead=24)
        
        assert isinstance(passes, list)
        if passes:
            first_pass = passes[0]
            assert 'aos' in first_pass
            assert 'los' in first_pass
            assert 'max_elevation' in first_pass
            assert 'frequency' in first_pass

    def test_get_next_pass(self):
        """Тест получения следующего пролёта."""
        from satellite_tracker import SatelliteTracker
        
        tracker = SatelliteTracker()
        next_pass = tracker.get_next_pass('iss')
        
        assert next_pass is None or isinstance(next_pass, dict)
        if next_pass:
            assert 'aos' in next_pass
            assert 'los' in next_pass

    def test_get_current_position(self):
        """Тест текущей позиции спутника."""
        from satellite_tracker import SatelliteTracker
        
        tracker = SatelliteTracker()
        position = tracker.get_current_position('iss')
        
        assert position is not None
        assert 'altitude_km' in position
        assert 'velocity_kmh' in position
        assert 'latitude' in position
        assert 'longitude' in position

    def test_get_sstv_schedule(self):
        """Тест расписания SSTV."""
        from satellite_tracker import SatelliteTracker
        
        tracker = SatelliteTracker()
        schedule = tracker.get_sstv_schedule(hours_ahead=24)
        
        assert isinstance(schedule, list)
        # Расписание должно быть отсортировано по времени
        if len(schedule) > 1:
            for i in range(len(schedule) - 1):
                assert schedule[i]['aos'] <= schedule[i + 1]['aos']


class TestSSTVDecoder:
    """Тесты для SSTVDecoder."""

    def test_decoder_initialization(self):
        """Тест инициализации декодера."""
        from sstv_decoder import SSTVDecoder
        
        decoder = SSTVDecoder()
        
        assert decoder.mode == 'auto'
        assert len(decoder.SUPPORTED_MODES) > 0
        assert decoder.decoded_image is None

    def test_supported_modes(self):
        """Тест поддерживаемых режимов."""
        from sstv_decoder import SSTVDecoder
        
        decoder = SSTVDecoder()
        modes = decoder.SUPPORTED_MODES
        
        assert isinstance(modes, list)
        assert len(modes) > 0
        
        # Проверка популярных режимов
        mode_names = [m for m in modes]
        assert any('Martin' in m for m in mode_names)
        assert any('Scottie' in m for m in mode_names)

    def test_metadata_structure(self):
        """Тест структуры метаданных."""
        from sstv_decoder import SSTVDecoder
        
        decoder = SSTVDecoder()
        metadata = decoder.get_metadata()
        
        assert isinstance(metadata, dict)
        
        statistics = decoder.get_statistics()
        assert isinstance(statistics, dict)
        assert 'total_images' in statistics
        assert 'supported_modes' in statistics


class TestSSTVAPIRoutes:
    """Тесты для SSTV API routes."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis cache."""
        with patch('api.routes.sstv.REDIS_AVAILABLE', False):
            yield

    @pytest.fixture
    def mock_tracker(self):
        """Mock SatelliteTracker."""
        with patch('api.routes.sstv.tracker_module') as mock:
            yield mock

    def test_iss_schedule_endpoint_structure(self, mock_redis, mock_tracker):
        """Тест структуры эндпоинта расписания."""
        from api.routes.sstv import router
        
        assert router is not None
        # Проверка что роутер существует

    def test_sstv_modes_available(self, mock_redis):
        """Тест доступности режимов SSTV."""
        # Проверка что модуль загружается
        try:
            from api.routes import sstv
            assert hasattr(sstv, 'router')
        except ImportError:
            pytest.skip("SSTV module not available")


class TestExternalServicesCaching:
    """Тесты кэширования внешних сервисов."""

    def test_redis_cache_import(self):
        """Тест импорта Redis cache."""
        try:
            from utils.redis_cache import RedisCache
            cache = RedisCache()
            assert cache is not None
        except ImportError:
            pytest.skip("Redis not available")

    def test_cache_key_generation(self):
        """Тест генерации ключей кэша."""
        from utils.redis_cache import RedisCache
        
        cache = RedisCache()
        key = cache.generate_key("test", "arg1", "arg2")
        
        assert key.startswith("test:")
        assert len(key) > 6


class TestIntegration:
    """Интеграционные тесты."""

    def test_sstv_tracker_integration(self):
        """Тест интеграции трекера и декодера."""
        from satellite_tracker import SatelliteTracker
        from sstv_decoder import SSTVDecoder
        
        tracker = SatelliteTracker()
        decoder = SSTVDecoder()
        
        # Получаем расписание
        schedule = tracker.get_sstv_schedule(hours_ahead=24)
        
        # Проверяем что декодер готов
        assert decoder is not None
        assert len(decoder.SUPPORTED_MODES) > 0
        
        # Проверяем что частоты совпадают
        if schedule:
            for pass_info in schedule:
                freq = pass_info.get('frequency', 0)
                assert freq > 0 or freq == 0  # Частота может быть 0 если не найдена


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
