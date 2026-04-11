#!/usr/bin/env python3
"""
Тесты для новых улучшений Nanoprobe Sim Lab
Проверка улучшенных компонентов: schemas, dashboard routes, database cache, enhanced monitor
"""

import os
import sys
import time
from pathlib import Path

import pytest

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSchemasValidation:
    """Тесты валидации Pydantic схем"""

    def test_login_request_valid(self):
        """Тест валидного запроса логина"""
        from api.schemas import LoginRequest

        request = LoginRequest(username="test_user", password="Password123!")
        assert request.username == "test_user"
        assert request.password == "Password123!"

    def test_login_request_password_too_short(self):
        """Тест слишком короткого пароля"""
        from api.schemas import LoginRequest

        with pytest.raises(ValueError) as exc_info:
            LoginRequest(username="test_user", password="short1")
        assert "8" in str(exc_info.value)

    def test_login_request_password_no_uppercase(self):
        """Тест пароля без заглавных букв"""
        from api.schemas import LoginRequest

        with pytest.raises(ValueError) as exc_info:
            LoginRequest(username="test_user", password="password123")
        assert "заглавную" in str(exc_info.value)

    def test_login_request_password_no_digit(self):
        """Тест пароля без цифр"""
        from api.schemas import LoginRequest

        with pytest.raises(ValueError) as exc_info:
            LoginRequest(username="test_user", password="Passwordabc")
        assert "цифру" in str(exc_info.value)

    def test_login_request_password_no_lowercase(self):
        """Тест пароля без строчных букв"""
        from api.schemas import LoginRequest

        with pytest.raises(ValueError) as exc_info:
            LoginRequest(username="test_user", password="PASSWORD123")
        assert "строчную" in str(exc_info.value)

    def test_login_request_password_too_long(self):
        """Тест слишком длинного пароля"""
        from api.schemas import LoginRequest

        long_password = "A" * 129
        with pytest.raises(ValueError) as exc_info:
            LoginRequest(username="test_user", password=long_password)
        assert "128" in str(exc_info.value)

    def test_pagination_params(self):
        """Тест параметров пагинации"""
        from api.schemas import PaginationParams

        params = PaginationParams(page=2, page_size=50)
        assert params.offset == 50
        assert params.limit == 50

    def test_pagination_params_invalid(self):
        """Тест невалидных параметров пагинации"""
        from pydantic import ValidationError

        from api.schemas import PaginationParams

        with pytest.raises(ValidationError):
            PaginationParams(page=0)  # page < 1

        with pytest.raises(ValidationError):
            PaginationParams(page_size=101)  # page_size > 100


class TestDatabaseCache:
    """Тесты кэширования в DatabaseManager"""

    @pytest.fixture
    def db(self):
        """Создание тестовой БД"""
        import tempfile

        from utils.database import DatabaseManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = DatabaseManager(db_path=db_path, enable_cache=True)
        yield db

        # Очистка - закрываем пул перед удалением файла
        try:
            db.close_pool()
            time.sleep(0.2)
            os.unlink(db_path)
        except Exception:
            pass

    def test_cache_initialization(self, db):
        """Тест инициализации кэша"""
        assert hasattr(db, "_query_cache")
        assert hasattr(db, "_cache_ttl")
        assert db._cache_ttl == 60
        assert db._cache_max_size == 100

    def test_cache_set_get(self, db):
        """Тест сохранения и получения из кэша"""
        db._set_cache("test_key", {"data": "value"})
        result = db._get_from_cache("test_key")
        assert result == {"data": "value"}

    def test_cache_ttl(self, db):
        """Тест времени жизни кэша"""
        db._cache_ttl = 1  # 1 секунда
        db._set_cache("test_key", {"data": "value"})

        time.sleep(1.5)  # Ждём истечения TTL
        result = db._get_from_cache("test_key")
        assert result is None  # Кэш истёк

    def test_cache_invalidate(self, db):
        """Тест инвалидации кэша"""
        db._set_cache("scans:all:100:0", {"data": "value1"})
        db._set_cache("scans:spm:50:0", {"data": "value2"})
        db._set_cache("other:key", {"data": "value3"})

        db.invalidate_cache("scans:")
        assert "scans:all:100:0" not in db._query_cache
        assert "scans:spm:50:0" not in db._query_cache
        assert "other:key" in db._query_cache

    def test_cache_stats(self, db):
        """Тест статистики кэша"""
        db._set_cache("key1", "value1")
        db._set_cache("key2", "value2")

        stats = db.get_cache_stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 60

    def test_scan_methods_use_cache(self, db):
        """Тест использования кэша в методах сканирований"""
        # Добавляем тестовое сканирование
        _ = db.add_scan_result(scan_type="spm", surface_type="test", width=100, height=100)

        # Первый запрос - кэш должен заполниться
        scans = db.get_scan_results(limit=10)
        assert len(scans) >= 1

        # Проверяем, что кэш заполнен
        cache_stats = db.get_cache_stats()
        assert cache_stats["valid_entries"] > 0

        # Второй запрос - должен вернуться из кэша
        scans_cached = db.get_scan_results(limit=10)
        assert scans_cached == scans

    def test_delete_scan_invalidates_cache(self, db):
        """Тест инвалидации кэша при удалении"""
        scan_id = db.add_scan_result(scan_type="spm", surface_type="test", width=100, height=100)

        # Заполняем кэш
        db.get_scan_by_id(scan_id)
        db.get_scan_results()

        # Удаляем сканирование
        db.delete_scan(scan_id)

        # Кэш должен быть очищен
        cache_stats = db.get_cache_stats()
        assert cache_stats["valid_entries"] == 0


class TestEnhancedMonitorPrometheus:
    """Тесты экспорта Prometheus метрик"""

    @pytest.fixture
    def monitor(self):
        """Создание монитора"""
        from utils.enhanced_monitor import EnhancedSystemMonitor

        monitor = EnhancedSystemMonitor(history_size=10)
        monitor.start_monitoring()
        time.sleep(2)  # Ждём сбора метрик
        yield monitor
        monitor.stop_monitoring()

    def test_prometheus_export_format(self, monitor):
        """Тест формата экспорта Prometheus"""
        metrics = monitor.export_prometheus_metrics()

        assert isinstance(metrics, str)
        assert "# HELP" in metrics
        assert "# TYPE" in metrics
        assert "nanoprobe_cpu_percent" in metrics
        assert "nanoprobe_memory_percent" in metrics
        assert "nanoprobe_disk_percent" in metrics

    def test_prometheus_metrics_types(self, monitor):
        """Тест типов метрик Prometheus"""
        metrics = monitor.export_prometheus_metrics()

        # Проверяем наличие gauge и counter
        assert "TYPE nanoprobe_cpu_percent gauge" in metrics
        assert "TYPE nanoprobe_uptime_seconds counter" in metrics
        assert "TYPE nanoprobe_samples_count counter" in metrics

    def test_prometheus_metrics_values(self, monitor):
        """Тест значений метрик Prometheus"""
        metrics = monitor.export_prometheus_metrics()
        lines = metrics.split("\n")

        # Находим строку с CPU
        cpu_line = None
        for line in lines:
            if "nanoprobe_cpu_percent " in line and not line.startswith("#"):
                cpu_value = float(line.split()[-1])
                cpu_line = line
                break

        assert cpu_line is not None, "Не найдена строка с CPU метрикой"
        cpu_value = float(cpu_line.split()[-1])
        assert 0 <= cpu_value <= 100

        # Находим строку с memory
        mem_line = None
        for line in lines:
            if "nanoprobe_memory_percent " in line and not line.startswith("#"):
                mem_value = float(line.split()[-1])
                mem_line = line
                break

        assert mem_line is not None, "Не найдена строка с memory метрикой"
        mem_value = float(mem_line.split()[-1])
        assert 0 <= mem_value <= 100


class TestDashboardRoutes:
    """Тесты dashboard API routes"""

    def test_storage_stats(self):
        """Тест статистики хранилища"""
        from api.routes.dashboard import get_storage_stats

        stats = get_storage_stats()
        assert "used_mb" in stats
        assert "total_mb" in stats
        assert "percent" in stats
        assert isinstance(stats["used_mb"], float)
        assert isinstance(stats["total_mb"], float)

    def test_format_uptime(self):
        """Тест форматирования аптайма"""
        from utils.enhanced_monitor import format_uptime

        assert format_uptime(0) == "< 1 мин"
        assert format_uptime(60) == "1 мин"
        assert format_uptime(3661) == "1 ч 1 мин"
        assert "дн" in format_uptime(86400)


class TestDatabaseCacheIntegration:
    """Интеграционные тесты кэширования"""

    @pytest.fixture
    def db(self):
        """Создание тестовой БД"""
        import tempfile

        from utils.database import DatabaseManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = DatabaseManager(db_path=db_path, enable_cache=True)
        yield db

        # Очистка - закрываем пул перед удалением файла
        try:
            db.close_pool()
            time.sleep(0.2)
            os.unlink(db_path)
        except Exception:
            pass

    def test_batch_insert_cache_invalidation(self, db):
        """Тест инвалидации кэша при пакетной вставке"""
        # Добавляем несколько сканирований
        scans_data = [
            {"scan_type": "spm", "surface_type": "test1", "width": 100, "height": 100},
            {"scan_type": "image", "surface_type": "test2", "width": 200, "height": 200},
            {"scan_type": "sstv", "surface_type": "test3", "width": 300, "height": 300},
        ]

        # Пакетная вставка
        db.add_scan_result_batch(scans_data)

        # Проверяем, что кэш очищен
        # После вставки кэш должен быть очищен для scans:
        scan_keys = [k for k in db._query_cache.keys() if "scans:" in k]
        assert len(scan_keys) == 0

    def test_count_scans_cache(self, db):
        """Тест кэширования count_scans"""
        # Добавляем сканирования
        db.add_scan_result("spm", "test", 100, 100)
        db.add_scan_result("spm", "test", 100, 100)
        db.add_scan_result("image", "test", 200, 200)

        # Первый запрос
        count_all = db.count_scans()
        _ = db.count_scans("spm")

        # Проверяем кэш
        cache_stats = db.get_cache_stats()
        assert cache_stats["valid_entries"] >= 2

        # Второй запрос - из кэша
        count_all_cached = db.count_scans()
        assert count_all_cached == count_all


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
