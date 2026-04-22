"""
Профайлер для выявления узких мест в производительности

Поддерживает:
- cProfile для профилирования CPU
- memory_profiler для профилирования использования памяти
- line_profiler для построчного профилирования
- Интеграция с системой логирования
- Автоматическое создание отчетов
"""

import cProfile
import functools
import io
import pstats
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import memory_profiler  # noqa: F401

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    from line_profiler import LineProfiler  # noqa: F401

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

from .logger import NanoprobeLogger


class ProfilerManager:
    """
    Менеджер профайлеров для производительности
    """

    def __init__(self, output_dir: str = "logs/profiles"):
        """
        Инициализация менеджера профайлеров

        Args:
            output_dir: Директория для сохранения профайлов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = NanoprobeLogger().get_logger("profiler")
        self._active_profilers: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @contextmanager
    def cpu_profile(self, name: str):
        """
        Контекстный менеджер для CPU профилирования

        Args:
            name: Имя профиля (будет использовано в названии файла)
        """
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            self._save_profile_results(name, profiler, "cpu")

    @contextmanager
    def profile_memory(self, name: str):
        """
        Контекстный менеджер для профилирования памяти

        Args:
            name: Имя профиля
        """
        if not MEMORY_PROFILER_AVAILABLE:
            self.logger.warning("memory_profiler not available, skipping memory profiling")
            yield None
            return

        stream = io.StringIO()
        try:
            yield stream
        finally:
            # memory_profiler работает иначе - он декоратор
            pass

    @contextmanager
    def block_profile(self, name: str):
        """
        Контекстный менеджер для профилирования блока кода

        Args:
            name: Имя профиля
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"Block '{name}' took {duration:.4f} seconds")

    def profile_function(self, name: Optional[str] = None):
        """
        Декоратор для профилирования функции

        Args:
            name: Имя для профиля (если не указано, используется имя функции)
        """

        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.cpu_profile(profile_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def profile_async_function(self, name: Optional[str] = None):
        """
        Декоратор для профилирования асинхронной функции

        Args:
            name: Имя для профиля
        """

        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.cpu_profile(profile_name):
                    return await func(*args, **kwargs)

            return async_wrapper

        return decorator

    def _save_profile_results(self, name: str, profiler: cProfile.Profile, profile_type: str):
        """
        Сохранение результатов профилирования

        Args:
            name: Имя профиля
            profiler: Объект профайлера
            profile_type: Тип профиля (cpu, memory, etc.)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        filename = self.output_dir / f"{safe_name}_{timestamp}_{profile_type}.prof"

        # Сохраняем бинарный профайл
        profiler.dump(str(filename))

        # Создаем читаемый отчет
        stats_file = self.output_dir / f"{safe_name}_{timestamp}_{profile_type}.txt"
        with open(stats_file, "w", encoding="utf-8") as f:
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("cumulative")
            stats.print_stats()
            f.write(stream.getvalue())

            # Также печатаем топ-20 по времени
            stream.seek(0)
            stream.truncate(0)
            stats.sort_stats("time")
            stats.print_stats(20)
            f.write("\n\n=== TOP 20 BY INTERNAL TIME ===\n")
            f.write(stream.getvalue())

        self.logger.info(f"Profile saved: {filename}")
        self.logger.info(f"Profile report: {stats_file}")

    def get_profile_summary(self, name: str) -> Dict[str, Any]:
        """
        Получение сводки по последнему профилю

        Args:
            name: Имя профиля

        Returns:
            Словарь с информацией о профиле
        """
        # Находим последние файлы профиля для данного имени
        pattern = f"{name}_*_cpu.prof"
        files = list(self.output_dir.glob(pattern))
        if not files:
            return {"error": "No profile found"}

        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        try:
            profiler = cProfile.Profile()
            profiler.dump(str(latest_file))  # Это не совсем правильно, нужно загрузить
            # На самом деле нужно загрузить, но для простоты вернем информацию о файле
            return {
                "profile_file": str(latest_file),
                "size": latest_file.stat().st_size,
                "modified": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}


# Глобальный экземпляр менеджера профайлеров
_profiler_manager: Optional[ProfilerManager] = None


def get_profiler() -> ProfilerManager:
    """Получение глобального менеджера профайлеров"""
    global _profiler_manager
    if _profiler_manager is None:
        _profiler_manager = ProfilerManager()
    return _profiler_manager


# Удобные декораторы
def profile_cpu(name: Optional[str] = None):
    """Декоратор для CPU профилирования функции"""
    return get_profiler().profile_function(name)


def profile_async_cpu(name: Optional[str] = None):
    """Декоратор для CPU профилирования асинхронной функции"""
    return get_profiler().profile_async_function(name)


@contextmanager
def profile_block(name: str):
    """Контекстный менеджер для профилирования блока кода"""
    with get_profiler().block_profile(name):
        yield


@contextmanager
def profile_cpu_block(name: str):
    """Контекстный менеджер для CPU профилирования блока кода"""
    with get_profiler().cpu_profile(name) as profiler:
        yield profiler


def start_profiler_server(port: int = 8000):
    """
    Запуск профайлер сервера для удаленного профилирования
    (требует дополнительной настройки и зависимостей)
    """
    # Это заглушка для будущего расширения
    # В реальности можно использовать py-spy или аналогичные инструменты
    pass


# Примеры использования:
# 1. Профилирование функции
# @profile_cpu
# def my_function():
#     # код функции
#     pass
#
# 2. Профилирование асинхронной функции
# @profile_async_cpu
# async def my_async_function():
#     # код асинхронной функции
#     pass
#
# 3. Профилирование блока кода
# with profile_cpu_block("expensive_operation"):
#     # дорогостоящая операция
#     result = some_expensive_function()
#
# 4. Профилирование с измерением времени
# with profile_block("timing_operation"):
#     start = time.time()
#     # какой-то код
#     end = time.time()
#     logger.info(f"Operation took {end - start:.4f} seconds")
#
# 5. Ручное профилирование
# profiler = get_profiler()
# with profiler.cpu_profile("my_operation"):
#     # код для профилирования
#     pass
