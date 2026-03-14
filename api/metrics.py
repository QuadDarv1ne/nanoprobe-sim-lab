"""
Prometheus метрики для Nanoprobe FastAPI приложения
Мониторинг производительности, запросов, бизнес-метрик
"""

import time
import os
from functools import wraps
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List
from threading import Lock

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        multiprocess,
        start_http_server,
    )
    from prometheus_client import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None
    # Заглушки если prometheus_client не установлен
    class Counter:
        """TODO: Add description"""
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Histogram:
        """TODO: Add description"""
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Gauge:
        """TODO: Add description"""
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Summary:
        """TODO: Add description"""
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass


# ==================== Метрики ====================

# HTTP запросы
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
) if PROMETHEUS_AVAILABLE else None

# Время ответа HTTP
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
) if PROMETHEUS_AVAILABLE else None

# Активные запросы
http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    ['method', 'endpoint']
) if PROMETHEUS_AVAILABLE else None

# Размер ответа
http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
) if PROMETHEUS_AVAILABLE else None

# Ошибки приложений
app_errors_total = Counter(
    'app_errors_total',
    'Total application errors',
    ['type', 'module']
) if PROMETHEUS_AVAILABLE else None

# Бизнес-метрики: Сканирования
scans_created_total = Counter(
    'scans_created_total',
    'Total scans created',
    ['scan_type']
) if PROMETHEUS_AVAILABLE else None

scans_in_database = Gauge(
    'scans_in_database',
    'Number of scans currently in database'
) if PROMETHEUS_AVAILABLE else None

# Бизнес-метрики: Симуляции
simulations_created_total = Counter(
    'simulations_created_total',
    'Total simulations created',
    ['simulation_type', 'status']
) if PROMETHEUS_AVAILABLE else None

simulations_active = Gauge(
    'simulations_active',
    'Number of active simulations'
) if PROMETHEUS_AVAILABLE else None

# Бизнес-метрики: Анализ дефектов
defect_analyses_total = Counter(
    'defect_analyses_total',
    'Total defect analyses performed',
    ['model_name']
) if PROMETHEUS_AVAILABLE else None

defects_detected_total = Counter(
    'defects_detected_total',
    'Total defects detected',
    ['defect_type']
) if PROMETHEUS_AVAILABLE else None

# Бизнес-метрики: Сравнение поверхностей
comparisons_total = Counter(
    'comparisons_total',
    'Total surface comparisons performed'
) if PROMETHEUS_AVAILABLE else None

# Бизнес-метрики: PDF отчёты
reports_generated_total = Counter(
    'reports_generated_total',
    'Total PDF reports generated',
    ['report_type']
) if PROMETHEUS_AVAILABLE else None

# Система
system_info = Gauge(
    'system_info',
    'System information',
    ['version', 'environment']
) if PROMETHEUS_AVAILABLE else None

# WebSocket подключения
websocket_connections = Gauge(
    'websocket_connections',
    'Number of active WebSocket connections'
) if PROMETHEUS_AVAILABLE else None

# Кэш
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
) if PROMETHEUS_AVAILABLE else None

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
) if PROMETHEUS_AVAILABLE else None


# ==================== Middleware ====================

class PrometheusMiddleware:
    """
    Middleware для сбора метрик HTTP запросов
    Использование в FastAPI:
        app.add_middleware(PrometheusMiddleware)
    """

    def __init__(self, app):
        """TODO: Add description"""
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        if not PROMETHEUS_AVAILABLE:
            return await self.app(scope, receive, send)

        method = scope['method']
        path = scope['path']
        start_time = time.time()

        # Увеличиваем счётчик активных запросов
        http_requests_in_progress.labels(method=method, endpoint=path).inc()

        # Обёртка для отправки для перехвата статуса
        status_code = None
        response_size = 0

        async def send_wrapper(message):
            nonlocal status_code, response_size
            if message['type'] == 'http.response.start':
                status_code = message['status']
            elif message['type'] == 'http.response.body':
                response_size += len(message.get('body', b''))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            status_code = 500
            app_errors_total.labels(type=type(e).__name__, module='http').inc()
            raise
        finally:
            duration = time.time() - start_time

            # Уменьшаем счётчик активных запросов
            http_requests_in_progress.labels(method=method, endpoint=path).dec()

            # Записываем метрики
            if status_code:
                http_requests_total.labels(
                    method=method,
                    endpoint=path,
                    status=status_code
                ).inc()

                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)

                http_response_size_bytes.labels(
                    method=method,
                    endpoint=path
                ).observe(response_size)


# ==================== Декораторы ====================

def track_metrics(endpoint_name: str = None):
    """
    Декоратор для отслеживания метрик функции
    Использование:
        @track_metrics('analyze_defects')
        def analyze_defects(...):
            """TODO: Add description"""
            pass
    """
    def decorator(func):
        """TODO: Add description"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = endpoint_name or func.__name__
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                if PROMETHEUS_AVAILABLE:
                    app_errors_total.labels(type=type(e).__name__, module=func.__module__).inc()
                raise
            finally:
                if PROMETHEUS_AVAILABLE:
                    duration = time.time() - start_time
                    http_request_duration_seconds.labels(method='function', endpoint=name).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            """TODO: Add description"""
            name = endpoint_name or func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if PROMETHEUS_AVAILABLE:
                    app_errors_total.labels(type=type(e).__name__, module=func.__module__).inc()
                raise
            finally:
                if PROMETHEUS_AVAILABLE:
                    duration = time.time() - start_time
                    http_request_duration_seconds.labels(method='function', endpoint=name).observe(duration)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ==================== Бизнес-метрики ====================

class BusinessMetrics:
    """
    Менеджер бизнес-метрик
    """

    @staticmethod
    def inc_scan_created(scan_type: str):
        """Увеличить счётчик созданных сканирований"""
        if PROMETHEUS_AVAILABLE:
            scans_created_total.labels(scan_type=scan_type).inc()

    @staticmethod
    def set_scans_in_database(count: int):
        """Установить количество сканирований в БД"""
        if PROMETHEUS_AVAILABLE:
            scans_in_database.set(count)

    @staticmethod
    def inc_simulation_created(simulation_type: str, status: str = 'created'):
        """Увеличить счётчик созданных симуляций"""
        if PROMETHEUS_AVAILABLE:
            simulations_created_total.labels(simulation_type=simulation_type, status=status).inc()

    @staticmethod
    def set_simulations_active(count: int):
        """Установить количество активных симуляций"""
        if PROMETHEUS_AVAILABLE:
            simulations_active.set(count)

    @staticmethod
    def inc_defect_analysis(model_name: str, defects: List[Dict] = None):
        """Увеличить счётчик анализов дефектов"""
        if PROMETHEUS_AVAILABLE:
            defect_analyses_total.labels(model_name=model_name).inc()
            if defects:
                for defect in defects:
                    defect_type = defect.get('type', 'unknown')
                    defects_detected_total.labels(defect_type=defect_type).inc()

    @staticmethod
    def inc_comparison():
        """Увеличить счётчик сравнений поверхностей"""
        if PROMETHEUS_AVAILABLE:
            comparisons_total.inc()

    @staticmethod
    def inc_report_generated(report_type: str):
        """Увеличить счётчик сгенерированных отчётов"""
        if PROMETHEUS_AVAILABLE:
            reports_generated_total.labels(report_type=report_type).inc()

    @staticmethod
    def inc_cache_hit(cache_type: str):
        """Увеличить счётчик попаданий кэша"""
        if PROMETHEUS_AVAILABLE:
            cache_hits_total.labels(cache_type=cache_type).inc()

    @staticmethod
    def inc_cache_miss(cache_type: str):
        """Увеличить счётчик промахов кэша"""
        if PROMETHEUS_AVAILABLE:
            cache_misses_total.labels(cache_type=cache_type).inc()

    @staticmethod
    def set_websocket_connections(count: int):
        """Установить количество WebSocket подключений"""
        if PROMETHEUS_AVAILABLE:
            websocket_connections.set(count)


# ==================== Endpoint для метрик ====================

async def get_metrics():
    """
    Endpoint для метрик Prometheus
    Использование в FastAPI:
        from api.metrics import get_metrics
        app.add_api_route('/metrics', get_metrics)
    """
    from fastapi import Response
    from fastapi.responses import PlainTextResponse

    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse(
            "Prometheus client not installed. Install with: pip install prometheus-client",
            status_code=503
        )

    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ==================== Инициализация ====================

def setup_prometheus(port: int = 9090):
    """
    Настройка Prometheus
    Запуск сервера метрик на отдельном порту
    """
    if not PROMETHEUS_AVAILABLE:
        print("[WARN] Prometheus client not installed")
        return False

    try:
        # Установка системной информации
        version = os.getenv('APP_VERSION', '1.0.0')
        environment = os.getenv('ENVIRONMENT', 'production')
        system_info.labels(version=version, environment=environment).set(1)

        # Запуск сервера метрик
        start_http_server(port)
        print(f"[OK] Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to start Prometheus metrics server: {e}")
        return False


# ==================== Утилиты ====================

def get_metrics_as_dict() -> Dict:
    """
    Получение всех метрик в виде словаря
    Полезно для отладки и API ответов
    """
    if not PROMETHEUS_AVAILABLE:
        return {'error': 'Prometheus not available'}

    from prometheus_client import REGISTRY

    metrics = {}
    for collector in REGISTRY._names_to_collectors.values():
        try:
            for metric in collector.collect():
                for sample in metric.samples:
                    key = sample.name
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append({
                        'value': sample.value,
                        'labels': sample.labels
                    })
        except Exception:
            continue

    return metrics


# Для совместимости с asyncio
try:
    import asyncio
except ImportError:
    asyncio = None
