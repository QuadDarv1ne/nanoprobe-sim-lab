"""
Performance Monitoring Dashboard

Мониторинг производительности Nanoprobe Sim Lab в реальном времени.

Features:
- Prometheus метрики
- System metrics (CPU, RAM, Disk, Network)
- API performance metrics
- Custom business metrics (SSTV, Scans, Simulations)
- Real-time dashboard
"""

import psutil
import time
from datetime import datetime
from typing import Dict, Optional
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
    start_http_server,
)
import logging

logger = logging.getLogger(__name__)


# ==================== Prometheus Metrics ====================

# System metrics
CPU_PERCENT = Gauge(
    'nanoprobe_cpu_percent',
    'CPU usage percentage',
    ['core']
)

MEMORY_PERCENT = Gauge(
    'nanoprobe_memory_percent',
    'Memory usage percentage'
)

MEMORY_AVAILABLE = Gauge(
    'nanoprobe_memory_available_bytes',
    'Available memory in bytes'
)

DISK_PERCENT = Gauge(
    'nanoprobe_disk_percent',
    'Disk usage percentage',
    ['mountpoint']
)

DISK_FREE = Gauge(
    'nanoprobe_disk_free_bytes',
    'Free disk space in bytes',
    ['mountpoint']
)

NETWORK_BYTES_SENT = Counter(
    'nanoprobe_network_sent_bytes_total',
    'Total network bytes sent'
)

NETWORK_BYTES_RECV = Counter(
    'nanoprobe_network_recv_bytes_total',
    'Total network bytes received'
)

# API metrics
API_REQUESTS_TOTAL = Counter(
    'nanoprobe_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

API_REQUEST_LATENCY = Histogram(
    'nanoprobe_api_request_latency_seconds',
    'API request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

API_REQUESTS_IN_PROGRESS = Gauge(
    'nanoprobe_api_requests_in_progress',
    'Number of API requests currently being processed'
)

# Business metrics
SSTV_RECORDINGS_TOTAL = Counter(
    'nanoprobe_sstv_recordings_total',
    'Total SSTV recordings'
)

SSTV_DECODED_IMAGES = Counter(
    'nanoprobe_sstv_decoded_images_total',
    'Total SSTV images decoded successfully'
)

SCANS_TOTAL = Gauge(
    'nanoprobe_scans_total',
    'Total number of scans in database'
)

SIMULATIONS_ACTIVE = Gauge(
    'nanoprobe_simulations_active',
    'Number of active simulations'
)

UPLOAD_QUEUE_SIZE = Gauge(
    'nanoprobe_upload_queue_size',
    'Number of items in upload queue'
)

# Uptime
UPTIME_SECONDS = Counter(
    'nanoprobe_uptime_seconds_total',
    'Total uptime in seconds'
)


class PerformanceMonitor:
    """
    Менеджер мониторинга производительности
    
    Usage:
        monitor = PerformanceMonitor()
        monitor.start()
        
        # В любом месте кода
        monitor.record_api_request('GET', '/api/v1/scans', 200, 0.125)
        monitor.record_sstv_recording()
    """

    def __init__(self, metrics_port: int = 9090):
        """
        Инициализация монитора.
        
        Args:
            metrics_port: Порт для Prometheus metrics endpoint
        """
        self.metrics_port = metrics_port
        self.start_time = time.time()
        self._running = False

    def start(self):
        """Запуск сбора метрик"""
        logger.info(f"Starting performance monitor on port {self.metrics_port}")
        
        # Запуск Prometheus metrics server
        start_http_server(self.metrics_port)
        
        self._running = True
        
        # Запуск фонового сбора системных метрик
        import threading
        thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        thread.start()
        
        logger.info("Performance monitor started")

    def stop(self):
        """Остановка монитора"""
        self._running = False
        logger.info("Performance monitor stopped")

    def _collect_system_metrics(self):
        """Сбор системных метрик"""
        while self._running:
            try:
                # CPU
                for i, cpu_percent in enumerate(psutil.cpu_percent(percpu=True)):
                    CPU_PERCENT.labels(core=f'core_{i}').set(cpu_percent)

                # Memory
                memory = psutil.virtual_memory()
                MEMORY_PERCENT.set(memory.percent)
                MEMORY_AVAILABLE.set(memory.available)

                # Disk
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        DISK_PERCENT.labels(mountpoint=partition.device).set(usage.percent)
                        DISK_FREE.labels(mountpoint=partition.device).set(usage.free)
                    except (PermissionError, OSError):
                        pass

                # Network
                net = psutil.net_io_counters()
                NETWORK_BYTES_SENT.inc(net.bytes_sent)
                NETWORK_BYTES_RECV.inc(net.bytes_recv)

                # Uptime
                UPTIME_SECONDS.inc(time.time() - self.start_time)

                time.sleep(5)  # Сбор каждые 5 секунд

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(5)

    def record_api_request(self, method: str, endpoint: str, status: int, latency: float):
        """
        Запись метрик API запроса.
        
        Args:
            method: HTTP метод (GET, POST, etc.)
            endpoint: Endpoint path
            status: HTTP status code
            latency: Время выполнения в секундах
        """
        API_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        API_REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint
        ).observe(latency)

    def record_sstv_recording(self):
        """Запись SSTV записи"""
        SSTV_RECORDINGS_TOTAL.inc()

    def record_sstv_decoded(self):
        """Запись успешного SSTV декодирования"""
        SSTV_DECODED_IMAGES.inc()

    def update_scans_count(self, count: int):
        """
        Обновление количества сканирований.
        
        Args:
            count: Количество сканирований
        """
        SCANS_TOTAL.set(count)

    def update_simulations_active(self, count: int):
        """
        Обновление количества активных симуляций.
        
        Args:
            count: Количество активных симуляций
        """
        SIMULATIONS_ACTIVE.set(count)

    def update_upload_queue(self, size: int):
        """
        Обновление размера очереди загрузки.
        
        Args:
            size: Размер очереди
        """
        UPLOAD_QUEUE_SIZE.set(size)

    def get_metrics(self) -> str:
        """
        Получение Prometheus метрик.
        
        Returns:
            Строка с метриками в Prometheus format
        """
        return generate_latest().decode('utf-8')


# Singleton instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Получение экземпляра монитора"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def start_monitoring(metrics_port: int = 9090):
    """
    Запуск мониторинга.
    
    Args:
        metrics_port: Порт для Prometheus metrics
    """
    monitor = get_monitor()
    monitor.start()
    logger.info(f"Monitoring started on port {metrics_port}")


def record_api_request(method: str, endpoint: str, status: int, latency: float):
    """Запись API запроса"""
    monitor = get_monitor()
    monitor.record_api_request(method, endpoint, status, latency)


def record_sstv_recording():
    """Запись SSTV записи"""
    monitor = get_monitor()
    monitor.record_sstv_recording()


def record_sstv_decoded():
    """Запись SSTV декодирования"""
    monitor = get_monitor()
    monitor.record_sstv_decoded()


# Middleware для FastAPI
async def monitoring_middleware(request, call_next):
    """
    Middleware для автоматического сбора метрик API.
    
    Usage:
        from fastapi import FastAPI
        from utils.monitoring.performance_monitor import monitoring_middleware
        
        app = FastAPI()
        
        @app.middleware("http")
        async def track_requests(request, call_next):
            return await monitoring_middleware(request, call_next)
    """
    import time
    from fastapi import Request
    
    start_time = time.time()
    
    response = await call_next(request)
    
    latency = time.time() - start_time
    
    record_api_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        latency=latency
    )
    
    return response
