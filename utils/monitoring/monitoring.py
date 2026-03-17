"""
Enhanced System Monitor for Nanoprobe Sim Lab
Расширенный мониторинг системы с аналитикой и алертами
"""

import psutil
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Структура системных метрик"""
    timestamp: str
    cpu_percent: float
    cpu_cores: int
    cpu_freq_mhz: Optional[float]
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    processes_count: int
    uptime_seconds: int
    boot_time: str


@dataclass
class Alert:
    """Структура алерта"""
    level: str  # info, warning, critical
    component: str
    message: str
    timestamp: str
    value: Optional[float] = None
    threshold: Optional[float] = None


class EnhancedSystemMonitor:
    """
    Расширенный монитор системы
    Предоставляет детальную информацию о системе с аналитикой
    """

    def __init__(self, history_size: int = 300):
        """
        Инициализация монитора

        Args:
            history_size: Количество записей в истории (по умолчанию 300 = 5 минут при 1с)
        """
        self.history_size = history_size
        self.monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.update_interval = 1.0  # секунды

        # История метрик
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics: Optional[SystemMetrics] = None

        # Статистика
        self.start_time: Optional[datetime] = None
        self.total_samples = 0

        # Пороги для алертов
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_warning": 80.0,
            "disk_critical": 95.0,
        }

        # Алерты
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Последнее время сбора метрик
        self.last_net_io = psutil.net_io_counters()
        self.last_net_time = time.time()

    def start_monitoring(self):
        """Запуск мониторинга"""
        if self.monitoring:
            return

        self.monitoring = True
        self.start_time = datetime.now()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self.monitoring_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(asdict(metrics))
                self.total_samples += 1

                # Проверка алертов
                self._check_alerts(metrics)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            time.sleep(self.update_interval)

    def _collect_metrics(self) -> SystemMetrics:
        """Сбор текущих метрик"""
        now = datetime.now()
        timestamp = now.isoformat()

        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else None

        # Memory
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)

        # Disk
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)

        # Network
        net_io = psutil.net_io_counters()
        network_bytes_sent = net_io.bytes_sent
        network_bytes_recv = net_io.bytes_recv
        network_packets_sent = net_io.packets_sent
        network_packets_recv = net_io.packets_recv

        # Processes
        processes_count = len(psutil.pids())

        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time()).isoformat()
        uptime_seconds = int((now - datetime.fromtimestamp(psutil.boot_time())).total_seconds())

        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_cores=psutil.cpu_count(logical=True),
            cpu_freq_mhz=cpu_freq_mhz,
            memory_percent=memory.percent,
            memory_used_gb=round(memory_used_gb, 2),
            memory_total_gb=round(memory_total_gb, 2),
            memory_available_gb=round(memory_available_gb, 2),
            disk_percent=disk.percent,
            disk_used_gb=round(disk_used_gb, 2),
            disk_total_gb=round(disk_total_gb, 2),
            disk_free_gb=round(disk_free_gb, 2),
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            network_packets_sent=network_packets_sent,
            network_packets_recv=network_packets_recv,
            processes_count=processes_count,
            uptime_seconds=uptime_seconds,
            boot_time=boot_time
        )

    def _check_alerts(self, metrics: SystemMetrics):
        """Проверка метрик на превышение порогов"""
        # CPU alerts
        if metrics.cpu_percent >= self.thresholds["cpu_critical"]:
            self._add_alert("critical", "cpu",
                          f"Критическая загрузка CPU: {metrics.cpu_percent:.1f}%",
                          metrics.cpu_percent, self.thresholds["cpu_critical"])
        elif metrics.cpu_percent >= self.thresholds["cpu_warning"]:
            self._add_alert("warning", "cpu",
                          f"Высокая загрузка CPU: {metrics.cpu_percent:.1f}%",
                          metrics.cpu_percent, self.thresholds["cpu_warning"])

        # Memory alerts
        if metrics.memory_percent >= self.thresholds["memory_critical"]:
            self._add_alert("critical", "memory",
                          f"Критическое использование памяти: {metrics.memory_percent:.1f}%",
                          metrics.memory_percent, self.thresholds["memory_critical"])
        elif metrics.memory_percent >= self.thresholds["memory_warning"]:
            self._add_alert("warning", "memory",
                          f"Высокое использование памяти: {metrics.memory_percent:.1f}%",
                          metrics.memory_percent, self.thresholds["memory_warning"])

        # Disk alerts
        if metrics.disk_percent >= self.thresholds["disk_critical"]:
            self._add_alert("critical", "disk",
                          f"Критическое заполнение диска: {metrics.disk_percent:.1f}%",
                          metrics.disk_percent, self.thresholds["disk_critical"])
        elif metrics.disk_percent >= self.thresholds["disk_warning"]:
            self._add_alert("warning", "disk",
                          f"Высокое заполнение диска: {metrics.disk_percent:.1f}%",
                          metrics.disk_percent, self.thresholds["disk_warning"])

    def _add_alert(self, level: str, component: str, message: str,
                   value: Optional[float] = None, threshold: Optional[float] = None):
        """Добавление алерта"""
        alert = Alert(
            level=level,
            component=component,
            message=message,
            timestamp=datetime.now().isoformat(),
            value=value,
            threshold=threshold
        )
        self.alerts.append(alert)

        # Оставляем только последние 100 алертов
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        # Вызов callback'ов
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Добавление callback'а для алертов"""
        self.alert_callbacks.append(callback)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Получение текущих метрик"""
        if self.current_metrics:
            return asdict(self.current_metrics)
        return asdict(self._collect_metrics())

    def get_metrics_history(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Получение истории метрик"""
        return list(self.metrics_history)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики"""
        if not self.metrics_history:
            return {}

        history_list = list(self.metrics_history)

        def avg(key):
            """
            Вычисляет среднее значение метрики.

            Args:
                key: Ключ метрики

            Returns:
                Среднее значение или 0
            """
            values = [m[key] for m in history_list if key in m]
            return sum(values) / len(values) if values else 0

        def max_val(key):
            """
            Вычисляет максимальное значение метрики.

            Args:
                key: Ключ метрики

            Returns:
                Максимальное значение или 0
            """
            values = [m[key] for m in history_list if key in m]
            return max(values) if values else 0

        def min_val(key):
            """
            Вычисляет минимальное значение метрики.

            Args:
                key: Ключ метрики

            Returns:
                Минимальное значение или 0
            """
            values = [m[key] for m in history_list if key in m]
            return min(values) if values else 0

        return {
            "samples": self.total_samples,
            "uptime_seconds": self.current_metrics.uptime_seconds if self.current_metrics else 0,
            "cpu": {
                "avg": round(avg("cpu_percent"), 2),
                "max": round(max_val("cpu_percent"), 2),
                "min": round(min_val("cpu_percent"), 2),
                "current": self.current_metrics.cpu_percent if self.current_metrics else 0
            },
            "memory": {
                "avg": round(avg("memory_percent"), 2),
                "max": round(max_val("memory_percent"), 2),
                "min": round(min_val("memory_percent"), 2),
                "current": self.current_metrics.memory_percent if self.current_metrics else 0
            },
            "disk": {
                "avg": round(avg("disk_percent"), 2),
                "current": self.current_metrics.disk_percent if self.current_metrics else 0
            },
            "alerts_count": len(self.alerts),
            "recent_alerts": [asdict(a) for a in self.alerts[-10:]]
        }

    def get_network_speed(self) -> Dict[str, float]:
        """Получение скорости сети (байт/сек)"""
        now = time.time()
        current_net_io = psutil.net_io_counters()

        time_diff = now - self.last_net_time
        if time_diff <= 0:
            time_diff = 1.0

        upload_speed = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / time_diff
        download_speed = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / time_diff

        self.last_net_io = current_net_io
        self.last_net_time = now

        return {
            "upload_bps": upload_speed,
            "download_bps": download_speed,
            "upload_mbps": round(upload_speed * 8 / 1_000_000, 2),
            "download_mbps": round(download_speed * 8 / 1_000_000, 2)
        }

    def get_process_list(self, limit: int = 10, sort_by: str = "cpu") -> List[Dict[str, Any]]:
        """Получение списка процессов"""
        processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "cpu_percent": proc.info['cpu_percent'] or 0,
                    "memory_percent": proc.info['memory_percent'] or 0
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Сортировка
        if sort_by == "cpu":
            processes.sort(key=lambda x: x["cpu_percent"], reverse=True)
        elif sort_by == "memory":
            processes.sort(key=lambda x: x["memory_percent"], reverse=True)

        return processes[:limit]

    def set_thresholds(self, thresholds: Dict[str, float]):
        """Установка порогов для алертов"""
        self.thresholds.update(thresholds)
        logger.info(f"Alert thresholds updated: {thresholds}")

    def get_alerts(self, limit: int = 50, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение алертов"""
        alerts = self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        return [asdict(a) for a in alerts[-limit:]]

    def clear_alerts(self):
        """Очистка алертов"""
        self.alerts.clear()
        logger.info("Alerts cleared")

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format
        Returns:
            String with metrics in Prometheus format
        """
        metrics = self.get_current_metrics()

        lines = [
            "# HELP nanoprobe_cpu_percent Current CPU usage percentage",
            "# TYPE nanoprobe_cpu_percent gauge",
            f"nanoprobe_cpu_percent {metrics['cpu_percent']}",
            "",
            "# HELP nanoprobe_cpu_cores Number of CPU cores",
            "# TYPE nanoprobe_cpu_cores gauge",
            f"nanoprobe_cpu_cores {metrics['cpu_cores']}",
            "",
            "# HELP nanoprobe_memory_percent Memory usage percentage",
            "# TYPE nanoprobe_memory_percent gauge",
            f"nanoprobe_memory_percent {metrics['memory_percent']}",
            "",
            "# HELP nanoprobe_memory_used_gb Memory used (GB)",
            "# TYPE nanoprobe_memory_used_gb gauge",
            f"nanoprobe_memory_used_gb {metrics['memory_used_gb']}",
            "",
            "# HELP nanoprobe_memory_total_gb Total memory (GB)",
            "# TYPE nanoprobe_memory_total_gb gauge",
            f"nanoprobe_memory_total_gb {metrics['memory_total_gb']}",
            "",
            "# HELP nanoprobe_disk_percent Disk usage percentage",
            "# TYPE nanoprobe_disk_percent gauge",
            f"nanoprobe_disk_percent {metrics['disk_percent']}",
            "",
            "# HELP nanoprobe_disk_used_gb Disk used (GB)",
            "# TYPE nanoprobe_disk_used_gb gauge",
            f"nanoprobe_disk_used_gb {metrics['disk_used_gb']}",
            "",
            "# HELP nanoprobe_disk_total_gb Total disk space (GB)",
            "# TYPE nanoprobe_disk_total_gb gauge",
            f"nanoprobe_disk_total_gb {metrics['disk_total_gb']}",
            "",
            "# HELP nanoprobe_uptime_seconds System uptime in seconds",
            "# TYPE nanoprobe_uptime_seconds counter",
            f"nanoprobe_uptime_seconds {metrics['uptime_seconds']}",
            "",
            "# HELP nanoprobe_processes_count Number of processes",
            "# TYPE nanoprobe_processes_count gauge",
            f"nanoprobe_processes_count {metrics['processes_count']}",
            "",
            "# HELP nanoprobe_network_bytes_sent Network bytes sent",
            "# TYPE nanoprobe_network_bytes_sent counter",
            f"nanoprobe_network_bytes_sent {metrics['network_bytes_sent']}",
            "",
            "# HELP nanoprobe_network_bytes_recv Network bytes received",
            "# TYPE nanoprobe_network_bytes_recv counter",
            f"nanoprobe_network_bytes_recv {metrics['network_bytes_recv']}",
            "",
            "# HELP nanoprobe_alerts_count Number of alerts",
            "# TYPE nanoprobe_alerts_count gauge",
            f"nanoprobe_alerts_count {len(self.alerts)}",
            "",
            "# HELP nanoprobe_samples_count Number of collected samples",
            "# TYPE nanoprobe_samples_count counter",
            f"nanoprobe_samples_count {self.total_samples}",
        ]

        return "\n".join(lines)


# Singleton instance
_monitor_instance: Optional[EnhancedSystemMonitor] = None


def get_monitor() -> EnhancedSystemMonitor:
    """Получение singleton экземпляра монитора"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = EnhancedSystemMonitor()
        _monitor_instance.start_monitoring()
    return _monitor_instance


def stop_monitor() -> None:
    """Остановка singleton экземпляра"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop_monitoring()
        _monitor_instance = None


# Утилита для форматирования аптайма
def format_uptime(seconds: int) -> str:
    """Форматирование аптайма в человекочитаемый вид"""
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days} дн")
    if hours > 0:
        parts.append(f"{hours} ч")
    if minutes > 0:
        parts.append(f"{minutes} мин")

    return " ".join(parts) if parts else "< 1 мин"


if __name__ == "__main__":
    # Тестирование
    import pprint

    monitor = get_monitor()

    print("Testing EnhancedSystemMonitor...")
    time.sleep(3)

    print("\nCurrent Metrics:")
    pprint.pprint(monitor.get_current_metrics())

    print("\nStatistics:")
    pprint.pprint(monitor.get_statistics())

    print("\nTop 5 Processes by CPU:")
    pprint.pprint(monitor.get_process_list(limit=5))

    print("\nNetwork Speed:")
    pprint.pprint(monitor.get_network_speed())

    stop_monitor()
    print("\nMonitor stopped.")
