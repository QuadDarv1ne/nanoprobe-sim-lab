"""
Performance Monitor - заглушка для совместимости
"""
from typing import Dict, Any


class PerformanceMonitor:
    """Мониторинг производительности"""

    def __init__(self):
        self._metrics: Dict[str, Any] = {}

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def record_metric(self, name: str, value: float):
        self._metrics[name] = value
