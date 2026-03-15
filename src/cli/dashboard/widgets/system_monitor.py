"""
System Monitor Widget

Мониторинг системы: CPU, память, диск, сеть.
"""

import psutil
from datetime import datetime
from typing import Dict
from .base import Widget, WidgetData, WidgetPriority


class SystemMonitorWidget(Widget):
    """Виджет мониторинга системы"""

    def __init__(self):
        super().__init__(
            name="system_monitor",
            title="🖥️ System Monitor",
            priority=WidgetPriority.CRITICAL,
            refresh_interval=5,
            position=(0, 0),
            size=(4, 1),  # 4 строки, полная ширина
        )
        self._history = []

    async def refresh(self) -> WidgetData:
        """Обновление метрик системы"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()

        metrics = {
            'cpu': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available': memory.available // (1024 * 1024),  # MB
            'disk_percent': disk.percent,
            'disk_free': disk.free // (1024 * 1024 * 1024),  # GB
            'net_sent': net.bytes_sent // (1024 * 1024),  # MB
            'net_recv': net.bytes_recv // (1024 * 1024),  # MB
        }

        # Сохраняем историю (последние 60 значений)
        self._history.append((datetime.now(), metrics))
        if len(self._history) > 60:
            self._history.pop(0)

        return WidgetData(
            title=self.title,
            content=metrics,
            timestamp=datetime.now()
        )

    def render(self, width: int = 60) -> str:
        """Отрисовка виджета"""
        if self.error:
            return self._render_error(width)

        if self.data is None:
            return self._render_loading(width)

        metrics = self.data.content
        lines = []

        # CPU
        cpu_bar = self._make_bar(metrics['cpu'], width - 15)
        cpu_color = self._get_color(metrics['cpu'])
        lines.append(f"CPU    [{cpu_color}{cpu_bar}{self._reset_color()}] {metrics['cpu']:5.1f}%")

        # Memory
        mem_bar = self._make_bar(metrics['memory_percent'], width - 15)
        mem_color = self._get_color(metrics['memory_percent'])
        lines.append(f"RAM    [{mem_color}{mem_bar}{self._reset_color()}] {metrics['memory_percent']:5.1f}% ({metrics['memory_available']} MB)")

        # Disk
        disk_bar = self._make_bar(metrics['disk_percent'], width - 15)
        disk_color = self._get_color(metrics['disk_percent'])
        lines.append(f"Disk   [{disk_color}{disk_bar}{self._reset_color()}] {metrics['disk_percent']:5.1f}% ({metrics['disk_free']} GB free)")

        # Network
        lines.append(f"Net ↑↓ {metrics['net_sent']:6d} MB | {metrics['net_recv']:6d} MB")

        return "\n".join(lines)

    def _make_bar(self, percent: float, width: int) -> str:
        """Создать прогресс бар"""
        filled = int(width * percent / 100)
        return "█" * filled + "░" * (width - filled)

    def _get_color(self, value: float) -> str:
        """Получить ANSI цвет по значению"""
        if value < 50:
            return "\033[92m"  # Green
        elif value < 80:
            return "\033[93m"  # Yellow
        else:
            return "\033[91m"  # Red

    def _reset_color(self) -> str:
        """Сброс цвета"""
        return "\033[0m"

    def _render_error(self, width: int) -> str:
        """Отрисовка ошибки"""
        return f"❌ Error: {self.error}"

    def _render_loading(self, width: int) -> str:
        """Отрисовка загрузки"""
        return "⏳ Loading..."

    def get_trend(self, metric: str, samples: int = 5) -> str:
        """
        Получить тренд метрики.

        Args:
            metric: Имя метрики (cpu, memory_percent, disk_percent)
            samples: Количество сэмплов для анализа

        Returns:
            str: "↑" (рост), "↓" (падение), "→" (стабильно)
        """
        if len(self._history) < samples:
            return "→"

        recent = [h[1][metric] for h in self._history[-samples:]]
        avg_first = sum(recent[:2]) / 2
        avg_last = sum(recent[-2:]) / 2

        diff = avg_last - avg_first
        if diff > 2:
            return "↑"
        elif diff < -2:
            return "↓"
        else:
            return "→"
