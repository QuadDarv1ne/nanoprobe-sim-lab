"""
CLI Dashboard Widgets

Модульная система виджетов для консольного dashboard.
Каждый виджет независим и может быть включён/выключен.
"""

from .activity import ActivityWidget
from .base import Widget, WidgetPriority
from .component_status import ComponentStatusWidget
from .log_viewer import LogViewerWidget
from .metrics import MetricsWidget
from .system_monitor import SystemMonitorWidget

__all__ = [
    "Widget",
    "WidgetPriority",
    "SystemMonitorWidget",
    "ComponentStatusWidget",
    "LogViewerWidget",
    "MetricsWidget",
    "ActivityWidget",
]
