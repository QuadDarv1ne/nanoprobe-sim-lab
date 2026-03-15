"""
CLI Dashboard Widgets

Модульная система виджетов для консольного dashboard.
Каждый виджет независим и может быть включён/выключен.
"""

from .base import Widget, WidgetPriority
from .system_monitor import SystemMonitorWidget
from .component_status import ComponentStatusWidget
from .log_viewer import LogViewerWidget
from .metrics import MetricsWidget
from .activity import ActivityWidget

__all__ = [
    'Widget',
    'WidgetPriority',
    'SystemMonitorWidget',
    'ComponentStatusWidget',
    'LogViewerWidget',
    'MetricsWidget',
    'ActivityWidget',
]
