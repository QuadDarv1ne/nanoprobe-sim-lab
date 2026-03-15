"""
CLI Dashboard Package

Модульный CLI dashboard с виджетами и раскладками.

Usage:
    from src.cli.dashboard import UnifiedDashboard, WidgetMode
    dashboard = UnifiedDashboard(mode=WidgetMode.ENHANCED)
    await dashboard.start()
"""

from .core import UnifiedDashboard, run_dashboard
from .widgets.base import Widget, WidgetPriority, WidgetMode, WidgetData

__all__ = [
    'UnifiedDashboard',
    'run_dashboard',
    'Widget',
    'WidgetPriority',
    'WidgetMode',
    'WidgetData',
]
