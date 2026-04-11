"""
Enhanced Layout

Полная раскладка: все виджеты.
"""

from ..widgets.base import WidgetMode


class EnhancedLayout:
    """Расширенная раскладка dashboard"""

    name = "enhanced"
    mode = WidgetMode.ENHANCED

    def __init__(self):
        self.visible_widgets = [
            "system_monitor",  # CRITICAL
            "component_status",  # CRITICAL
            "log_viewer",  # HIGH
            "metrics",  # NORMAL
            "activity",  # NORMAL
        ]

    def get_widgets(self, dashboard):
        """Получить виджеты для раскладки"""
        return [
            dashboard.widgets[name] for name in self.visible_widgets if name in dashboard.widgets
        ]
