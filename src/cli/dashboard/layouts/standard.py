"""
Standard Layout

Стандартная раскладка: CRITICAL + HIGH виджеты.
"""

from ..widgets.base import WidgetMode


class StandardLayout:
    """Стандартная раскладка dashboard"""

    name = "standard"
    mode = WidgetMode.STANDARD

    def __init__(self):
        self.visible_widgets = [
            'system_monitor',      # CRITICAL
            'component_status',    # CRITICAL
            'log_viewer',          # HIGH
        ]

    def get_widgets(self, dashboard):
        """Получить виджеты для раскладки"""
        return [
            dashboard.widgets[name]
            for name in self.visible_widgets
            if name in dashboard.widgets
        ]
