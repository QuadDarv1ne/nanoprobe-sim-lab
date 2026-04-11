"""
Minimal Layout

Минимальная раскладка: только CRITICAL виджеты.
"""

from ..widgets.base import WidgetMode


class MinimalLayout:
    """Минимальная раскладка dashboard"""

    name = "minimal"
    mode = WidgetMode.MINIMAL

    def __init__(self):
        self.visible_widgets = [
            "system_monitor",  # CRITICAL
            "component_status",  # CRITICAL
        ]

    def get_widgets(self, dashboard):
        """Получить виджеты для раскладки"""
        return [
            dashboard.widgets[name] for name in self.visible_widgets if name in dashboard.widgets
        ]
