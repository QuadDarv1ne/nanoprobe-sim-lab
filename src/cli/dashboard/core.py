"""
Unified CLI Dashboard Core

Единый dashboard с модульной архитектурой виджетов.
Поддержка разных режимов отображения: Minimal, Standard, Enhanced.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Type
from enum import Enum

# Добавляем parent directory в path для импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.cli.dashboard.widgets.base import Widget, WidgetPriority, WidgetMode, WidgetData
from src.cli.dashboard.widgets.system_monitor import SystemMonitorWidget
from src.cli.dashboard.widgets.component_status import ComponentStatusWidget
from src.cli.dashboard.widgets.log_viewer import LogViewerWidget
from src.cli.dashboard.widgets.metrics import MetricsWidget
from src.cli.dashboard.widgets.activity import ActivityWidget

import logging
logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Темы оформления"""
    DARK = "dark"
    LIGHT = "light"
    COLOR = "color"


class UnifiedDashboard:
    """
    Единый CLI Dashboard с поддержкой разных режимов.

    Features:
    - Modular widget system
    - Multiple display modes (Minimal, Standard, Enhanced)
    - Real-time updates
    - Keyboard navigation
    - Extensible architecture

    Usage:
        dashboard = UnifiedDashboard(mode=WidgetMode.ENHANCED)
        await dashboard.start()
    """

    def __init__(
        self,
        mode: WidgetMode = WidgetMode.ENHANCED,
        theme: DashboardTheme = DashboardTheme.DARK,
        refresh_interval: int = 5,
    ):
        self.mode = mode
        self.theme = theme
        self.refresh_interval = refresh_interval
        self.widgets: Dict[str, Widget] = {}
        self.running = False
        self._refresh_task: Optional[asyncio.Task] = None

        # Инициализация основных виджетов
        self._init_core_widgets()

    def _init_core_widgets(self):
        """Инициализация основных виджетов"""
        # System Monitor (CRITICAL)
        self.register_widget(SystemMonitorWidget())

        # Component Status (CRITICAL)
        self.register_widget(ComponentStatusWidget())

        # Log Viewer (HIGH)
        self.register_widget(LogViewerWidget())

        # Metrics (NORMAL) - виден только в ENHANCED
        self.register_widget(MetricsWidget())

        # Activity (NORMAL) - виден только в ENHANCED
        self.register_widget(ActivityWidget())

    def register_widget(self, widget: Widget):
        """Зарегистрировать виджет"""
        self.widgets[widget.name] = widget
        logger.debug(f"Registered widget: {widget.name}")

    def unregister_widget(self, name: str):
        """Удалить виджет"""
        if name in self.widgets:
            del self.widgets[name]
            logger.debug(f"Unregistered widget: {name}")

    def set_mode(self, mode: WidgetMode):
        """Изменить режим отображения"""
        self.mode = mode
        logger.info(f"Dashboard mode changed to: {mode.value}")

    def get_visible_widgets(self) -> List[Widget]:
        """Получить видимые виджеты для текущего режима"""
        visible = [w for w in self.widgets.values() if w.is_visible(self.mode)]
        return sorted(visible, key=lambda w: w.priority.value)

    async def refresh_all(self):
        """Обновить все видимые виджеты"""
        tasks = []
        for widget in self.get_visible_widgets():
            tasks.append(widget._safe_refresh())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def render(self):
        """Отрендерить dashboard"""
        # Очистка экрана
        print("\033[2J\033[H", end="")

        # Header
        print(self._render_header())

        # Виджеты по строкам
        visible = self.get_visible_widgets()

        # Группировка по строкам
        rows: Dict[int, List[Widget]] = {}
        for widget in visible:
            row = widget.position[0]
            if row not in rows:
                rows[row] = []
            rows[row].append(widget)

        # Отрисовка каждой строки
        for row_num in sorted(rows.keys()):
            row_widgets = sorted(rows[row_num], key=lambda w: w.position[1])
            print(self._render_row(row_widgets))

        # Footer
        print(self._render_footer())

    def _render_header(self) -> str:
        """Отрисовка заголовка"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_str = f"{self.mode.value:10}"

        # Цвета для темы
        if self.theme == DashboardTheme.DARK:
            header = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🔬 Nanoprobe Sim Lab Dashboard    │ Mode: {mode_str} │ {now}     ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""
        else:
            header = f"""
+==============================================================================+
|  Nanoprobe Sim Lab Dashboard    | Mode: {mode_str} | {now}     |
+------------------------------------------------------------------------------+
"""
        return header

    def _render_row(self, widgets: List[Widget]) -> str:
        """Отрисовка строки виджетов"""
        lines = []
        for widget in widgets:
            # Заголовок виджета
            status_icon = widget.get_status_icon()
            title_line = f"├─ {status_icon} {widget.title} " + "─" * (70 - len(widget.title))
            lines.append(title_line)

            # Содержимое
            content = widget.render(width=65)
            for line in content.split("\n"):
                lines.append(f"│  {line}")

        return "\n".join(lines)

    def _render_footer(self) -> str:
        """Отрисовка футера"""
        if self.theme == DashboardTheme.DARK:
            footer = """
╠══════════════════════════════════════════════════════════════════════════════╣
║  [Q] Quit  [R] Refresh  [M] Mode  [1-5] Toggle Widget  [H] Help              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        else:
            footer = """
+------------------------------------------------------------------------------+
|  [Q] Quit  [R] Refresh  [M] Mode  [1-5] Toggle Widget  [H] Help              |
+------------------------------------------------------------------------------+
"""
        return footer

    async def start(self):
        """Запустить dashboard с auto-refresh"""
        self.running = True

        async def refresh_loop():
            while self.running:
                await self.refresh_all()
                self.render()
                await asyncio.sleep(self.refresh_interval)

        self._refresh_task = asyncio.create_task(refresh_loop())

        # Обработка клавиш (в фоне)
        await self._handle_input()

    async def _handle_input(self):
        """Обработка ввода пользователя (асинхронно)"""
        # Примечание: это упрощённая версия
        # Для полноценной обработки нужен curses или подобная библиотека
        pass

    async def stop(self):
        """Остановить dashboard"""
        self.running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

    def get_summary(self) -> Dict:
        """Получить сводку dashboard"""
        visible = self.get_visible_widgets()
        return {
            'mode': self.mode.value,
            'total_widgets': len(self.widgets),
            'visible_widgets': len(visible),
            'theme': self.theme.value,
        }


def run_dashboard(mode: str = "enhanced", theme: str = "dark"):
    """
    Точка входа для запуска dashboard.

    Args:
        mode: Режим отображения (minimal, standard, enhanced)
        theme: Тема (dark, light, color)
    """
    mode_map = {
        'minimal': WidgetMode.MINIMAL,
        'standard': WidgetMode.STANDARD,
        'enhanced': WidgetMode.ENHANCED,
    }

    theme_map = {
        'dark': DashboardTheme.DARK,
        'light': DashboardTheme.LIGHT,
        'color': DashboardTheme.COLOR,
    }

    dashboard_mode = mode_map.get(mode.lower(), WidgetMode.ENHANCED)
    dashboard_theme = theme_map.get(theme.lower(), DashboardTheme.DARK)

    dashboard = UnifiedDashboard(mode=dashboard_mode, theme=dashboard_theme)

    try:
        print(f"Starting Dashboard in {mode} mode...")
        print("Press Ctrl+C to stop\n")
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "enhanced"
    theme = sys.argv[2] if len(sys.argv) > 2 else "dark"
    run_dashboard(mode, theme)
