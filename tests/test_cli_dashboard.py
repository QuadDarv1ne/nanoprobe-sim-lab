"""
Тесты для модульного CLI Dashboard
"""

import pytest
import asyncio
from datetime import datetime

from src.cli.dashboard.widgets.base import Widget, WidgetPriority, WidgetMode, WidgetData
from src.cli.dashboard.widgets.system_monitor import SystemMonitorWidget
from src.cli.dashboard.widgets.component_status import ComponentStatusWidget
from src.cli.dashboard.widgets.log_viewer import LogViewerWidget
from src.cli.dashboard.core import UnifiedDashboard, DashboardTheme


# Тесты базового виджета
class TestWidgetBase:
    """Тесты базового класса Widget"""

    def test_widget_creation(self):
        """Создание виджета"""
        widget = Widget(
            name="test_widget",
            title="Test Widget",
            priority=WidgetPriority.NORMAL,
            refresh_interval=10,
        )

        assert widget.name == "test_widget"
        assert widget.title == "Test Widget"
        assert widget.priority == WidgetPriority.NORMAL
        assert widget.refresh_interval == 10
        assert widget.enabled is True

    def test_widget_visibility(self):
        """Проверка видимости виджета в разных режимах"""
        critical_widget = Widget(
            name="critical",
            title="Critical",
            priority=WidgetPriority.CRITICAL,
        )

        high_widget = Widget(
            name="high",
            title="High",
            priority=WidgetPriority.HIGH,
        )

        normal_widget = Widget(
            name="normal",
            title="Normal",
            priority=WidgetPriority.NORMAL,
        )

        # MINIMAL режим - только CRITICAL
        assert critical_widget.is_visible(WidgetMode.MINIMAL) is True
        assert high_widget.is_visible(WidgetMode.MINIMAL) is False
        assert normal_widget.is_visible(WidgetMode.MINIMAL) is False

        # STANDARD режим - CRITICAL + HIGH
        assert critical_widget.is_visible(WidgetMode.STANDARD) is True
        assert high_widget.is_visible(WidgetMode.STANDARD) is True
        assert normal_widget.is_visible(WidgetMode.STANDARD) is False

        # ENHANCED режим - все виджеты
        assert critical_widget.is_visible(WidgetMode.ENHANCED) is True
        assert high_widget.is_visible(WidgetMode.ENHANCED) is True
        assert normal_widget.is_visible(WidgetMode.ENHANCED) is True

    def test_widget_disabled(self):
        """Проверка отключенного виджета"""
        widget = Widget(
            name="disabled",
            title="Disabled",
            priority=WidgetPriority.CRITICAL,
            enabled=False,
        )

        # Отключенный виджет не виден ни в одном режиме
        assert widget.is_visible(WidgetMode.MINIMAL) is False
        assert widget.is_visible(WidgetMode.STANDARD) is False
        assert widget.is_visible(WidgetMode.ENHANCED) is False


# Тесты System Monitor Widget
class TestSystemMonitorWidget:
    """Тесты виджета мониторинга системы"""

    @pytest.mark.asyncio
    async def test_system_monitor_refresh(self):
        """Обновление данных системного монитора"""
        widget = SystemMonitorWidget()
        data = await widget.refresh()

        assert data is not None
        assert data.title == widget.title
        assert data.content is not None

        metrics = data.content
        assert 'cpu' in metrics
        assert 'memory_percent' in metrics
        assert 'disk_percent' in metrics
        assert 'net_sent' in metrics
        assert 'net_recv' in metrics

        # Проверка диапазонов
        assert 0 <= metrics['cpu'] <= 100
        assert 0 <= metrics['memory_percent'] <= 100
        assert 0 <= metrics['disk_percent'] <= 100

    def test_system_monitor_render(self):
        """Отрисовка системного монитора"""
        widget = SystemMonitorWidget()

        # Без данных (loading)
        render_str = widget.render(width=60)
        assert "Loading" in render_str or "⏳" in render_str

    def test_system_monitor_progress_bar(self):
        """Создание прогресс бара"""
        widget = SystemMonitorWidget()

        # 0%
        bar = widget._make_bar(0, 10)
        assert len(bar) == 10
        assert bar == "░" * 10

        # 50%
        bar = widget._make_bar(50, 10)
        assert len(bar) == 10
        assert bar == "█" * 5 + "░" * 5

        # 100%
        bar = widget._make_bar(100, 10)
        assert len(bar) == 10
        assert bar == "█" * 10


# Тесты Component Status Widget
class TestComponentStatusWidget:
    """Тесты виджета статуса компонентов"""

    def test_component_status_creation(self):
        """Создание виджета статуса"""
        widget = ComponentStatusWidget()

        assert widget.name == "component_status"
        assert widget.priority == WidgetPriority.CRITICAL
        assert widget.refresh_interval == 10

    @pytest.mark.asyncio
    async def test_component_status_refresh(self):
        """Обновление статуса компонентов"""
        widget = ComponentStatusWidget()
        data = await widget.refresh()

        assert data is not None
        assert isinstance(data.content, dict)

        # Компоненты должны быть в статусе up или down
        for name, status in data.content.items():
            assert status in ['up', 'down']


# Тесты Log Viewer Widget
class TestLogViewerWidget:
    """Тесты виджета просмотра логов"""

    @pytest.mark.asyncio
    async def test_log_viewer_refresh(self):
        """Обновление логов"""
        widget = LogViewerWidget(max_lines=5)
        data = await widget.refresh()

        assert data is not None
        assert isinstance(data.content, list)
        assert len(data.content) <= 5

    def test_log_viewer_filter(self):
        """Фильтрация логов"""
        widget = LogViewerWidget()
        widget.data = WidgetData(
            title="Test Logs",
            content=[
                "INFO: System started",
                "ERROR: Connection failed",
                "WARNING: Low memory",
                "INFO: User logged in",
            ],
            timestamp=datetime.now()
        )

        # Фильтр по INFO
        info_logs = widget.filter_by_level("INFO")
        assert len(info_logs) == 2

        # Фильтр по ERROR
        error_logs = widget.filter_by_level("ERROR")
        assert len(error_logs) == 1

    def test_log_viewer_error_count(self):
        """Подсчёт количества ошибок"""
        widget = LogViewerWidget()
        widget.data = WidgetData(
            title="Test Logs",
            content=[
                "INFO: System started",
                "ERROR: Connection failed",
                "ERROR: Database error",
                "CRITICAL: System crash",
            ],
            timestamp=datetime.now()
        )

        error_count = widget.get_error_count()
        assert error_count == 2  # ERROR + CRITICAL


# Тесты Unified Dashboard
class TestUnifiedDashboard:
    """Тесты единого dashboard"""

    def test_dashboard_creation(self):
        """Создание dashboard"""
        dashboard = UnifiedDashboard(
            mode=WidgetMode.ENHANCED,
            theme=DashboardTheme.DARK,
            refresh_interval=5,
        )

        assert dashboard.mode == WidgetMode.ENHANCED
        assert dashboard.theme == DashboardTheme.DARK
        assert len(dashboard.widgets) >= 3  # Минимум 3 виджета

    def test_dashboard_register_widget(self):
        """Регистрация виджета"""
        dashboard = UnifiedDashboard()

        initial_count = len(dashboard.widgets)

        new_widget = Widget(
            name="custom_widget",
            title="Custom Widget",
            priority=WidgetPriority.LOW,
        )

        dashboard.register_widget(new_widget)

        assert len(dashboard.widgets) == initial_count + 1
        assert "custom_widget" in dashboard.widgets

    def test_dashboard_unregister_widget(self):
        """Удаление виджета"""
        dashboard = UnifiedDashboard()

        # Создаём и регистрируем временный виджет
        temp_widget = Widget(
            name="temp_widget",
            title="Temp",
            priority=WidgetPriority.LOW,
        )
        dashboard.register_widget(temp_widget)

        assert "temp_widget" in dashboard.widgets

        # Удаляем
        dashboard.unregister_widget("temp_widget")
        assert "temp_widget" not in dashboard.widgets

    def test_dashboard_get_visible_widgets(self):
        """Получение видимых виджетов"""
        dashboard = UnifiedDashboard(mode=WidgetMode.MINIMAL)

        visible = dashboard.get_visible_widgets()

        # В MINIMAL режиме только CRITICAL
        for widget in visible:
            assert widget.priority == WidgetPriority.CRITICAL

    def test_dashboard_set_mode(self):
        """Смена режима dashboard"""
        dashboard = UnifiedDashboard(mode=WidgetMode.MINIMAL)

        assert dashboard.mode == WidgetMode.MINIMAL

        dashboard.set_mode(WidgetMode.ENHANCED)

        assert dashboard.mode == WidgetMode.ENHANCED

    def test_dashboard_get_summary(self):
        """Получение сводки dashboard"""
        dashboard = UnifiedDashboard(mode=WidgetMode.STANDARD)

        summary = dashboard.get_summary()

        assert 'mode' in summary
        assert 'total_widgets' in summary
        assert 'visible_widgets' in summary
        assert 'theme' in summary

        assert summary['mode'] == 'standard'
        assert summary['total_widgets'] >= 3


# Тесты виджетов в сборе
class TestWidgetIntegration:
    """Интеграционные тесты виджетов"""

    @pytest.mark.asyncio
    async def test_all_widgets_refresh(self):
        """Обновление всех виджетов"""
        widgets = [
            SystemMonitorWidget(),
            ComponentStatusWidget(),
            LogViewerWidget(),
            MetricsWidget(),
        ]

        for widget in widgets:
            data = await widget.refresh()
            assert data is not None
            assert data.title == widget.title

    def test_widget_priority_ordering(self):
        """Проверка порядка приоритетов виджетов"""
        widgets = [
            Widget(name="critical", title="Critical", priority=WidgetPriority.CRITICAL),
            Widget(name="high", title="High", priority=WidgetPriority.HIGH),
            Widget(name="normal", title="Normal", priority=WidgetPriority.NORMAL),
            Widget(name="low", title="Low", priority=WidgetPriority.LOW),
        ]

        # Сортировка по приоритету
        sorted_widgets = sorted(widgets, key=lambda w: w.priority.value)

        assert sorted_widgets[0].priority == WidgetPriority.CRITICAL
        assert sorted_widgets[1].priority == WidgetPriority.HIGH
        assert sorted_widgets[2].priority == WidgetPriority.NORMAL
        assert sorted_widgets[3].priority == WidgetPriority.LOW


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/cli/dashboard", "--cov-report=term-missing"])
