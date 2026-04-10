"""
Log Viewer Widget

Просмотр последних логов проекта.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from .base import Widget, WidgetData, WidgetPriority


class LogViewerWidget(Widget):
    """Виджет просмотра логов"""

    def __init__(self, log_file: str = "logs/api/nanoprobe_info.log", max_lines: int = 10):
        super().__init__(
            name="log_viewer",
            title="📋 Recent Logs",
            priority=WidgetPriority.HIGH,
            refresh_interval=5,
            position=(2, 0),
            size=(max_lines + 2, 1),
        )
        self.log_file = Path(log_file)
        self.max_lines = max_lines
        self._last_position = 0

    async def refresh(self) -> WidgetData:
        """Чтение последних строк логов"""
        logs = []

        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    # Читаем с конца файла
                    lines = f.readlines()
                    logs = [line.strip() for line in lines[-self.max_lines:]]
            except Exception as e:
                logs = [f"Error reading log: {e}"]
        else:
            logs = ["Log file not found"]

        return WidgetData(
            title=self.title,
            content=logs,
            timestamp=datetime.now(timezone.utc)
        )

    def render(self, width: int = 60) -> str:
        """Отрисовка логов"""
        if self.error:
            return f"❌ Error: {self.error}"

        if self.data is None:
            return "⏳ Loading..."

        logs = self.data.content
        lines = []

        for log in logs:
            # Обрезаем длинные строки
            if len(log) > width - 2:
                log = log[:width - 5] + "..."

            # Добавляем отступ
            lines.append(f"  {log}")

        if not logs:
            lines.append("  No recent logs")

        return "\n".join(lines)

    def filter_by_level(self, level: str) -> List[str]:
        """
        Фильтрация логов по уровню.

        Args:
            level: Уровень логирования (INFO, WARNING, ERROR)

        Returns:
            List[str]: Отфильтрованные логи
        """
        if self.data is None:
            return []

        return [log for log in self.data.content if level in log]

    def get_error_count(self) -> int:
        """Получить количество ошибок"""
        if self.data is None:
            return 0

        return sum(1 for log in self.data.content if 'ERROR' in log or 'CRITICAL' in log)
