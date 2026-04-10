"""
Component Status Widget

Статус компонентов проекта: API, Frontend, Redis, Database.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List

from .base import Widget, WidgetData, WidgetPriority


class ComponentStatusWidget(Widget):
    """Виджет статуса компонентов"""

    def __init__(self):
        super().__init__(
            name="component_status",
            title="🔧 Components",
            priority=WidgetPriority.CRITICAL,
            refresh_interval=10,
            position=(1, 0),
            size=(len(self._get_components()), 1),
        )
        self._components = self._get_components()

    def _get_components(self) -> List[Dict]:
        """Список компонентов для проверки"""
        return [
            {"name": "API Server", "host": "localhost", "port": 8000, "type": "http"},
            {"name": "Flask Frontend", "host": "localhost", "port": 5000, "type": "http"},
            {"name": "Next.js Frontend", "host": "localhost", "port": 3000, "type": "http"},
            {"name": "Redis", "host": "localhost", "port": 6379, "type": "tcp"},
            {"name": "PostgreSQL", "host": "localhost", "port": 5432, "type": "tcp"},
        ]

    async def _check_http(self, host: str, port: int) -> bool:
        """Проверка HTTP сервиса"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return False

    async def _check_tcp(self, host: str, port: int) -> bool:
        """Проверка TCP порта"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return False

    async def refresh(self) -> WidgetData:
        """Обновление статуса компонентов"""
        statuses = {}

        for component in self._components:
            if component["type"] == "http":
                is_up = await self._check_http(component["host"], component["port"])
            else:
                is_up = await self._check_tcp(component["host"], component["port"])

            statuses[component["name"]] = "up" if is_up else "down"

        return WidgetData(title=self.title, content=statuses, timestamp=datetime.now(timezone.utc))

    def render(self, width: int = 40) -> str:
        """Отрисовка статуса компонентов"""
        if self.error:
            return f"❌ Error: {self.error}"

        if self.data is None:
            return "⏳ Loading..."

        lines = []
        statuses = self.data.content

        for name, status in statuses.items():
            icon = "🟢" if status == "up" else "🔴"
            lines.append(f"{icon} {name:20s} {status}")

        return "\n".join(lines)

    def get_summary(self) -> Dict:
        """Получить сводку по компонентам"""
        if self.data is None:
            return {"total": 0, "up": 0, "down": 0}

        statuses = self.data.content.values()
        return {
            "total": len(statuses),
            "up": sum(1 for s in statuses if s == "up"),
            "down": sum(1 for s in statuses if s == "down"),
        }
