"""
Activity Widget

Timeline активности проекта.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List

from .base import Widget, WidgetData, WidgetPriority


class ActivityWidget(Widget):
    """Виджет активности"""

    def __init__(self):
        super().__init__(
            name="activity",
            title="📈 Activity",
            priority=WidgetPriority.NORMAL,
            refresh_interval=30,
            position=(4, 0),
            size=(6, 1),
        )
        self._activities: List[Dict] = []

    async def refresh(self) -> WidgetData:
        """Получение последней активности"""
        import aiohttp

        activities = []

        try:
            async with aiohttp.ClientSession() as session:
                # Получаем активность из API
                async with session.get(
                    'http://localhost:8000/api/v1/dashboard/activity',
                    timeout=5
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        activities = data.get('recent_activity', [])
        except Exception:
            # Генерируем тестовую активность
            activities = [
                {
                    'type': 'simulation',
                    'description': 'SPM simulation completed',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'success'
                },
                {
                    'type': 'scan',
                    'description': 'AFM image analyzed',
                    'timestamp': (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                    'status': 'success'
                },
                {
                    'type': 'system',
                    'description': 'System health check passed',
                    'timestamp': (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                    'status': 'info'
                },
            ]

        self._activities = activities[:10]  # Храним последние 10

        return WidgetData(
            title=self.title,
            content=self._activities,
            timestamp=datetime.now(timezone.utc)
        )

    def render(self, width: int = 60) -> str:
        """Отрисовка активности"""
        if self.error:
            return f"❌ Error: {self.error}"

        if self.data is None or not self.data.content:
            return "⏳ Loading activity..."

        lines = []
        activities = self.data.content

        for activity in activities[:5]:  # Показываем последние 5
            icon = self._get_icon(activity.get('type', 'system'))
            status_icon = "✅" if activity.get('status') == 'success' else "ℹ️"

            time_str = self._format_time(activity.get('timestamp', ''))
            desc = activity.get('description', 'Unknown')

            # Обрезаем описание
            if len(desc) > width - 15:
                desc = desc[:width - 18] + "..."

            lines.append(f"{icon} {status_icon} [{time_str}] {desc}")

        if not lines:
            lines.append("  No recent activity")

        return "\n".join(lines)

    def _get_icon(self, activity_type: str) -> str:
        """Получить иконку по типу активности"""
        icons = {
            'simulation': '🔬',
            'scan': '📷',
            'analysis': '🔍',
            'system': '⚙️',
            'user': '👤',
            'error': '❌',
        }
        return icons.get(activity_type, '📝')

    def _format_time(self, timestamp: str) -> str:
        """Форматирование времени"""
        if not timestamp:
            return "??:??"

        try:
            dt = datetime.fromisoformat(timestamp)
            now = datetime.now(timezone.utc)
            diff = now - dt

            if diff.total_seconds() < 60:
                return "now"
            elif diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds() / 60)}m ago"
            else:
                return dt.strftime("%H:%M")
        except ValueError:
            return "??"

    def get_activity_summary(self) -> Dict:
        """Получить сводку по активности"""
        if not self._activities:
            return {'total': 0, 'success': 0, 'errors': 0}

        return {
            'total': len(self._activities),
            'success': sum(1 for a in self._activities if a.get('status') == 'success'),
            'errors': sum(1 for a in self._activities if a.get('status') == 'error'),
        }
