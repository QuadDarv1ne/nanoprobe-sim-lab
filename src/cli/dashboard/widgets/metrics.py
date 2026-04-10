"""
Metrics Widget

Метрики API: запросы, время ответа, кэш.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from .base import Widget, WidgetData, WidgetPriority


class MetricsWidget(Widget):
    """Виджет метрик API"""

    def __init__(self):
        super().__init__(
            name="metrics",
            title="📊 API Metrics",
            priority=WidgetPriority.NORMAL,
            refresh_interval=10,
            position=(3, 0),
            size=(5, 1),
        )
        self._prev_stats: Optional[Dict] = None

    async def refresh(self) -> WidgetData:
        """Получение метрик из API"""
        import aiohttp

        metrics = {
            'requests_total': 0,
            'requests_per_min': 0,
            'avg_response_ms': 0,
            'cache_hit_rate': 0,
            'active_users': 0,
            'errors_last_hour': 0,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/v1/dashboard/stats', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        metrics.update({
                            'requests_total': data.get('requests_total', 0),
                            'requests_per_min': data.get('requests_per_min', 0),
                            'avg_response_ms': data.get('avg_response_time_ms', 0),
                            'cache_hit_rate': data.get('cache_hit_rate', 0),
                        })
        except Exception:
            # API недоступно — используем последние известные значения
            pass

        # Вычисляем дельту
        if self._prev_stats:
            metrics['requests_delta'] = metrics['requests_total'] - self._prev_stats.get('requests_total', 0)
        else:
            metrics['requests_delta'] = 0

        self._prev_stats = metrics.copy()

        return WidgetData(
            title=self.title,
            content=metrics,
            timestamp=datetime.now(timezone.utc)
        )

    def render(self, width: int = 50) -> str:
        """Отрисовка метрик"""
        if self.error:
            return f"❌ Error: {self.error}"

        if self.data is None:
            return "⏳ Connecting to API..."

        m = self.data.content

        lines = [
            f"Requests:      {m['requests_total']:,} total",
            f"Rate:          {m['requests_per_min']:.1f} req/min",
            f"Avg Response:  {m['avg_response_ms']:.0f} ms",
            f"Cache Hit:     {m['cache_hit_rate'] * 100:.1f}%",
            f"Delta:         {m['requests_delta']:+d} (last check)",
        ]

        return "\n".join(lines)

    def get_health_status(self) -> str:
        """Получить статус здоровья API"""
        if self.data is None:
            return "unknown"

        m = self.data.content

        if m['avg_response_ms'] > 1000:
            return "degraded"
        elif m['avg_response_ms'] > 500:
            return "slow"
        elif m['avg_response_ms'] < 100:
            return "healthy"
        else:
            return "normal"
