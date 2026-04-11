"""
Dashboard API модули

Разбитие большого dashboard.py файла на отдельные модули:
- helpers: Утилиты, кэширование, константы
- stats: Статистика дашборда, health checks
- realtime: Real-time метрики, WebSocket
- activity: Activity timeline, история метрик
- alerts: Настройки алертов, мониторинг процессов
- actions: Экспорт данных, управление компонентами
"""

from api.routes.dashboard.actions import router as actions_router
from api.routes.dashboard.activity import router as activity_router
from api.routes.dashboard.alerts import router as alerts_router
from api.routes.dashboard.helpers import (
    CACHE_PREFIX,
    METRICS_CACHE_TTL,
    STATS_CACHE_TTL,
    cache_stats,
    get_cached_stats,
    get_project_root,
    get_storage_stats,
)
from api.routes.dashboard.realtime import router as realtime_router
from api.routes.dashboard.stats import router as stats_router

__all__ = [
    "stats_router",
    "realtime_router",
    "activity_router",
    "alerts_router",
    "actions_router",
    "CACHE_PREFIX",
    "METRICS_CACHE_TTL",
    "STATS_CACHE_TTL",
    "cache_stats",
    "get_cached_stats",
    "get_project_root",
    "get_storage_stats",
]
