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

from fastapi import APIRouter

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

# Создаём основной router который объединяет все подмодули
router = APIRouter()

# Подключаем все роутеры из подмодулей
router.include_router(stats_router)
router.include_router(realtime_router)
router.include_router(activity_router)
router.include_router(alerts_router)
router.include_router(actions_router)

__all__ = [
    "router",
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
