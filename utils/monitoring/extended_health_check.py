"""
Extended Health Checks для фоновых задач

Проверяет здоровье:
- WebSocket connections
- TLE (Two-Line Element) updates
- RTL-SDR device status
- Background tasks (asyncio)
- Redis connection
- Database connection pool
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExtendedHealthChecker:
    """
    Расширенная проверка здоровья для фоновых задач
    """

    def __init__(
        self,
        db_path: str = "data/nanoprobe.db",
        redis_url: Optional[str] = None,
        tle_update_interval: int = 3600,  # 1 час
    ):
        self.db_path = Path(db_path)
        self.redis_url = redis_url
        self.tle_update_interval = tle_update_interval
        self._last_tle_update: Optional[datetime] = None
        self._ws_connections = 0
        self._background_tasks: Dict[str, asyncio.Task] = {}

    def register_background_task(self, name: str, task: asyncio.Task):
        """Регистрация фоновой задачи для мониторинга"""
        self._background_tasks[name] = task
        logger.info(f"Registered background task: {name}")

    def update_ws_connections(self, count: int):
        """Обновление счётчика WebSocket подключений"""
        self._ws_connections = count

    def update_tle_timestamp(self):
        """Обновление времени последнего TLE обновления"""
        self._last_tle_update = datetime.now(timezone.utc)

    async def check_database(self) -> Dict[str, Any]:
        """Проверка здоровья базы данных"""
        try:
            import sqlite3

            if not self.db_path.exists():
                return {"status": "error", "message": "Database file not found"}

            conn = sqlite3.connect(str(self.db_path), timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()

            return {
                "status": "healthy",
                "path": str(self.db_path),
                "size_mb": self.db_path.stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def check_redis(self) -> Dict[str, Any]:
        """Проверка здоровья Redis"""
        if not self.redis_url:
            return {"status": "disabled", "message": "Redis not configured"}

        try:
            import redis.asyncio as redis

            client = redis.from_url(self.redis_url)
            await client.ping()
            await client.close()

            return {"status": "healthy", "url": self.redis_url}
        except ImportError:
            return {"status": "disabled", "message": "Redis library not installed"}
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def check_websocket(self) -> Dict[str, Any]:
        """Проверка WebSocket подключений"""
        return {
            "status": "healthy" if self._ws_connections >= 0 else "error",
            "active_connections": self._ws_connections,
        }

    async def check_tle_updates(self) -> Dict[str, Any]:
        """Проверка актуальности TLE данных"""
        if not self._last_tle_update:
            return {
                "status": "warning",
                "message": "TLE never updated",
                "last_update": None,
            }

        now = datetime.now(timezone.utc)
        time_since_update = (now - self._last_tle_update).total_seconds()

        if time_since_update > self.tle_update_interval * 2:
            status = "error"
            message = f"TLE data is stale ({time_since_update / 3600:.1f}h old)"
        elif time_since_update > self.tle_update_interval:
            status = "warning"
            message = f"TLE data approaching refresh ({time_since_update / 3600:.1f}h old)"
        else:
            status = "healthy"
            message = f"TLE data fresh ({time_since_update / 60:.1f}m old)"

        return {
            "status": status,
            "message": message,
            "last_update": self._last_tle_update.isoformat(),
            "age_seconds": time_since_update,
            "refresh_interval_seconds": self.tle_update_interval,
        }

    async def check_rtl_sdr(self) -> Dict[str, Any]:
        """Проверка статуса RTL-SDR устройства"""
        try:
            # Проверяем наличие утилит
            import subprocess

            result = subprocess.run(
                ["rtl_test"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return {
                    "status": "healthy",
                    "message": "RTL-SDR device detected",
                    "output": result.stdout[:200],
                }
            else:
                return {
                    "status": "warning",
                    "message": "RTL-SDR device not found or error",
                    "error": result.stderr[:200],
                }

        except FileNotFoundError:
            return {
                "status": "disabled",
                "message": "rtl_test not found in PATH",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": "RTL-SDR check timeout",
            }
        except Exception as e:
            logger.error(f"RTL-SDR health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def check_background_tasks(self) -> Dict[str, Any]:
        """Проверка фоновых задач"""
        tasks_status = {}

        for name, task in self._background_tasks.items():
            if task.done():
                if task.exception():
                    tasks_status[name] = {
                        "status": "error",
                        "message": str(task.exception()),
                    }
                else:
                    tasks_status[name] = {"status": "completed"}
            else:
                tasks_status[name] = {"status": "running"}

        overall_status = "healthy"
        if any(t["status"] == "error" for t in tasks_status.values()):
            overall_status = "error"
        elif any(t["status"] == "warning" for t in tasks_status.values()):
            overall_status = "warning"

        return {"status": overall_status, "tasks": tasks_status}

    async def full_health_check(self) -> Dict[str, Any]:
        """Полная проверка здоровья всех компонентов"""
        checks = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_websocket(),
            self.check_tle_updates(),
            self.check_rtl_sdr(),
            self.check_background_tasks(),
        )

        result = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "database": checks[0],
                "redis": checks[1],
                "websocket": checks[2],
                "tle_updates": checks[3],
                "rtl_sdr": checks[4],
                "background_tasks": checks[5],
            },
        }

        # Определяем общий статус
        statuses = [check["status"] for check in result["components"].values()]
        if "error" in statuses:
            result["status"] = "degraded"
        elif "warning" in statuses:
            result["status"] = "warning"

        return result


# Singleton instance
_health_checker: Optional[ExtendedHealthChecker] = None


def get_health_checker() -> ExtendedHealthChecker:
    """Получение singleton экземпляра health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = ExtendedHealthChecker()
    return _health_checker


def init_health_checker(
    db_path: str = "data/nanoprobe.db",
    redis_url: Optional[str] = None,
    tle_update_interval: int = 3600,
) -> ExtendedHealthChecker:
    """Инициализация health checker"""
    global _health_checker
    _health_checker = ExtendedHealthChecker(
        db_path=db_path,
        redis_url=redis_url,
        tle_update_interval=tle_update_interval,
    )
    return _health_checker
