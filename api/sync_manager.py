"""
Менеджер синхронизации Backend (FastAPI) ↔ Frontend (Flask)
Централизованное управление WebSocket и API интеграцией
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 30.0
HEALTH_CHECK_TIMEOUT = 5
SYNC_TIMEOUT = 10


class BackendFrontendSync:
    """
    Менеджер синхронизации между Backend и Frontend

    Обеспечивает:
    1. Трансляцию событий из Backend во Frontend
    2. Синхронизацию WebSocket подключений
    3. Кэширование общих данных
    4. Health monitoring обоих сервисов
    """

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        frontend_url: str = "http://localhost:5000",
    ):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: Dict[str, Set] = {
            "backend": set(),  # WebSocket подключения к FastAPI
            "frontend": set(),  # Socket.IO подключения к Flask
        }
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._last_sync_time: Optional[datetime] = None

    async def initialize(self):
        """Инициализация HTTP сессии"""
        self.session = aiohttp.ClientSession()
        logger.info(
            f"[SYNC] Инициализирован: Backend={self.backend_url}, Frontend={self.frontend_url}"
        )

    async def close(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()
            logger.info("[SYNC] Сессия закрыта")

    async def check_backend_health(self) -> bool:
        """Проверка доступности Backend"""
        if not self.session:
            logger.error("❌ Session not initialized")
            return False

        try:
            async with self.session.get(
                f"{self.backend_url}/health",
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return True
                else:
                    logger.warning(f"⚠️ Backend вернул статус {resp.status}")
                    return False
        except asyncio.TimeoutError:
            logger.error(f"❌ Backend health check timeout after {HEALTH_CHECK_TIMEOUT}s")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Backend недоступен (client error): {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Backend недоступен (unknown error): {e}", exc_info=True)
            return False

    async def check_frontend_health(self) -> bool:
        """Проверка доступности Frontend"""
        if not self.session:
            logger.error("❌ Session not initialized")
            return False

        try:
            async with self.session.get(
                f"{self.frontend_url}/api/health",
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return True
                else:
                    logger.warning(f"⚠️ Frontend вернул статус {resp.status}")
                    return False
        except asyncio.TimeoutError:
            logger.error(f"❌ Frontend health check timeout after {HEALTH_CHECK_TIMEOUT}s")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Frontend недоступен (client error): {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Frontend недоступен (unknown error): {e}", exc_info=True)
            return False

    async def sync_dashboard_stats(self) -> Optional[Dict[str, Any]]:
        """
        Синхронизация статистики дашборда

        Получает данные из Backend и возвращает для передачи во Frontend
        """
        if not self.session:
            logger.error("❌ Session not initialized")
            return None

        try:
            async with self.session.get(
                f"{self.backend_url}/api/v1/dashboard/stats",
                timeout=aiohttp.ClientTimeout(total=SYNC_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    stats = await resp.json()
                    self._last_sync_time = datetime.now(timezone.utc)
                    logger.info(f"[SYNC] Статистика обновлена: {len(stats)} полей")
                    return stats
                else:
                    error_body = await resp.text()
                    logger.warning(
                        f"[SYNC] Ошибка получения статистики: {resp.status} - {error_body[:200]}"
                    )
                    return None
        except asyncio.TimeoutError:
            logger.error("[SYNC] Timeout получения статистики")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"[SYNC] Client error при получении статистики: {e}")
            return None
        except Exception as e:
            logger.error(f"[SYNC] Ошибка: {e}", exc_info=True)
            return None

    async def sync_realtime_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Синхронизация метрик реального времени

        Получает метрики из Backend для передачи во Frontend
        """
        try:
            assert self.session is not None
            async with self.session.get(
                f"{self.backend_url}/api/v1/dashboard/metrics/realtime",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    metrics = await resp.json()
                    logger.debug(
                        f"[SYNC] Метрики получены: CPU={metrics.get('cpu_percent', 'N/A')}%"
                    )
                    return metrics
                else:
                    return None
        except Exception as e:
            logger.error(f"[SYNC] Ошибка получения метрик: {e}")
            return None

    async def broadcast_to_frontend(self, event: str, data: Dict[str, Any]):
        """
        Отправка события во Frontend через Socket.IO

        Args:
            event: Имя события
            data: Данные события
        """
        try:
            assert self.session is not None
            # Socket.IO HTTP API для отправки событий
            payload = {
                "event": event,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Отправка через Flask-SocketIO HTTP endpoint
            async with self.session.post(
                f"{self.frontend_url}/socketio/event",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    logger.debug(f"[SYNC] Событие '{event}' отправлено во Frontend")
                else:
                    logger.warning(f"[SYNC] Ошибка отправки события: {resp.status}")
        except Exception as e:
            logger.error(f"[SYNC] Ошибка отправки события: {e}")

    async def start_sync_loop(self, interval: float = 5.0):
        """
        Запуск цикла синхронизации с exponential backoff при ошибках

        Args:
            interval: Базовый интервал синхронизации в секундах
        """
        self._running = True
        consecutive_failures = 0
        logger.info(f"[SYNC] Запуск цикла синхронизации (интервал={interval}с)")

        while self._running:
            try:
                # Проверка здоровья сервисов
                backend_ok = await self.check_backend_health()
                frontend_ok = await self.check_frontend_health()

                if backend_ok and frontend_ok:
                    consecutive_failures = 0  # Reset on success

                    # Синхронизация статистики
                    stats = await self.sync_dashboard_stats()
                    if stats:
                        await self.broadcast_to_frontend("stats_update", stats)

                    # Синхронизация метрик
                    metrics = await self.sync_realtime_metrics()
                    if metrics:
                        await self.broadcast_to_frontend("metrics_update", metrics)

                    self._last_sync_time = datetime.now(timezone.utc)
                    await asyncio.sleep(interval)
                else:
                    # Health check failed - increase delay
                    consecutive_failures += 1
                    delay = min(
                        interval * (2 ** min(consecutive_failures, 5)),  # Exponential backoff
                        MAX_RECONNECT_DELAY,
                    )
                    logger.warning(
                        f"[SYNC] Health check failed (attempt {consecutive_failures}), "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            except asyncio.CancelledError:
                logger.info("[SYNC] Цикл синхронизации отменён")
                break
            except Exception as e:
                consecutive_failures += 1
                delay = min(interval * (2 ** min(consecutive_failures, 5)), MAX_RECONNECT_DELAY)
                logger.error(
                    f"[SYNC] Ошибка в цикле синхронизации (attempt {consecutive_failures}): "
                    f"{e}, retrying in {delay:.1f}s",
                    exc_info=True,
                )
                await asyncio.sleep(delay)

    def stop_sync_loop(self):
        """Остановка цикла синхронизации"""
        self._running = False
        logger.info("[SYNC] Остановка цикла синхронизации")

    def get_sync_status(self) -> Dict[str, Any]:
        """Получение статуса синхронизации"""
        return {
            "running": self._running,
            "backend_url": self.backend_url,
            "frontend_url": self.frontend_url,
            "last_sync_time": self._last_sync_time.isoformat() if self._last_sync_time else None,
            "backend_connections": len(self.ws_connections["backend"]),
            "frontend_connections": len(self.ws_connections["frontend"]),
        }


# ==================== Интеграция с Flask ====================


def setup_flask_sync_integration(app, socketio):
    """
    Настройка интеграции синхронизации с Flask приложением

    Args:
        app: Flask приложение
        socketio: Flask-SocketIO экземпляр
    """
    from flask import request
    from flask_socketio import emit

    sync_manager = BackendFrontendSync()

    @socketio.on("connect")
    def handle_connect():
        """Подключение клиента к Socket.IO"""
        logger.info(f"[SOCKET] Клиент подключился: {request.sid}")
        sync_manager.ws_connections["frontend"].add(request.sid)
        emit(
            "connected",
            {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sync_status": sync_manager.get_sync_status(),
            },
        )

    @socketio.on("disconnect")
    def handle_disconnect():
        """Отключение клиента от Socket.IO"""
        logger.info(f"[SOCKET] Клиент отключился: {request.sid}")
        sync_manager.ws_connections["frontend"].discard(request.sid)

    @socketio.on("request_stats")
    def handle_request_stats():
        """Запрос статистики из Backend"""
        try:
            loop = asyncio.new_event_loop()
            stats = loop.run_until_complete(sync_manager.sync_dashboard_stats())
            loop.close()
            if stats:
                emit("stats", stats)
        except Exception as e:
            logger.error(f"[SOCKET] Ошибка получения статистики: {e}")
            emit("error", {"message": str(e)})

    @socketio.on("request_metrics")
    def handle_request_metrics():
        """Запрос метрик из Backend"""
        try:
            loop = asyncio.new_event_loop()
            metrics = loop.run_until_complete(sync_manager.sync_realtime_metrics())
            loop.close()
            if metrics:
                emit("metrics", metrics)
        except Exception as e:
            logger.error(f"[SOCKET] Ошибка получения метрик: {e}")
            emit("error", {"message": str(e)})

    # HTTP endpoint для приёма событий от Backend
    @app.route("/socketio/event", methods=["POST"])
    def handle_backend_event():
        """
        HTTP endpoint для приёма событий от Backend

        Backend может отправлять события сюда для трансляции во Frontend
        """
        from flask import jsonify, request

        data = request.json
        event = data.get("event")
        event_data = data.get("data")

        if event and event_data:
            socketio.emit(event, event_data)
            logger.debug(f"[HTTP] Событие '{event}' получено от Backend")
            return jsonify({"status": "ok"})
        else:
            return jsonify({"status": "error", "message": "Invalid event"}), 400

    return sync_manager


# ==================== Запуск синхронизации ====================


async def run_sync_manager():
    """Запуск менеджера синхронизации"""
    sync = BackendFrontendSync()
    await sync.initialize()

    try:
        await sync.start_sync_loop(interval=5.0)
    except KeyboardInterrupt:
        logger.info("[SYNC] Остановка по сигналу")
    finally:
        sync.stop_sync_loop()
        await sync.close()


if __name__ == "__main__":
    # Запуск для тестирования
    asyncio.run(run_sync_manager())
