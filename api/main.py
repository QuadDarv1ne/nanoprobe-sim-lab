"""
FastAPI REST API для Nanoprobe Simulation Lab
Совместная работа с Flask веб-интерфейсом
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.error_handlers import register_error_handlers
from api.router_config import register_routes

# Импорт существующих утилит
from utils.database import DatabaseManager

logger = logging.getLogger(__name__)

# Версия приложения (единый источник)
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("Application starting up...")

    # Инициализация Sentry для мониторинга ошибок
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.starlette import StarletteIntegration

            sentry_sdk.init(
                dsn=sentry_dsn,
                traces_sample_rate=0.1,  # 10% транзакций для мониторинга
                environment=os.getenv("ENVIRONMENT", "development"),
                release=APP_VERSION,
                integrations=[
                    StarletteIntegration(),
                    FastApiIntegration(),
                ],
                # Не отправлять PII данные
                send_default_pii=False,
            )
            logger.info("Sentry SDK initialized")
        except ImportError:
            logger.warning("sentry-sdk not installed, skipping Sentry initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize Sentry: {e}")
    else:
        logger.info("SENTRY_DSN not set, skipping Sentry initialization")

    # Инициализация ресурсов при старте
    try:
        from api.state import init_app_state, set_db_manager, set_redis
        from utils.caching.redis_cache import RedisCache
        from utils.database import DatabaseManager

        # Инициализация БД
        db = DatabaseManager("data/nanoprobe.db")
        set_db_manager(db)
        logger.info("Database manager initialized")

        # Инициализация Redis (не критично, если Redis недоступен)
        try:
            redis = RedisCache()
            set_redis(redis)
            if redis.is_available():
                logger.info("Redis cache initialized and connected")
            else:
                logger.info(
                    "Redis cache initialized (Redis server not available, using fallback mode)"
                )
        except Exception as e:
            logger.warning(f"Redis initialization warning: {e}")

        # Инициализация app state
        init_app_state(db, redis if "redis" in locals() else None)
        logger.info("App state initialized")

        # Запуск автоматической очистки rate limiter
        try:
            from utils.security.rate_limiter import start_rate_limit_cleanup

            start_rate_limit_cleanup()
            logger.info("Rate limiter auto-cleanup started")
        except Exception as e:
            logger.warning(f"Failed to start rate limiter cleanup: {e}")

        # Инициализация Sync Manager
        try:
            from api.routes.sync_manager import set_sync_manager
            from api.sync_manager import BackendFrontendSync

            sync_mgr = BackendFrontendSync()
            await sync_mgr.initialize()
            set_sync_manager(sync_mgr)
            logger.info("Sync Manager initialized")

            # Запуск цикла синхронизации в фоне
            asyncio.create_task(sync_mgr.start_sync_loop(interval=5.0))
            logger.info("Sync Manager sync loop started")
        except Exception as e:
            logger.warning(f"Failed to initialize Sync Manager: {e}")

    except Exception as e:
        logger.warning(f"Startup initialization warning (may be expected in dev): {e}")

    # Запуск фоновой задачи push-уведомлений
    push_task = asyncio.create_task(push_realtime_updates())
    logger.info("Realtime push task started")

    yield
    logger.info("Application shutting down...")

    # Остановка фоновой задачи push-уведомлений
    try:
        push_task.cancel()
        await asyncio.gather(push_task, return_exceptions=True)
        logger.info("Realtime push task stopped")
    except Exception as e:
        logger.debug(f"Push task cleanup error: {e}")

    # 1. Остановка мониторинга
    try:
        from utils.monitoring.performance_monitor import get_monitor

        monitor = get_monitor()
        monitor.stop()
        logger.info("Performance monitor stopped")
    except Exception as e:
        logger.debug(f"Monitor cleanup error: {e}")

    # 2. Закрытие circuit breakers
    try:
        from utils.caching.circuit_breaker import close_all_circuit_breakers

        close_all_circuit_breakers()
        logger.info("Circuit breakers closed")
    except Exception as e:
        logger.debug(f"Circuit breakers cleanup error: {e}")

    # 3. Закрытие HTTP сессий
    try:
        from api.routes.external_services import close_http_session

        close_http_session()
        logger.info("HTTP session closed")
    except Exception as e:
        logger.debug(f"HTTP session cleanup error: {e}")

    # 4. Закрытие Redis
    try:
        from api.dependencies import get_redis_cache

        redis = get_redis_cache()
        if redis:
            redis.close()
            logger.info("Redis cache closed")
    except Exception as e:
        logger.debug(f"Redis cleanup error: {e}")

    # 5. Закрытие соединений БД (последним, т.к. может использоваться другими компонентами)
    try:
        from api.state import get_db_manager

        db = get_db_manager()
        if db:
            db.close_pool()
            DatabaseManager.close_all_pools()
            logger.info("Database connections closed")
    except Exception as e:
        logger.debug(f"Database cleanup error: {e}")

    # 6. Закрытие Sync Manager
    try:
        from api.routes.sync_manager import get_sync_manager

        sync_mgr = get_sync_manager()
        if sync_mgr:
            sync_mgr._running = False  # Остановить цикл
            await sync_mgr.close()
            logger.info("Sync Manager stopped")
    except Exception as e:
        logger.debug(f"Sync Manager cleanup error: {e}")

    logger.info("Application stopped gracefully")


# Создание FastAPI приложения
app = FastAPI(
    title="Nanoprobe Sim Lab API",
    description="""
## API для Лаборатории моделирования нанозонда

### Возможности:
- **Сканирования** - управление результатами сканирований СЗМ
- **Симуляции** - запуск и мониторинг симуляций
- **Анализ дефектов** - AI/ML детектирование дефектов
- **Сравнение поверхностей** - сравнение изображений поверхностей
- **PDF отчёты** - генерация научных отчётов
- **WebSocket** - real-time обновления

### Аутентификация:
Большинство эндпоинтов требуют JWT токен. Получите токен через `/api/v1/auth/login`.
    """,
    version=APP_VERSION,
    contact={
        "name": "Школа программирования Maestro7IT",
        "email": "maksimqwe42@mail.ru",
        "url": "https://school-maestro7it.ru/",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://github.com/QuadDarv1ne/nanoprobe-sim-lab/blob/main/legal/LICENCE_RU",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS настройки (для Next.js frontend в будущем)
_cors_env = os.getenv("CORS_ORIGINS", "")
if _cors_env.startswith("["):
    # JSON формат: ["url1","url2"]
    CORS_ORIGINS = json.loads(_cors_env)
else:
    # CSV формат: url1,url2,url3
    CORS_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

if not CORS_ORIGINS:
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5000", "http://127.0.0.1:5000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# GZip сжатие для уменьшения размера ответов
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus middleware для сбора метрик
try:
    from api.metrics import PrometheusMiddleware

    app.add_middleware(PrometheusMiddleware)
except ImportError:
    pass

# Rate Limiting для защиты от DDoS/bruteforce
try:
    from api.rate_limiter import setup_rate_limiter

    setup_rate_limiter(app)
    logger.info("Rate limiting enabled")
except ImportError as e:
    logger.warning(f"Rate limiting disabled: {e}")

# Security Headers для защиты от XSS, Clickjacking, MIME sniffing
try:
    from api.security_headers import setup_security_headers

    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    setup_security_headers(app, production=is_production)
except ImportError as e:
    logger.warning(f"Security headers disabled: {e}")

# Регистрация централизованных обработчиков ошибок
register_error_handlers(app)


# Middleware для мониторинга производительности
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware для сбора метрик производительности"""
    import time

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    # Запись метрик
    try:
        from utils.monitoring.performance_monitor import record_api_request

        record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            latency=latency,
        )
    except Exception as e:
        logger.debug(f"Metrics recording error: {e}")

    return response


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья API"""
    import logging
    import traceback

    logger = logging.getLogger(__name__)
    logger.info("Health check endpoint called")
    try:
        result = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": APP_VERSION,
        }
        logger.info(f"Health check result: {result}")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(f"HEALTH ERROR: {e}\n{tb}")  # noqa: T201
        return {"error": str(e), "traceback": tb}


# Detailed health check
@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Детальная проверка здоровья системы"""
    from api.health import CPU_CRITICAL, DISK_CRITICAL, MEMORY_CRITICAL, compute_system_health

    health = compute_system_health()

    # Redis status
    try:
        from api.state import get_redis

        _r = get_redis()
        cache_status = "running" if _r and _r.is_available() else "unavailable"
    except Exception as e:
        logger.debug("Failed to check Redis status: %s", e)
        cache_status = "unavailable"

    return {
        "status": health["status"],
        "timestamp": health["timestamp"],
        "version": APP_VERSION,
        "python_version": f"{os.sys.version}",
        "database": "SQLite 3.x",
        "metrics": {
            "cpu": {
                "percent": health["cpu_percent"],
                "status": (
                    "ok"
                    if health["cpu_percent"] is None or health["cpu_percent"] < CPU_CRITICAL
                    else "warning"
                ),
            },
            "memory": {
                "percent": health["memory_percent"],
                "status": (
                    "ok"
                    if health["memory_percent"] is None
                    or health["memory_percent"] < MEMORY_CRITICAL
                    else "warning"
                ),
            },
            "disk": {
                "percent": health["disk_percent"],
                "status": (
                    "ok"
                    if health["disk_percent"] is None or health["disk_percent"] < DISK_CRITICAL
                    else "warning"
                ),
            },
        },
        "issues": health["issues"],
        "services": {
            "api": "running",
            "database": "running",
            "cache": cache_status,
        },
    }


# Realtime metrics
@app.get("/metrics/realtime", tags=["Monitoring"])
async def realtime_metrics():
    """Метрики в реальном времени"""
    import psutil

    from api.state import get_system_disk_usage

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": get_system_disk_usage().percent,
    }


# Главная страница API
@app.get("/api", tags=["Root"])
async def api_root():
    """Информация об API"""
    return {
        "name": "Nanoprobe Sim Lab API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


# Регистрация всех роутов через централизованный модуль
register_routes(app)

# Раздача статических файлов (изображения сканов, отчёты и т.д.)
try:
    from fastapi.staticfiles import StaticFiles

    # Монтируем директорию data для доступа к файлам сканов
    if Path("data").exists():
        app.mount("/data", StaticFiles(directory="data"), name="data")
        logger.info("Static files mounted: /data -> data/")

    # Монтируем директорию output для отчётов и визуализаций
    if Path("output").exists():
        app.mount("/output", StaticFiles(directory="output"), name="output")
        logger.info("Static files mounted: /output -> output/")
except Exception as e:
    logger.warning(f"Static files mounting disabled: {e}")


# Metrics endpoint для Prometheus
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus метрики"""
    from utils.monitoring.monitoring import get_monitor

    # Получаем метрики из enhanced monitor
    monitor = get_monitor()
    prometheus_metrics = monitor.export_prometheus_metrics()

    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(content=prometheus_metrics, media_type="text/plain")


# GraphQL endpoint
@app.post("/graphql")
async def graphql_endpoint(request: Request):
    """GraphQL endpoint для запросов"""
    from api.graphql_schema import schema

    body = await request.json()
    query = body.get("query")
    variables = body.get("variables")
    operation_name = body.get("operationName")

    result = await schema.execute(query, variable_values=variables, operation_name=operation_name)

    if result.errors:
        logger.error(f"GraphQL errors: {result.errors}")

    return {"data": result.data, "errors": result.errors}


@app.get("/graphql/playground")
async def graphql_playground():
    """GraphQL Playground UI"""
    return {
        "message": "GraphQL Playground",
        "endpoint": "/graphql",
        "example_query": """
query {
    stats {
        totalScans
        totalSimulations
        totalImages
        activeSimulations
    }
    scans(limit: 10) {
        id
        scanType
        timestamp
    }
}
""",
    }


# WebSocket эндпоинт для real-time обновлений
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений с ConnectionManager"""
    from api.websocket_manager import get_connection_manager

    manager = get_connection_manager()

    # Принятие подключения через менеджер
    if not await manager.connect(websocket):
        return

    last_pong = datetime.now(timezone.utc)

    try:
        while True:
            try:
                # Получение команд от клиента с timeout (30s heartbeat)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                last_pong = datetime.now(timezone.utc)

                # Валидация сообщения через менеджер
                try:
                    message = manager.validate_message(data)
                except ValueError as e:
                    await manager.send_personal(
                        websocket,
                        {
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    continue

                # Обработка команд
                if message.get("type") == "subscribe":
                    channel = message.get("channel")
                    if await manager.subscribe(websocket, channel):
                        await manager.send_personal(
                            websocket,
                            {
                                "type": "subscribed",
                                "channel": channel,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        logger.info(f"Client subscribed to channel: {channel}")
                    else:
                        await manager.send_personal(
                            websocket,
                            {
                                "type": "error",
                                "message": f"Failed to subscribe to {channel}",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                elif message.get("type") == "unsubscribe":
                    channel = message.get("channel")
                    await manager.unsubscribe(websocket, channel)
                    await manager.send_personal(
                        websocket,
                        {
                            "type": "unsubscribed",
                            "channel": channel,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                elif message.get("type") == "ping":
                    await manager.send_personal(
                        websocket,
                        {
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                elif message.get("type") == "get_metrics":
                    # Отправка текущих метрик
                    import psutil

                    from api.state import get_system_disk_usage

                    metrics = {
                        "type": "metrics",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "cpu_percent": psutil.cpu_percent(interval=0.1),
                            "memory_percent": psutil.virtual_memory().percent,
                            "disk_percent": get_system_disk_usage().percent,
                        },
                    }
                    await manager.send_personal(websocket, metrics)

            except asyncio.TimeoutError:
                # Проверка heartbeat
                if (datetime.now(timezone.utc) - last_pong).total_seconds() > 60:
                    logger.warning("Client heartbeat timeout, disconnecting")
                    break
                continue

    except asyncio.CancelledError:
        # Задача отменена (например, при shutdown)
        logger.info("WebSocket task cancelled, closing connection")
        await manager.disconnect(websocket)
        raise  # Re-raise для корректной отмены задачи
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Проверка что подключение ещё активно
        try:
            await manager.disconnect(websocket)
        except Exception:
            pass  # Уже отключено, это нормально


# Фоновая задача для push-уведомлений подписчикам
async def push_realtime_updates():
    """Периодическая отправка метрик подписчикам канала 'metrics'"""
    import psutil

    from api.state import get_system_disk_usage
    from api.websocket_manager import get_connection_manager

    while True:
        try:
            await asyncio.sleep(5)

            manager = get_connection_manager()
            if not manager.get_stats().get("total_connections", 0):
                continue  # нет подключений — пропускаем

            payload = {
                "type": "metrics_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": get_system_disk_usage().percent,
                },
            }
            await manager.send_to_channel("metrics", payload)

        except asyncio.CancelledError:
            logger.info("Realtime updates task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in push_realtime_updates: {e}")


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД (устарело, используйте api.dependencies.get_db)"""
    from api.state import get_db_manager

    return get_db_manager()


def create_app() -> FastAPI:
    """Фабрика приложения (для тестов)"""
    return app


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
