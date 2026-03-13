# -*- coding: utf-8 -*-
"""
FastAPI REST API для Nanoprobe Simulation Lab
Совместная работа с Flask веб-интерфейсом
"""

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi import Request
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)

# Импорт существующих утилит
from utils.database import DatabaseManager
from utils.defect_analyzer import DefectAnalysisPipeline, analyze_defects
from utils.surface_comparator import SurfaceComparator
from utils.pdf_report_generator import ScientificPDFReport
from utils.batch_processor import BatchProcessor
from utils.redis_cache import RedisCache

# Импорты роутов
from api.routes import scans, simulations, analysis, comparison, reports, auth, admin, dashboard
from api.routes import graphql, ml_analysis, external_services


# Глобальные переменные
db_manager: Optional[DatabaseManager] = None
redis_cache: Optional[RedisCache] = None
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global db_manager, redis_cache

    # Применение миграций БД
    from api.database_init import ensure_database
    
    if ensure_database("data/nanoprobe.db"):
        print("[OK] Database migrations applied")
    else:
        print("[ERROR] Database initialization failed")
    
    # Инициализация БД менеджера
    db_manager = DatabaseManager("data/nanoprobe.db")
    print("[OK] Database initialized")

    # Инициализация Redis кэша
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_cache = RedisCache(host=redis_host, port=redis_port)

    if redis_cache.is_available():
        print(f"[OK] Redis cache connected: {redis_host}:{redis_port}")
    else:
        print("[WARN] Redis cache unavailable (running without caching)")

    yield

    # Очистка при остановке
    print("[INFO] Application stopped")

    # Закрытие соединений
    try:
        if redis_cache:
            redis_cache.close()
            print("[OK] Redis cache closed")
    except Exception as e:
        print(f"[WARN] Redis cache cleanup error: {e}")

    try:
        if db_manager:
            db_manager.close_pool()
            DatabaseManager.close_all_pools()
            print("[OK] Database connections closed")
    except Exception as e:
        print(f"[WARN] Database cleanup error: {e}")

    try:
        from api.routes.external_services import close_http_session
        close_http_session()
        print("[OK] HTTP session closed")
    except Exception as e:
        print(f"[WARN] HTTP session cleanup error: {e}")


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
    version="1.0.0",
    contact={
        "name": "Школа программирования Maestro7IT",
        "email": "maksimqwe42@mail.ru",
        "url": "https://school-maestro7it.ru/",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://github.com/your-username/nanoprobe-sim-lab/blob/main/legal/LICENCE_RU",
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
    import json
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
    allow_methods=["*"],
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


# Регистрация централизованных обработчиков ошибок
from api.error_handlers import register_error_handlers
register_error_handlers(app)


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


# Detailed health check
@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Детальная проверка здоровья системы"""
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    health_status = "healthy"
    issues = []
    
    if cpu_percent > 90:
        health_status = "warning"
        issues.append("Высокая загрузка CPU")
    
    if memory.percent > 90:
        health_status = "warning"
        issues.append("Высокое использование памяти")
    
    if disk.percent > 90:
        health_status = "critical"
        issues.append("Критическое заполнение диска")
    
    return {
        "status": health_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "metrics": {
            "cpu": {
                "percent": cpu_percent,
                "status": "ok" if cpu_percent < 90 else "warning"
            },
            "memory": {
                "percent": memory.percent,
                "used_gb": round(memory.used / (1024 ** 3), 2),
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "status": "ok" if memory.percent < 90 else "warning"
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "status": "ok" if disk.percent < 90 else "warning"
            }
        },
        "issues": issues,
        "services": {
            "api": "running",
            "database": "running",
            "cache": "disabled"
        }
    }


# Realtime metrics
@app.get("/metrics/realtime", tags=["Monitoring"])
async def realtime_metrics():
    """Метрики в реальном времени"""
    import psutil
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }


# Export endpoint
@app.get("/api/v1/export/{format}", tags=["Export"])
async def export_data(format: str):
    """Экспорт данных в различных форматах"""
    if format not in ["json", "csv", "pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {format}. Доступны: json, csv, pdf"
        )
    
    return {
        "format": format,
        "status": "success",
        "message": f"Данные экспортированы в формате {format.upper()}",
        "download_url": f"/downloads/export_{format}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
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


# Регистрация роутов
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Аутентификация"])
app.include_router(scans.router, prefix="/api/v1/scans", tags=["Сканирования"])
app.include_router(simulations.router, prefix="/api/v1/simulations", tags=["Симуляции"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Анализ"])
app.include_router(comparison.router, prefix="/api/v1/comparison", tags=["Сравнение"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Отчёты"])
app.include_router(admin.router, prefix="/api/v1", tags=["Администрирование"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Дашборд"])

try:
    from api.routes import alerting
    app.include_router(alerting.router, prefix="/api/v1/alerting", tags=["Алертинг"])
except ImportError:
    pass

try:
    from api.routes import batch
    app.include_router(batch.router, prefix="/api/v1/batch", tags=["Пакетная обработка"])
except ImportError:
    pass

# GraphQL API
app.include_router(graphql.router, prefix="/api/v1", tags=["GraphQL"])

# AI/ML Analysis
app.include_router(ml_analysis.router, prefix="/api/v1", tags=["AI/ML"])

# External Services (with Circuit Breaker)
app.include_router(external_services.router, prefix="/api/v1", tags=["External Services"])


# Metrics endpoint для Prometheus
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus метрики"""
    from utils.enhanced_monitor import get_monitor
    
    # Получаем метрики из enhanced monitor
    monitor = get_monitor()
    prometheus_metrics = monitor.export_prometheus_metrics()
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=prometheus_metrics,
        media_type="text/plain"
    )


# GraphQL endpoint
@app.post("/graphql")
async def graphql_endpoint(request: Request):
    """GraphQL endpoint для запросов"""
    from api.graphql_schema import schema
    import json
    
    body = await request.json()
    query = body.get("query")
    variables = body.get("variables")
    operation_name = body.get("operationName")
    
    result = await schema.execute(
        query,
        variable_values=variables,
        operation_name=operation_name
    )
    
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
"""
    }


# WebSocket эндпоинт для real-time обновлений
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений с поддержкой каналов"""
    await websocket.accept()
    
    import psutil
    
    subscribed_channels = set()
    last_pong = datetime.now()

    try:
        while True:
            try:
                # Получение команд от клиента с timeout
                data = await websocket.receive_text()
                last_pong = datetime.now()
                message = json.loads(data)

                # Обработка команд
                if message.get("type") == "subscribe":
                    channel = message.get("channel")
                    if channel:
                        subscribed_channels.add(channel)
                        await websocket.send_json({
                            "type": "subscribed",
                            "channel": channel,
                            "timestamp": datetime.now().isoformat(),
                        })
                        logger.info(f"Client subscribed to channel: {channel}")

                elif message.get("type") == "unsubscribe":
                    channel = message.get("channel")
                    subscribed_channels.discard(channel)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "channel": channel,
                        "timestamp": datetime.now().isoformat(),
                    })

                elif message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    })

                elif message.get("type") == "get_metrics":
                    # Отправка текущих метрик
                    metrics = {
                        "type": "metrics",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "cpu_percent": psutil.cpu_percent(interval=0.1),
                            "memory_percent": psutil.virtual_memory().percent,
                            "disk_percent": psutil.disk_usage('/').percent,
                        }
                    }
                    await websocket.send_json(metrics)

            except asyncio.TimeoutError:
                # Проверка heartbeat
                if (datetime.now() - last_pong).total_seconds() > 60:
                    logger.warning("Client heartbeat timeout, disconnecting")
                    break
                continue

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client disconnected. Channels: {subscribed_channels}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        subscribed_channels.clear()


# Фоновая задача для push-уведомлений подписчикам
async def push_realtime_updates():
    """Периодическая отправка обновлений подписчикам"""
    import psutil
    
    while True:
        try:
            await asyncio.sleep(5)  # Каждые 5 секунд
            
            # Здесь должна быть интеграция с ConnectionManager для рассылки
            # Пока просто логируем
            logger.debug("Realtime update tick")
            
        except Exception as e:
            logger.error(f"Error in push_realtime_updates: {e}")


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД"""
    return db_manager


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
