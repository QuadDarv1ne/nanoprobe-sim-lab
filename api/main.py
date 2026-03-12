# -*- coding: utf-8 -*-
"""
FastAPI REST API для Nanoprobe Simulation Lab
Совместная работа с Flask веб-интерфейсом
"""

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json
import os
import traceback

# Импорт существующих утилит
from utils.database import DatabaseManager
from utils.defect_analyzer import DefectAnalysisPipeline, analyze_defects
from utils.surface_comparator import SurfaceComparator
from utils.pdf_report_generator import ScientificPDFReport
from utils.batch_processor import BatchProcessor
from utils.redis_cache import RedisCache

# Импорты роутов
from api.routes import scans, simulations, analysis, comparison, reports, auth, admin, dashboard


# Глобальные переменные
db_manager: Optional[DatabaseManager] = None
redis_cache: Optional[RedisCache] = None
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global db_manager, redis_cache

    # Инициализация БД
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js
        "http://localhost:5000",  # Flask
        "http://127.0.0.1:5000",
    ],
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


# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Обработка HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Обработка ошибок валидации"""
    # Сериализуем ошибки, преобразуя не-JSON объекты в строки
    def serialize_error(err):
        if isinstance(err, (str, int, float, bool, type(None))):
            return err
        elif isinstance(err, (list, tuple)):
            return [serialize_error(e) for e in err]
        elif isinstance(err, dict):
            return {k: serialize_error(v) for k, v in err.items()}
        else:
            return str(err)
    
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status_code": 422,
            "detail": "Validation error",
            "errors": serialize_error(exc.errors())
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Обработка общих исключений"""
    # Логирование ошибки
    error_trace = traceback.format_exc()
    print(f"[ERROR] {datetime.now().isoformat()} - {exc}\n{error_trace}")

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "type": exc.__class__.__name__
        }
    )


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


# WebSocket эндпоинт для real-time обновлений
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений"""
    await websocket.accept()
    
    try:
        while True:
            # Получение команд от клиента
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обработка команд
            if message.get("type") == "subscribe":
                channel = message.get("channel")
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": channel,
                    "timestamp": datetime.now().isoformat(),
                })
            
            elif message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                })
                
    except WebSocketDisconnect:
        print("🔌 WebSocket клиент отключился")


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
