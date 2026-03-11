# -*- coding: utf-8 -*-
"""
FastAPI REST API для Nanoprobe Simulation Lab
Совместная работа с Flask веб-интерфейсом
"""

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json
import os

# Импорт существующих утилит
from utils.database import DatabaseManager
from utils.defect_analyzer import DefectAnalysisPipeline, analyze_defects
from utils.surface_comparator import SurfaceComparator
from utils.pdf_report_generator import ScientificPDFReport
from utils.batch_processor import BatchProcessor
from utils.redis_cache import RedisCache

# Импорты роутов
from api.routes import scans, simulations, analysis, comparison, reports, auth, admin


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
    print("✅ База данных инициализирована")

    # Инициализация Redis кэша
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_cache = RedisCache(host=redis_host, port=redis_port)
    
    if redis_cache.is_available():
        print(f"✅ Redis кэш подключён: {redis_host}:{redis_port}")
    else:
        print("⚠️  Redis кэш недоступен (работаем без кэширования)")

    yield

    # Очистка при остановке
    print("👋 Приложение остановлено")


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


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
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
