"""
Модуль регистрации роутов для FastAPI приложения
Вынесен из main.py для улучшения структуры кода
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI):
    """Регистрация всех API роутов"""
    
    # ============================================
    # Основные роуты (обязательные)
    # ============================================
    from api.routes import system_export  # Новые эндпоинты экспорта и системных операций
    from api.routes import (
        admin,
        analysis,
        auth,
        comparison,
        external_services,
        graphql,
        ml_analysis,
        monitoring,
        nasa,
        reports,
        scans,
        simulations,
        weather,
    )

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Аутентификация"])
    app.include_router(scans.router, prefix="/api/v1/scans", tags=["Сканирования"])
    app.include_router(simulations.router, prefix="/api/v1/simulations", tags=["Симуляции"])
    app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Анализ"])
    app.include_router(comparison.router, prefix="/api/v1/comparison", tags=["Сравнение"])
    app.include_router(reports.router, prefix="/api/v1/reports", tags=["Отчёты"])
    app.include_router(admin.router, prefix="/api/v1", tags=["Администрирование"])
    
    # Экспорт и системные операции (без prefix — роуты сами определяют пути)
    app.include_router(system_export.router, tags=["Экспорт и система"])
    
    # ============================================
    # Health endpoints (алиасы для фронтенда)
    # ============================================
    @app.get("/api/v1/health/database", tags=["Health"])
    async def health_database():
        """Проверка здоровья БД (алиас для фронтенда)"""
        from api.state import get_db_manager
        try:
            db = get_db_manager()
            with db.get_connection() as conn:
                conn.execute("SELECT 1")
            db_path = Path(db.db_path)
            return {
                "status": "healthy",
                "size_bytes": db_path.stat().st_size if db_path.exists() else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post("/api/v1/database/backup", tags=["Health"])
    async def database_backup_alias():
        """Бэкап БД (алиас для фронтенда, без авторизации)"""
        from api.state import get_db_manager
        db = get_db_manager()
        backup_dir = Path("data/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"nanoprobe_{ts}.db"
        try:
            shutil.copy2(str(db.db_path), str(backup_path))
            return {
                "status": "success", 
                "backup_path": str(backup_path), 
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            from api.error_handlers import ValidationError
            raise ValidationError(f"Ошибка бэкапа: {str(e)}")
    
    # ============================================
    # Опциональные роуты (с проверкой импорта)
    # ============================================
    
    # Dashboard API
    try:
        from api.routes import dashboard
        app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Дашборд"])
        logger.info("Dashboard routes registered")
    except ImportError as e:
        logger.warning(f"Dashboard routes disabled: {e}")

    # Alerting API
    try:
        from api.routes import alerting
        app.include_router(alerting.router, prefix="/api/v1/alerting", tags=["Алертинг"])
        logger.info("Alerting routes registered")
    except ImportError as e:
        logger.warning(f"Alerting routes disabled: {e}")

    # Batch processing API
    try:
        from api.routes import batch
        app.include_router(batch.router, prefix="/api/v1/batch", tags=["Пакетная обработка"])
        logger.info("Batch routes registered")
    except ImportError as e:
        logger.warning(f"Batch routes disabled: {e}")
    
    # GraphQL API
    app.include_router(graphql.router, prefix="/api/v1", tags=["GraphQL"])
    logger.info("GraphQL routes registered")

    # AI/ML Analysis
    app.include_router(ml_analysis.router, prefix="/api/v1", tags=["AI/ML"])
    logger.info("AI/ML routes registered")

    # External Services (with Circuit Breaker)
    app.include_router(external_services.router, prefix="/api/v1", tags=["External Services"])
    logger.info("External Services routes registered")

    # NASA API Integration
    app.include_router(nasa.router, prefix="/api/v1", tags=["NASA API"])
    logger.info("NASA API routes registered")

    # Weather API Integration (Open-Meteo, free, no API key needed)
    app.include_router(weather.router, prefix="/api/v1", tags=["Weather"])
    logger.info("Weather routes registered")

    # Performance Monitoring
    app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])
    logger.info("Monitoring routes registered")

    # SSTV Ground Station API
    try:
        from api.routes import sstv
        app.include_router(sstv.router, prefix="/api/v1/sstv", tags=["SSTV Ground Station"])
        logger.info("SSTV Ground Station routes registered")
        
        # Advanced SSTV endpoints (WebSocket, spectrum, real-time)
        from api.routes import sstv_advanced
        app.include_router(sstv_advanced.router, tags=["SSTV Advanced"])
        logger.info("SSTV Advanced routes registered")
    except ImportError as e:
        logger.warning(f"SSTV Ground Station routes disabled: {e}")
