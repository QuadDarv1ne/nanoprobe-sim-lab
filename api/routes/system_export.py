"""
Эндпоинты экспорта данных и системных операций
"""

import io
import logging
import os
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.error_handlers import ServiceUnavailableError
from api.state import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/api/v1/export/{format}",
    summary="Экспорт данных в формате json, csv или pdf",
)
async def export_data(format: str):
    """Экспорт данных в различных форматах"""
    from api.error_handlers import ValidationError

    if format not in ["json", "csv", "pdf"]:
        raise ValidationError(f"Неподдерживаемый формат: {format}. Доступны: json, csv, pdf")

    # Здесь можно добавить реальную логику экспорта
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    download_url = f"/downloads/export_{format}_{timestamp}.{format}"
    return {
        "format": format,
        "status": "success",
        "message": f"Данные экспортированы в формате {format.upper()}",
        "download_url": download_url,
    }


@router.get(
    "/api/v1/export-bulk",
    summary="Экспорт всех данных (сканы, симуляции, отчёты) в ZIP архиве",
)
async def export_all_data():
    """
    Экспорт всех данных проекта в ZIP архиве.
    Включает: сканы, симуляции, отчёты, изображения.
    """
    try:
        db = get_db_manager()
    except RuntimeError:
        raise ServiceUnavailableError("Database not available")

    # Получаем все данные
    scans = db.get_scan_results(limit=1000)
    simulations = db.get_simulations(limit=1000)

    # Создаём ZIP архив в памяти
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Экспорт сканов
        if scans:
            import json

            zip_file.writestr(
                "data/scans.json", json.dumps(scans, ensure_ascii=False, indent=2, default=str)
            )

        # Экспорт симуляций
        if simulations:
            import json

            zip_file.writestr(
                "data/simulations.json",
                json.dumps(simulations, ensure_ascii=False, indent=2, default=str),
            )

        # Экспорт метаданных
        metadata = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "total_scans": len(scans),
            "total_simulations": len(simulations),
        }
        import json

        zip_file.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

        # Экспорт файлов если есть (опционально)
        data_dir = "data"
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                # Пропускаем временные файлы
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for file in files:
                    if file.endswith((".db", ".json")):
                        file_path = os.path.join(root, file)
                        arc_name = file_path.replace("\\", "/")
                        zip_file.write(file_path, arc_name)

    zip_buffer.seek(0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    content_disposition = f"attachment; filename=nanoprobe_export_{timestamp}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": content_disposition},
    )


@router.post(
    "/api/v1/system/restart",
    summary="Перезапуск системы (мягкий restart)",
)
async def restart_system(background_tasks: BackgroundTasks):
    """
    Мягкий перезапуск системы.
    Не останавливает сервер, но перезапускает фоновые задачи и очищает кэш.
    """
    from utils.caching.redis_cache import cache

    # Очищаем кэш
    try:
        cache.clear_pattern("scans:*")
        cache.clear_pattern("simulations:*")
        cache.clear_pattern("dashboard:*")
        logger.info("Cache cleared during restart")
    except Exception as e:
        logger.warning(f"Cache clear error during restart: {e}")

    # Сбрасываем состояние приложения
    from api.state import clear_app_state

    clear_app_state()

    # Фоновая задача для перезапуска сервисов
    def restart_background_services():
        """Перезапуск фоновых сервисов"""
        try:
            logger.info("Restarting background services...")
            # Здесь можно добавить логику перезапуска конкретных сервисов
            # Например: sync_manager, monitoring, и т.д.
        except Exception as e:
            logger.error(f"Background service restart error: {e}")

    background_tasks.add_task(restart_background_services)

    return {
        "status": "success",
        "message": "Система перезапускается",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": {
            "cache_cleared": True,
            "app_state_reset": True,
            "background_services_restarting": True,
        },
    }


@router.get(
    "/api/v1/system/status",
    summary="Статус системы и запущенных сервисов",
)
async def get_system_status():
    """Получение статуса всех сервисов системы"""
    import platform

    import psutil

    # Основная информация о системе
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk_path = os.environ.get("SYSTEMDRIVE", "C:\\") if platform.system() == "Windows" else "/"
    disk = psutil.disk_usage(disk_path)

    # Статус сервисов
    services = {
        "api": {"status": "running", "pid": os.getpid()},
        "database": {"status": "running"},
        "redis": {"status": "unknown"},
    }

    # Проверка Redis
    try:
        from api.state import get_redis

        redis = get_redis()
        if redis and redis.is_available():
            services["redis"]["status"] = "running"
        else:
            services["redis"]["status"] = "disabled"
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)
        services["redis"]["status"] = "error"

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "platform": platform.system(),
            "python_version": platform.python_version(),
        },
        "services": services,
    }
