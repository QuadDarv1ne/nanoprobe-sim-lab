"""
Панель администратора для Nanoprobe Sim Lab
Управление пользователями, системные настройки, мониторинг
"""

from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import os
import logging
from pathlib import Path

from api.dependencies import get_current_user, require_admin
from api.dependencies import get_redis_cache
from api.error_handlers import AuthorizationError, NotFoundError, ValidationError
from api.state import get_system_disk_usage
from api.state import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Администрирование"])


# ==================== Системная информация ====================

@router.get(
    "/cache/redis",
    summary="Статус Redis кэша",
    description="Получить статистику Redis кэша",
)
async def get_redis_cache_status(
    current_user: dict = Depends(get_current_user),
    redis_cache=Depends(get_redis_cache),
):
    """Статус Redis кэша"""
    require_admin(current_user)

    if redis_cache is None:
        return {"available": False, "message": "Redis не инициализирован"}

    return redis_cache.get_stats()


# ==================== Система ====================

@router.get(
    "/system/info",
    summary="Системная информация",
    description="Получить информацию о системе",
)
async def get_system_info(current_user: dict = Depends(get_current_user)):
    """Системная информация"""
    require_admin(current_user)

    return {
        "platform": os.name,
        "python_version": os.sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "disk_total": get_system_disk_usage().total,
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
    }


@router.get(
    "/system/resources",
    summary="Использование ресурсов",
    description="Мониторинг использования CPU, памяти, диска",
)
async def get_system_resources(current_user: dict = Depends(get_current_user)):
    """Использование ресурсов"""
    require_admin(current_user)

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)

    # Память
    memory = psutil.virtual_memory()

    # Диск
    disk = get_system_disk_usage()

    # Сеть
    net_io = psutil.net_io_counters()

    return {
        "cpu": {
            "percent": cpu_percent,
            "per_core": cpu_per_core,
            "count": psutil.cpu_count(),
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        },
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.get(
    "/system/processes",
    summary="Список процессов",
    description="Получить список процессов приложения",
)
async def get_system_processes(current_user: dict = Depends(get_current_user)):
    """Список процессов"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    current_pid = os.getpid()
    process = psutil.Process(current_pid)

    return {
        "pid": current_pid,
        "name": process.name(),
        "status": process.status(),
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "memory_info": process.memory_info()._asdict(),
        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
        "threads": process.num_threads(),
        "open_files": [f.path for f in process.open_files()],
    }


# ==================== Управление логами ====================

@router.get(
    "/logs/list",
    summary="Список логов",
    description="Получить список файлов логов",
)
async def list_logs(current_user: dict = Depends(get_current_user)):
    """Список файлов логов"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    logs_dir = Path("logs")
    if not logs_dir.exists():
        return {"files": []}

    files = []
    for f in logs_dir.glob("*.log"):
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })

    return {"files": sorted(files, key=lambda x: x["modified"], reverse=True)}


@router.get(
    "/logs/view",
    summary="Просмотр лога",
    description="Просмотр содержимого файла лога",
)
async def view_log(
    filename: str,
    lines: int = 100,
    current_user: dict = Depends(get_current_user),
):
    """Просмотр лога"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    log_path = Path("logs") / filename

    # Проверка на directory traversal
    if not log_path.is_relative_to(Path("logs")):
        raise ValidationError("Неверное имя файла")

    if not log_path.exists():
        raise NotFoundError(f"Файл {filename} не найден", resource_type="log_file")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        logger.debug(f"Read {len(last_lines)} lines from log file: {filename}")
        return {
            "filename": filename,
            "lines": len(all_lines),
            "content": "".join(last_lines),
        }
    except Exception as e:
        logger.error(f"Error reading log file {filename}: {e}")
        raise ValidationError(f"Ошибка чтения лога: {str(e)}")


@router.post(
    "/logs/clear",
    summary="Очистить логи",
    description="Очистить файлы логов",
)
async def clear_logs(
    filename: str = None,
    current_user: dict = Depends(get_current_user),
):
    """Очистка логов"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    logs_dir = Path("logs")

    if filename:
        # Очистка конкретного файла
        log_path = logs_dir / filename
        if not log_path.is_relative_to(logs_dir):
            raise ValidationError("Неверное имя файла")

        if log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("")
            return {"message": f"Лог {filename} очищен"}
        else:
            raise NotFoundError(f"Файл {filename} не найден", resource_type="log_file")
    else:
        # Очистка всех логов
        for log_file in logs_dir.glob("*.log"):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("")

        return {"message": "Все логи очищены"}


# ==================== Управление базой данных ====================

@router.get(
    "/database/health",
    summary="Здоровье БД",
    description="Проверка состояния базы данных",
)
async def get_database_health(current_user: dict = Depends(get_current_user)):
    """Проверка здоровья БД"""
    require_admin(current_user)

    db = get_db_manager()
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        db_path = Path(db.db_path)
        return {
            "status": "healthy",
            "path": str(db_path),
            "size_bytes": db_path.stat().st_size if db_path.exists() else 0,
            "pool_stats": db.get_pool_stats(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}


@router.post(
    "/database/backup",
    summary="Создать бэкап БД",
    description="Создать резервную копию базы данных",
)
async def backup_database(current_user: dict = Depends(get_current_user)):
    """Создание бэкапа БД"""
    require_admin(current_user)

    db = get_db_manager()
    backup_dir = Path("data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"nanoprobe_{timestamp}.db"

    try:
        import shutil
        shutil.copy2(str(db.db_path), str(backup_path))
        size = backup_path.stat().st_size
        logger.info(f"Database backup created: {backup_path}")
        return {
            "status": "success",
            "backup_path": str(backup_path),
            "size_bytes": size,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise ValidationError(f"Ошибка создания бэкапа: {str(e)}")

@router.get(
    "/database/stats",
    summary="Статистика БД",
    description="Получить статистику базы данных",
)
async def get_database_stats(current_user: dict = Depends(get_current_user)):
    """Статистика БД"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    db = get_db_manager()
    stats = db.get_statistics()

    # Размер файла БД
    db_path = Path(db.db_path)
    db_size = db_path.stat().st_size if db_path.exists() else 0

    return {
        **stats,
        "database_size_bytes": db_size,
        "database_path": str(db_path),
    }


@router.post(
    "/database/vacuum",
    summary="Оптимизировать БД",
    description="Выполнить VACUUM и ANALYZE",
)
async def vacuum_database(current_user: dict = Depends(get_current_user)):
    """Оптимизация БД"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    db = get_db_manager()
    try:
        with db.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")

        logger.info("Database VACUUM and ANALYZE completed")
        return {
            "message": "База данных оптимизирована",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        raise ValidationError(f"Ошибка оптимизации: {str(e)}")


@router.get(
    "/database/tables",
    summary="Список таблиц",
    description="Получить список таблиц БД",
)
async def get_database_tables(current_user: dict = Depends(get_current_user)):
    """Список таблиц"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    db = get_db_manager()
    tables = []
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

    return {"tables": tables}


# ==================== Управление пользователями ====================

@router.get(
    "/users/list",
    summary="Список пользователей",
    description="Получить список всех пользователей",
)
async def list_users(current_user: dict = Depends(get_current_user)):
    """Список пользователей"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    from api.routes.auth import USERS_DB

    users = []
    for username, user_data in USERS_DB.items():
        users.append({
            "id": user_data["id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "created_at": user_data["created_at"],
        })

    return {"users": users, "total": len(users)}


@router.post(
    "/users/create",
    summary="Создать пользователя",
    description="Создать нового пользователя",
)
async def create_user(
    username: str,
    password: str,
    role: str = "user",
    current_user: dict = Depends(get_current_user),
):
    """Создание пользователя"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    from api.routes.auth import USERS_DB, hash_password

    if username in USERS_DB:
        raise ValidationError("Пользователь уже существует")

    new_id = max(u["id"] for u in USERS_DB.values()) + 1 if USERS_DB else 1

    USERS_DB[username] = {
        "id": new_id,
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
    }

    return {
        "message": f"Пользователь {username} создан",
        "user": {
            "id": new_id,
            "username": username,
            "role": role,
        },
    }


@router.delete(
    "/users/delete/{username}",
    summary="Удалить пользователя",
    description="Удалить пользователя",
)
async def delete_user(
    username: str,
    current_user: dict = Depends(get_current_user),
):
    """Удаление пользователя"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    from api.routes.auth import USERS_DB

    if username not in USERS_DB:
        raise NotFoundError(f"Пользователь {username} не найден", resource_type="user")

    if username == current_user.get("username"):
        raise ValidationError("Нельзя удалить самого себя")

    del USERS_DB[username]

    return {"message": f"Пользователь {username} удалён"}


# ==================== Кэш ====================

@router.get(
    "/cache/stats",
    summary="Статистика кэша",
    description="Получить статистику кэша",
)
async def get_cache_stats(current_user: dict = Depends(get_current_user)):
    """Статистика кэша"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    cache_dirs = ["cache", "temp", "__pycache__"]
    stats = {}

    for dir_name in cache_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            file_count = len(list(dir_path.rglob("*")))
            stats[dir_name] = {
                "size_bytes": total_size,
                "file_count": file_count,
            }
        else:
            stats[dir_name] = {"size_bytes": 0, "file_count": 0}

    return stats


@router.post(
    "/cache/clear",
    summary="Очистить кэш",
    description="Очистить кэш директории",
)
async def clear_cache(
    target: str = "all",
    current_user: dict = Depends(get_current_user),
):
    """Очистка кэша"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    from utils.cache_manager import CacheManager

    cache_mgr = CacheManager()
    cleared = cache_mgr.auto_cleanup()

    return {
        "message": "Кэш очищен",
        "cleared": cleared,
        "timestamp": datetime.now().isoformat(),
    }


# ==================== Задачи ====================

@router.get(
    "/tasks/list",
    summary="Список задач",
    description="Получить список фоновых задач",
)
async def list_tasks(current_user: dict = Depends(get_current_user)):
    """Список задач"""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")

    # Заглушка для будущей интеграции с Celery
    return {
        "tasks": [],
        "message": "Интеграция с Celery в разработке",
    }
