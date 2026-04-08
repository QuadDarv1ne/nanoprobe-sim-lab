"""
JWT Secret Management - единый источник для JWT секретов
Безопасное хранение и инициализация JWT секретов
"""

import os
import secrets
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Пути к файлам
_jwt_secret_file = Path("data/.jwt_secret")
_admin_password_file = Path("data/.admin_password")
_user_password_file = Path("data/.user_password")


def _read_or_create_secret_file(file_path: Path, length: int = 32) -> str:
    """
    Читает секрет из файла или создаёт новый.
    
    Args:
        file_path: Путь к файлу секрета
        length: Длина генерируемого секрета
        
    Returns:
        str: Секрет
    """
    file_path = Path(file_path)
    
    if file_path.exists():
        secret = file_path.read_text().strip()
        if secret:
            return secret
    
    # Создаём новый секрет
    file_path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(length)
    file_path.write_text(secret)
    file_path.chmod(0o600)  # Только владелец может читать
    logger.info(f"Created new secret file: {file_path}")
    
    return secret


def get_jwt_secret() -> str:
    """
    Получает JWT секрет из ENV или файла.
    
    Приоритет:
    1. ENV переменная JWT_SECRET
    2. Файл data/.jwt_secret
    3. Генерация нового секрета и сохранение в файл
    
    Returns:
        str: JWT секрет
        
    Raises:
        RuntimeError: Если ENV=REQUIRE_SECURE_SECRETS и секрет не найден
    """
    # Проверяем ENV
    env_secret = os.getenv("JWT_SECRET")
    if env_secret:
        if env_secret == "REQUIRE_SECURE_SECRETS":
            # Строгий режим - только из файла/ENV
            if _jwt_secret_file.exists():
                return _jwt_secret_file.read_text().strip()
            raise RuntimeError(
                "JWT_SECRET not set и REQUIRE_SECURE_SECRETS=1. "
                "Установите JWT_SECRET в environment или удалите REQUIRE_SECURE_SECRETS."
            )
        return env_secret
    
    # Используем файл или генерируем
    return _read_or_create_secret_file(_jwt_secret_file)


def get_default_passwords() -> dict:
    """
    Получает пароли по умолчанию из ENV или файлов.
    
    Пароли читаются из:
    1. ENV переменных ADMIN_PASSWORD / USER_PASSWORD
    2. Файлов data/.admin_password / data/.user_password
    3. Генерируются случайно при первом запуске
    
    Returns:
        dict: {"admin": "password", "user": "password"}
    """
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        admin_password = _read_or_create_secret_file(_admin_password_file, 16)
    
    user_password = os.getenv("USER_PASSWORD")
    if not user_password:
        user_password = _read_or_create_secret_file(_user_password_file, 16)
    
    return {
        "admin": admin_password,
        "user": user_password
    }


def is_first_run() -> bool:
    """
    Проверяет, является ли это первым запуском.
    Первый запуск = файлы секретов только что созданы.
    
    Returns:
        bool: True если первый запуск
    """
    return (
        not os.getenv("JWT_SECRET") and
        not _jwt_secret_file.exists()
    )


def get_redis_connection_pool():
    """
    Получает Redis connection pool через RedisCache (singleton).
    Делегирует в RedisCache чтобы не дублировать конфигурацию.
    """
    from utils.caching.redis_cache import RedisCache
    if not hasattr(get_redis_connection_pool, '_cache'):
        get_redis_connection_pool._cache = RedisCache()
    return get_redis_connection_pool._cache


def get_redis_client():
    """
    Получает Redis клиент.
    Сначала пробует app state, затем создаёт собственный через RedisCache.
    """
    try:
        from api.state import get_redis
        instance = get_redis()
        if instance and instance.is_available():
            return instance.client
    except Exception:
        pass

    cache = get_redis_connection_pool()
    return cache.client if cache.is_available() else None
