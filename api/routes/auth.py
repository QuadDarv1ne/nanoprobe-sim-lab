"""
API роуты для аутентификации
JWT токен, логин, регистрация с refresh token rotation
Redis integration для персистентного хранения токенов
"""

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBearer
from datetime import datetime, timedelta
from typing import Set
import jwt
import os
import secrets
import logging

from passlib.context import CryptContext

from api.schemas import (
    LoginRequest,
    LoginResponse,
    Token,
    ErrorResponse,
    RefreshTokenRequest,
)
from api.dependencies import rate_limit, get_current_user
from api.error_handlers import AuthenticationError, ValidationError
from api.rate_limiter import auth_limit

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60
JWT_REFRESH_EXPIRATION_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

USERS_DB = {
    "admin": {
        "id": 1,
        "username": "admin",
        "password_hash": pwd_context.hash("Admin123!"),
        "role": "admin",
        "created_at": "2026-03-11T00:00:00",
        "last_login": None,
    },
    "user": {
        "id": 2,
        "username": "user",
        "password_hash": pwd_context.hash("User123!"),
        "role": "user",
        "created_at": "2026-03-11T00:00:00",
        "last_login": None,
    },
}


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if len(password) > 128:
        return False, "Password must not exceed 128 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain an uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain a digit"
    if not any(c.islower() for c in password):
        return False, "Password must contain a lowercase letter"
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain a special character"
    return True, ""


def hash_password(password: str) -> str:
    """Хеширование пароля с bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создание access токена"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def _get_redis_client():
    """Получение Redis клиента"""
    try:
        import redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        redis_client.ping()  # Проверка подключения
        return redis_client
    except Exception as e:
        logger.warning(f"Redis not available, using in-memory storage: {e}")
        return None


# In-memory хранилище (fallback если Redis недоступен)
_in_memory_tokens: Set[str] = set()


def _store_refresh_token(jti: str, username: str):
    """Сохранение refresh токена в Redis"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            redis_client.setex(key, JWT_REFRESH_EXPIRATION_DAYS * 86400, username)
        except Exception as e:
            logger.error(f"Failed to store token in Redis: {e}")
            _in_memory_tokens.add(jti)
    else:
        _in_memory_tokens.add(jti)


def _is_token_valid(jti: str) -> bool:
    """Проверка валидности refresh токена"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            return redis_client.exists(key)
        except Exception as e:
            logger.error(f"Failed to check token in Redis: {e}")
            return jti in _in_memory_tokens
    return jti in _in_memory_tokens


def _revoke_refresh_token(jti: str):
    """Отмена refresh токена (rotation)"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to revoke token in Redis: {e}")
            _in_memory_tokens.discard(jti)
    else:
        _in_memory_tokens.discard(jti)


def create_refresh_token(data: dict) -> str:
    """Создание refresh токена с уникальным jti"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_EXPIRATION_DAYS)
    jti = secrets.token_urlsafe(16)  # Unique token ID
    to_encode.update({"exp": expire, "type": "refresh", "jti": jti})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

    # Сохранение jti в Redis
    _store_refresh_token(jti, data.get("sub", "unknown"))

    return encoded_jwt


def revoke_refresh_token(jti: str):
    """Отмена refresh токена (rotation)"""
    _revoke_refresh_token(jti)


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Вход в систему",
    description="Получение JWT токена для доступа к API",
    responses={
        200: {"description": "Успешный вход"},
        401: {"model": ErrorResponse, "description": "Неверный логин или пароль"},
        429: {"model": ErrorResponse, "description": "Слишком много запросов"},
    },
)
@rate_limit(max_requests=5, window_seconds=60)
@auth_limit(max_requests=10, window=60)
async def login(request: Request, login_data: LoginRequest):
    """Вход в систему"""
    user = USERS_DB.get(login_data.username)

    if not user or not verify_password(login_data.password, user["password_hash"]):
        raise AuthenticationError("Неверное имя пользователя или пароль")

    # Обновление last_login
    user["last_login"] = datetime.now().isoformat()

    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]}
    )
    refresh_token = create_refresh_token(
        data={"sub": user["username"], "user_id": user["id"]}
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_MINUTES * 60,
        user={
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        },
    )


@router.post(
    "/refresh",
    response_model=Token,
    summary="Обновить токен",
    description="Обновление access токена с помощью refresh токена (rotation)",
)
async def refresh_access_token(request: RefreshTokenRequest):
    """Обновление токена с refresh token rotation"""
    try:
        payload = jwt.decode(request.refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Проверка типа токена
        if payload.get("type") != "refresh":
            raise AuthenticationError("Неверный тип токена")

        # Проверка jti (rotation check) через Redis
        jti = payload.get("jti")
        if not _is_token_valid(jti):
            raise AuthenticationError("Refresh токен был отозван")

        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Неверный refresh токен")

        user = USERS_DB.get(username)
        if not user:
            raise AuthenticationError("Пользователь не найден")

        # Ревокация старого токена (rotation)
        revoke_refresh_token(jti)

        # Создание новой пары токенов
        new_access_token = create_access_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )
        new_refresh_token = create_refresh_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )

        logger.info(f"Token refreshed for user: {username}")

        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=JWT_EXPIRATION_MINUTES * 60,
        )

    except jwt.PyJWTError:
        raise AuthenticationError("Неверный refresh токен")


@router.post(
    "/2fa/setup",
    summary="Настроить 2FA",
    description="Инициация настройки двухфакторной аутентификации",
)
async def setup_2fa(current_user: dict = Depends(get_current_user)):
    """Настройка 2FA"""
    from utils.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    user_email = f"{username}@nanoprobe.local"

    two_factor = get_2fa_manager()
    secret, provisioning_uri = two_factor.setup_2fa(username, user_email)

    return {
        "secret": secret,
        "provisioning_uri": provisioning_uri,
        "message": "Отсканируйте QR код в Google Authenticator"
    }


@router.post(
    "/2fa/verify",
    summary="Верифицировать 2FA",
    description="Подтверждение настройки 2FA OTP кодом",
)
async def verify_2fa_setup(
    otp_code: str,
    current_user: dict = Depends(get_current_user)
):
    """Верификация 2FA"""
    from utils.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    if two_factor.verify_2fa_setup(username, otp_code):
        return {"success": True, "message": "2FA успешно включена"}

    raise ValidationError("Неверный OTP код")


@router.post(
    "/2fa/verify-login",
    summary="2FA верификация при входе",
    description="Проверка 2FA кода после успешного логина",
)
async def verify_2fa_login(
    otp_code: str,
    username: str,
    password: str
):
    """2FA при входе"""
    from utils.two_factor_auth import get_2fa_manager

    # Сначала проверяем логин/пароль
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise AuthenticationError("Неверное имя пользователя или пароль")

    # Проверяем 2FA если включена
    two_factor = get_2fa_manager()

    if two_factor.is_2fa_enabled(username):
        # Требуется 2FA верификация
        if not two_factor.verify_2fa(username, otp_code):
            # Пробуем резервный код
            if not two_factor.verify_backup_code(username, otp_code):
                raise AuthenticationError("Неверный 2FA код")

    # Генерация токенов
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]}
    )
    refresh_token = create_refresh_token(
        data={"sub": user["username"], "user_id": user["id"]}
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_MINUTES * 60,
        user={
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "2fa_enabled": two_factor.is_2fa_enabled(username)
        }
    )


@router.get(
    "/2fa/status",
    summary="Статус 2FA",
    description="Проверка статуса 2FA для текущего пользователя",
)
async def get_2fa_status(current_user: dict = Depends(get_current_user)):
    """Статус 2FA"""
    from utils.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    return {
        "enabled": two_factor.is_2fa_enabled(username),
        "username": username
    }


@router.post(
    "/2fa/disable",
    summary="Отключить 2FA",
    description="Отключение двухфакторной аутентификации",
)
async def disable_2fa(
    otp_code: str,
    current_user: dict = Depends(get_current_user)
):
    """Отключение 2FA"""
    from utils.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    if two_factor.disable_2fa(username, otp_code):
        return {"success": True, "message": "2FA отключена"}

    raise ValidationError("Неверный OTP код")


@router.post(
    "/2fa/backup-codes",
    summary="Генерировать резервные коды",
    description="Генерация резервных кодов для 2FA",
)
async def generate_backup_codes(current_user: dict = Depends(get_current_user)):
    """Генерация резервных кодов"""
    from utils.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    codes = two_factor.generate_backup_codes(username, count=10)

    return {
        "backup_codes": codes,
        "message": "Сохраните эти коды в безопасном месте!"
    }


@router.get(
    "/me",
    summary="Текущий пользователь",
    description="Получение информации о текущем пользователе",
)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Получение информации о текущем пользователе"""
    from utils.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"],
        "created_at": current_user["created_at"],
        "2fa_enabled": two_factor.is_2fa_enabled(username)
    }


@router.post(
    "/logout",
    summary="Выход из системы",
    description="Выход из системы с ревокацией refresh токена",
)
async def logout(refresh_token: Optional[str] = None):
    """Выход из системы с ревокацией refresh токена"""
    if refresh_token:
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            jti = payload.get("jti")
            if jti:
                revoke_refresh_token(jti)
        except jwt.PyJWTError:
            pass

    return {"message": "Успешный выход. Удалите токены на клиенте."}


@router.get(
    "/rate-limit-status",
    summary="Статус rate limiting",
    description="Получение статуса rate limiting для текущего IP",
)
async def get_rate_limit_status(request: Request):
    """Получение статуса rate limiting"""
    from utils.rate_limiter import limiter

    client_ip = request.client.host
    login_key = f"login:{client_ip}"

    status = limiter.get_status(login_key, max_requests=5, window_seconds=300)
    return {
        "ip": client_ip,
        "endpoint": "login",
        **status,
    }
