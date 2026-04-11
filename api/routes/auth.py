"""
API роуты для аутентификации
JWT токен, логин, регистрация с refresh token rotation
Redis integration для персистентного хранения токенов
Argon2 password hashing + Audit logging
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBearer
from passlib.context import CryptContext

from api.dependencies import get_current_user, rate_limit
from api.error_handlers import AuthenticationError, ValidationError
from api.rate_limiter import auth_limit
from api.schemas import ErrorResponse, LoginRequest, LoginResponse, RefreshTokenRequest, Token
from api.security.jwt_config import get_default_passwords, get_jwt_secret

logger = logging.getLogger(__name__)

# Audit logger для security событий
audit_logger = logging.getLogger("audit.security")

router = APIRouter()
security = HTTPBearer()

# JWT Secret из централизованного источника
JWT_SECRET = get_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
JWT_REFRESH_EXPIRATION_DAYS = int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7"))

# Argon2 + bcrypt fallback для совместимости
# Argon2 - победитель Password Hashing Competition, более безопасный чем bcrypt
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    default="argon2",
    deprecated="auto",
    # Argon2 параметры (рекомендованные OWASP)
    argon2__memory_cost=65536,  # 64 MB
    argon2__time_cost=3,  # 3 итерации
    argon2__parallelism=4,  # 4 параллельных потока
    argon2__type="id",  # Argon2id (гибрид Data-dependent + Data-independent)
)


# Инициализация пользователей с безопасными паролями из ENV/файлов
def _initialize_users_db():
    """
    Инициализирует базу пользователей.
    Хеши кэшируются в data/.password_hashes.json чтобы не пересчитывать
    Argon2 при каждом старте. Пользователи хранятся в SQLite — last_login
    персистентен между рестартами.
    """
    default_passwords = get_default_passwords()
    hash_cache_file = Path("data/.password_hashes.json")

    cached_hashes = {}
    if hash_cache_file.exists():
        try:
            import json as _json

            cached_hashes = _json.loads(hash_cache_file.read_text())
        except Exception:
            cached_hashes = {}

    def _get_or_create_hash(username: str, password: str) -> str:
        cache_key = f"{username}:{hashlib.sha256(password.encode()).hexdigest()[:16]}"
        if cache_key in cached_hashes:
            return cached_hashes[cache_key]
        new_hash = pwd_context.hash(password)
        cached_hashes[cache_key] = new_hash
        return new_hash

    import json as _json

    admin_hash = _get_or_create_hash("admin", default_passwords["admin"])
    user_hash = _get_or_create_hash("user", default_passwords["user"])

    # Сохраняем кэш хешей
    try:
        hash_cache_file.parent.mkdir(parents=True, exist_ok=True)
        hash_cache_file.write_text(_json.dumps(cached_hashes))
        hash_cache_file.chmod(0o600)
    except Exception as e:
        logger.warning(f"Could not save password hash cache: {e}")

    # Синхронизируем с SQLite (upsert) — создаём если нет, обновляем хеш если изменился
    try:
        from utils.database import get_database

        db = get_database()
        db.upsert_user("admin", admin_hash, "admin")
        db.upsert_user("user", user_hash, "user")
        logger.info("Users synced to database")
    except Exception as e:
        logger.warning(f"Could not sync users to database: {e}")

    # Возвращаем in-memory dict для совместимости с текущим кодом
    # last_login подгружается из БД
    def _load_from_db(username: str, uid: int, role: str, ph: str) -> dict:
        try:
            from utils.database import get_database

            row = get_database().get_user(username)
            last_login = row.get("last_login") if row else None
        except Exception:
            last_login = None
        return {
            "id": uid,
            "username": username,
            "password_hash": ph,
            "role": role,
            "created_at": "2026-03-11T00:00:00",
            "last_login": last_login,
        }

    return {
        "admin": _load_from_db("admin", 1, "admin", admin_hash),
        "user": _load_from_db("user", 2, "user", user_hash),
    }


_USERS_DB: Optional[dict] = None


def _get_users_db() -> dict:
    """Lazy-инициализация USERS_DB — вызывается при первом обращении, не при импорте."""
    global _USERS_DB
    if _USERS_DB is None:
        _USERS_DB = _initialize_users_db()
    return _USERS_DB


# Совместимость: USERS_DB как property-like proxy через __getattr__ невозможна на уровне модуля,
# поэтому оставляем прямой доступ через get_users_db() и инициализируем лениво при первом login.
USERS_DB: dict = {}  # заполняется при первом вызове get_users_db()


class AuditEventType(str, Enum):
    """Типы audit событий"""

    LOGIN_SUCCESS = "login.success"
    LOGIN_FAILURE = "login.failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token.refresh"
    TOKEN_REVOKED = "token.revoked"
    PASSWORD_CHANGED = "password.changed"
    _2FA_ENABLED = "2fa.enabled"
    _2FA_DISABLED = "2fa.disabled"
    _2FA_VERIFICATION_FAILED = "2fa.verification_failed"


def log_audit_event(event_type: AuditEventType, username: str, request: Request, **extra):
    """
    Логирование audit события безопасности

    Формат:
    {
        "timestamp": "2026-03-15T10:30:00Z",
        "event_type": "login.success",
        "username": "admin",
        "ip": "192.168.1.1",
        "user_agent": "Mozilla/5.0...",
        "extra": {...}
    }
    """
    audit_event = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_type": event_type.value,
        "username": username,
        "ip": request.client.host if request else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown") if request else "unknown",
        **extra,
    }

    # Логирование в JSON формате для удобного парсинга
    import json

    audit_logger.info(json.dumps(audit_event, ensure_ascii=False))

    return audit_event


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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str, allow_expired: bool = False) -> dict:
    """
    Декодирование JWT токена.

    Args:
        token: JWT токен
        allow_expired: Если True — декодирует даже истёкший токен (только для logout/rotation)
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        if allow_expired:
            payload = jwt.decode(
                token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False}
            )
            return payload
        return {"error": "token_expired"}
    except jwt.InvalidTokenError:
        return {"error": "invalid_token"}


def _get_redis_client():
    """Получение Redis клиента из connection pool"""
    try:
        from api.security.jwt_config import get_redis_client

        return get_redis_client()
    except Exception as e:
        logger.warning(f"Redis not available, using in-memory storage: {e}")
        return None


# In-memory хранилище (fallback если Redis недоступен)
# Формат: {jti: timestamp} для TTL-очистки
_in_memory_tokens: Dict[str, float] = {}
_in_memory_tokens_lock = __import__("threading").Lock()
_IN_MEMORY_TOKENS_MAX_AGE = JWT_REFRESH_EXPIRATION_DAYS * 86400  # 7 дней в секундах
_IN_MEMORY_TOKENS_MAX_SIZE = 10000  # Лимит размера


def _cleanup_expired_tokens():
    """Очистка протухших токенов (prevent memory leak)"""
    import time

    now = time.time()
    expired = [jti for jti, ts in _in_memory_tokens.items() if now - ts > _IN_MEMORY_TOKENS_MAX_AGE]
    for jti in expired:
        _in_memory_tokens.pop(jti, None)

    # Если всё ещё слишком много токенов, удаляем самые старые
    if len(_in_memory_tokens) > _IN_MEMORY_TOKENS_MAX_SIZE:
        sorted_tokens = sorted(_in_memory_tokens.items(), key=lambda x: x[1])
        for jti, _ in sorted_tokens[: len(sorted_tokens) // 2]:
            _in_memory_tokens.pop(jti, None)


def _store_refresh_token(jti: str, username: str):
    """Сохранение refresh токена в Redis"""
    import time

    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            redis_client.setex(key, JWT_REFRESH_EXPIRATION_DAYS * 86400, username)
            return
        except Exception as e:
            logger.error(f"Failed to store token in Redis: {e}")

    # Fallback: in-memory с TTL
    with _in_memory_tokens_lock:
        _in_memory_tokens[jti] = time.time()
        # Периодическая очистка
        if len(_in_memory_tokens) % 100 == 0:
            _cleanup_expired_tokens()


def _is_token_valid(jti: str) -> bool:
    """Проверка валидности refresh токена"""
    import time

    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            return redis_client.exists(key)
        except Exception as e:
            logger.error(f"Failed to check token in Redis: {e}")
            # Fallback к in-memory
            with _in_memory_tokens_lock:
                ts = _in_memory_tokens.get(jti)
                if ts and time.time() - ts <= _IN_MEMORY_TOKENS_MAX_AGE:
                    return True
                _in_memory_tokens.pop(jti, None)
                return False

    # Fallback: in-memory с проверкой TTL
    with _in_memory_tokens_lock:
        ts = _in_memory_tokens.get(jti)
        if ts is None:
            return False
        if time.time() - ts > _IN_MEMORY_TOKENS_MAX_AGE:
            _in_memory_tokens.pop(jti, None)
            return False
        return True


def _revoke_all_user_tokens(username: str):
    """Отмена всех refresh токенов пользователя (при компрометации)"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            # Удаляем все ключи refresh_token:* для данного пользователя
            pattern = "refresh_token:*"
            for key in redis_client.scan_iter(match=pattern):
                if redis_client.get(key) == username:
                    redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to revoke all user tokens in Redis: {e}")

    # Очищаем in-memory только токены данного пользователя
    # (in-memory не хранит username → jti mapping, поэтому очищаем всё как fallback)
    with _in_memory_tokens_lock:
        _in_memory_tokens.clear()


def _revoke_refresh_token(jti: str):
    """Отмена refresh токена (rotation)"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            redis_client.delete(key)
            return
        except Exception as e:
            logger.error(f"Failed to revoke token in Redis: {e}")

    # Fallback: in-memory
    with _in_memory_tokens_lock:
        _in_memory_tokens.pop(jti, None)


def create_refresh_token(data: dict) -> str:
    """Создание refresh токена с уникальным jti"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_EXPIRATION_DAYS)
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
@auth_limit(max_requests=10, window=60)
async def login(request: Request, login_data: LoginRequest):
    """Вход в систему с audit logging"""
    user = _get_users_db().get(login_data.username)

    if not user or not verify_password(login_data.password, user["password_hash"]):
        # Audit: Failed login attempt
        log_audit_event(
            AuditEventType.LOGIN_FAILURE,
            username=login_data.username,
            request=request,
            reason="invalid_credentials",
        )
        raise AuthenticationError("Неверное имя пользователя или пароль")

    # Обновление last_login в памяти и в БД
    now_iso = datetime.now(timezone.utc).isoformat()
    user["last_login"] = now_iso
    try:
        from utils.database import get_database

        get_database().update_last_login(user["username"])
    except Exception as e:
        logger.debug(f"Could not persist last_login: {e}")

    # Auto-migration: если хеш bcrypt, перехешируем на Argon2 при входе
    if not user["password_hash"].startswith("$argon2"):
        logger.info(f"Migrating user {login_data.username} from bcrypt to Argon2")
        user["password_hash"] = hash_password(login_data.password)

    access_token = create_access_token(data={"sub": user["username"], "user_id": user["id"]})
    refresh_token = create_refresh_token(data={"sub": user["username"], "user_id": user["id"]})

    # Audit: Successful login
    log_audit_event(
        AuditEventType.LOGIN_SUCCESS,
        username=user["username"],
        request=request,
        extra={
            "user_id": user["id"],
            "role": user["role"],
            "auth_method": "password",
        },
    )

    logger.info(f"User {user['username']} logged in successfully")

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
    """Обновление токена с refresh token rotation и audit logging"""
    try:
        payload = jwt.decode(request.refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Проверка типа токена
        if payload.get("type") != "refresh":
            logger.warning(f"Token refresh attempt with wrong type: {payload.get('type')}")
            raise AuthenticationError("Неверный тип токена")

        # Проверка jti (rotation check) через Redis
        jti = payload.get("jti")
        username: str = payload.get("sub")

        if not _is_token_valid(jti):
            # REUSE DETECTION: Токен уже был использован (возможная атака)
            if username:
                logger.warning(
                    f"⚠️ POSSIBLE TOKEN REUSE ATTACK detected for user: {username}, jti: {jti}"
                )
                _revoke_all_user_tokens(username)

                log_audit_event(
                    AuditEventType.TOKEN_REVOKED,
                    username=username,
                    request=request,
                    extra={"jti": jti, "reason": "reuse_detected", "all_sessions_revoked": True},
                )

            raise AuthenticationError("Refresh токен был отозван (possible reuse)")

        if username is None:
            raise AuthenticationError("Неверный refresh токен")

        user = _get_users_db().get(username)
        if not user:
            raise AuthenticationError("Пользователь не найден")

        # Ревокация старого токена (rotation)
        revoke_refresh_token(jti)

        # Audit: Token revoked
        log_audit_event(
            AuditEventType.TOKEN_REVOKED,
            username=username,
            request=request,
            extra={"jti": jti, "reason": "rotation"},
        )

        # Создание новой пары токенов
        new_access_token = create_access_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )
        new_refresh_token = create_refresh_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )

        # Audit: Token refreshed
        log_audit_event(
            AuditEventType.TOKEN_REFRESH, username=username, request=request, extra={"jti_old": jti}
        )

        logger.info(f"Token refreshed for user: {username}")

        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=JWT_EXPIRATION_MINUTES * 60,
        )

    except jwt.PyJWTError as e:
        logger.warning(f"Token refresh failed: {e}")
        raise AuthenticationError("Неверный refresh токен")


@router.post(
    "/2fa/setup",
    summary="Настроить 2FA",
    description="Инициация настройки двухфакторной аутентификации",
)
async def setup_2fa(current_user: dict = Depends(get_current_user)):
    """Настройка 2FA"""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    user_email = f"{username}@nanoprobe.local"

    two_factor = get_2fa_manager()
    secret, provisioning_uri = two_factor.setup_2fa(username, user_email)

    return {
        "secret": secret,
        "provisioning_uri": provisioning_uri,
        "message": "Отсканируйте QR код в Google Authenticator",
    }


@router.post(
    "/2fa/verify",
    summary="Верифицировать 2FA",
    description="Подтверждение настройки 2FA OTP кодом",
)
async def verify_2fa_setup(
    otp_code: str, request: Request, current_user: dict = Depends(get_current_user)
):
    """Верификация 2FA с audit logging"""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    if two_factor.verify_2fa_setup(username, otp_code):
        # Audit: 2FA enabled
        log_audit_event(AuditEventType._2FA_ENABLED, username=username, request=request)
        return {"success": True, "message": "2FA успешно включена"}

    # Audit: 2FA verification failed
    log_audit_event(
        AuditEventType._2FA_VERIFICATION_FAILED,
        username=username,
        request=request,
        extra={"stage": "setup"},
    )
    raise ValidationError("Неверный OTP код")


@router.post(
    "/2fa/verify-login",
    summary="2FA верификация при входе",
    description="Проверка 2FA кода после успешного логина",
)
@rate_limit(max_requests=5, window_seconds=60)
@auth_limit(max_requests=10, window=60)
async def verify_2fa_login(otp_code: str, request: Request, username: str, password: str):
    """2FA при входе с audit logging"""
    from utils.security.two_factor_auth import get_2fa_manager

    # Сначала проверяем логин/пароль
    user = _get_users_db().get(username)
    if not user or not verify_password(password, user["password_hash"]):
        log_audit_event(
            AuditEventType.LOGIN_FAILURE,
            username=username,
            request=request,
            reason="invalid_credentials_2fa",
        )
        raise AuthenticationError("Неверное имя пользователя или пароль")

    # Проверяем 2FA если включена
    two_factor = get_2fa_manager()

    if two_factor.is_2fa_enabled(username):
        # Требуется 2FA верификация
        if not two_factor.verify_2fa(username, otp_code):
            # Пробуем резервный код
            if not two_factor.verify_backup_code(username, otp_code):
                log_audit_event(
                    AuditEventType._2FA_VERIFICATION_FAILED,
                    username=username,
                    request=request,
                    extra={"stage": "login"},
                )
                raise AuthenticationError("Неверный 2FA код")

    # Генерация токенов
    access_token = create_access_token(data={"sub": user["username"], "user_id": user["id"]})
    refresh_token = create_refresh_token(data={"sub": user["username"], "user_id": user["id"]})

    # Audit: Successful login with 2FA
    log_audit_event(
        AuditEventType.LOGIN_SUCCESS,
        username=username,
        request=request,
        extra={
            "user_id": user["id"],
            "role": user["role"],
            "auth_method": "password+2fa",
        },
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
            "2fa_enabled": two_factor.is_2fa_enabled(username),
        },
    )


@router.get(
    "/2fa/status",
    summary="Статус 2FA",
    description="Проверка статуса 2FA для текущего пользователя",
)
async def get_2fa_status(current_user: dict = Depends(get_current_user)):
    """Статус 2FA"""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    return {"enabled": two_factor.is_2fa_enabled(username), "username": username}


@router.post(
    "/2fa/disable",
    summary="Отключить 2FA",
    description="Отключение двухфакторной аутентификации",
)
async def disable_2fa(
    otp_code: str, request: Request, current_user: dict = Depends(get_current_user)
):
    """Отключение 2FA с audit logging"""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    if two_factor.disable_2fa(username, otp_code):
        # Audit: 2FA disabled
        log_audit_event(AuditEventType._2FA_DISABLED, username=username, request=request)
        return {"success": True, "message": "2FA отключена"}

    # Audit: 2FA disable failed
    log_audit_event(
        AuditEventType._2FA_VERIFICATION_FAILED,
        username=username,
        request=request,
        extra={"stage": "disable"},
    )
    raise ValidationError("Неверный OTP код")


@router.post(
    "/2fa/backup-codes",
    summary="Генерировать резервные коды",
    description="Генерация резервных кодов для 2FA",
)
async def generate_backup_codes(current_user: dict = Depends(get_current_user)):
    """Генерация резервных кодов"""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    codes = two_factor.generate_backup_codes(username, count=10)

    return {"backup_codes": codes, "message": "Сохраните эти коды в безопасном месте!"}


@router.get(
    "/me",
    summary="Текущий пользователь",
    description="Получение информации о текущем пользователе",
)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Получение информации о текущем пользователе"""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"],
        "created_at": current_user["created_at"],
        "2fa_enabled": two_factor.is_2fa_enabled(username),
    }


@router.post(
    "/logout",
    summary="Выход из системы",
    description="Выход из системы с ревокацией refresh токена",
)
async def logout(request: Request, refresh_token: Optional[str] = None):
    """Выход из системы с ревокацией refresh токена и audit logging"""
    username = "unknown"

    if refresh_token:
        try:
            payload = jwt.decode(
                refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False}
            )
            jti = payload.get("jti")
            username = payload.get("sub", "unknown")

            if jti:
                revoke_refresh_token(jti)

                # Audit: Token revoked on logout
                log_audit_event(
                    AuditEventType.TOKEN_REVOKED,
                    username=username,
                    request=request,
                    extra={"jti": jti, "reason": "logout"},
                )
        except jwt.PyJWTError:
            pass

    # Audit: Logout
    log_audit_event(AuditEventType.LOGOUT, username=username, request=request)

    return {"message": "Успешный выход. Удалите токены на клиенте."}


@router.get(
    "/rate-limit-status",
    summary="Статус rate limiting",
    description="Получение статуса rate limiting для текущего IP",
)
async def get_rate_limit_status(request: Request):
    """Получение статуса rate limiting"""
    from utils.security.rate_limiter import limiter

    client_ip = request.client.host
    login_key = f"login:{client_ip}"

    status = limiter.get_status(login_key, max_requests=5, window_seconds=300)
    return {
        "ip": client_ip,
        "endpoint": "login",
        **status,
    }
