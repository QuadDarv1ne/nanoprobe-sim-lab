"""
Auth endpoints

Login, refresh token, logout, 2FA.
"""

import logging
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, Request

from api.dependencies import get_current_user, rate_limit
from api.error_handlers import AuthenticationError, ValidationError
from api.rate_limiter import auth_limit
from api.routes.auth_routes.helpers import (
    JWT_EXPIRATION_MINUTES,
    AuditEventType,
    _get_users_db,
    hash_password,
    log_audit_event,
    verify_password,
)
from api.routes.auth_routes.tokens import (
    create_access_token,
    create_refresh_token,
    decode_token,
    is_token_valid,
    revoke_all_user_tokens,
    revoke_refresh_token,
)
from api.schemas import ErrorResponse, LoginRequest, LoginResponse, RefreshTokenRequest, Token

logger = logging.getLogger(__name__)

router = APIRouter()


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
    """Вход в систему с audit logging."""
    user = _get_users_db().get(login_data.username)

    if not user or not verify_password(login_data.password, user["password_hash"]):
        log_audit_event(
            AuditEventType.LOGIN_FAILURE,
            username=login_data.username,
            request=request,
            reason="invalid_credentials",
        )
        raise AuthenticationError("Неверное имя пользователя или пароль")

    now_iso = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    user["last_login"] = now_iso
    try:
        from utils.database import get_database

        get_database().update_last_login(user["username"])
    except Exception as e:
        logger.debug(f"Could not persist last_login: {e}")

    if not user["password_hash"].startswith("$argon2"):
        logger.info(f"Migrating user {login_data.username} from bcrypt to Argon2")
        user["password_hash"] = hash_password(login_data.password)

    access_token = create_access_token(data={"sub": user["username"], "user_id": user["id"]})
    refresh_token = create_refresh_token(data={"sub": user["username"], "user_id": user["id"]})

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
    """Обновление токена с refresh token rotation."""
    try:
        from api.routes.auth_routes.helpers import JWT_ALGORITHM, JWT_SECRET

        payload = jwt.decode(request.refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        if payload.get("type") != "refresh":
            logger.warning(f"Token refresh attempt with wrong type: {payload.get('type')}")
            raise AuthenticationError("Неверный тип токена")

        jti = payload.get("jti")
        username: str = payload.get("sub")

        if not is_token_valid(jti):
            if username:
                logger.warning(
                    f"POSSIBLE TOKEN REUSE ATTACK detected for user: {username}, jti: {jti}"
                )
                revoke_all_user_tokens(username)

                log_audit_event(
                    AuditEventType.TOKEN_REVOKED,
                    username=username,
                    request=request,
                    extra={
                        "jti": jti,
                        "reason": "reuse_detected",
                        "all_sessions_revoked": True,
                    },
                )

            raise AuthenticationError("Refresh токен был отозван (possible reuse)")

        if username is None:
            raise AuthenticationError("Неверный refresh токен")

        user = _get_users_db().get(username)
        if not user:
            raise AuthenticationError("Пользователь не найден")

        revoke_refresh_token(jti)

        log_audit_event(
            AuditEventType.TOKEN_REVOKED,
            username=username,
            request=request,
            extra={"jti": jti, "reason": "rotation"},
        )

        new_access_token = create_access_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )
        new_refresh_token = create_refresh_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )

        log_audit_event(
            AuditEventType.TOKEN_REFRESH,
            username=username,
            request=request,
            extra={"jti_old": jti},
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
    """Настройка 2FA."""
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
    otp_code: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Верификация 2FA с audit logging."""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    if two_factor.verify_2fa_setup(username, otp_code):
        log_audit_event(AuditEventType._2FA_ENABLED, username=username, request=request)
        return {"success": True, "message": "2FA успешно включена"}

    log_audit_event(
        AuditEventType._2FA_VERIFICATION_FAILED,
        username=username,
        request=request,
        extra={"stage": "setup"},
    )
    raise ValidationError("Неверный OTP код")


@router.get(
    "/2fa/status",
    summary="Статус 2FA",
    description="Проверка статуса 2FA для текущего пользователя",
)
async def get_2fa_status(current_user: dict = Depends(get_current_user)):
    """Статус 2FA."""
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
    otp_code: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Отключение 2FA с audit logging."""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    if two_factor.disable_2fa(username, otp_code):
        log_audit_event(AuditEventType._2FA_DISABLED, username=username, request=request)
        return {"success": True, "message": "2FA отключена"}

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
    """Генерация резервных кодов."""
    from utils.security.two_factor_auth import get_2fa_manager

    username = current_user["username"]
    two_factor = get_2fa_manager()

    codes = two_factor.generate_backup_codes(username, count=10)

    return {
        "backup_codes": codes,
        "message": "Сохраните эти коды в безопасном месте!",
    }


@router.get(
    "/me",
    summary="Текущий пользователь",
    description="Получение информации о текущем пользователе",
)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Получение информации о текущем пользователе."""
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
async def logout(
    request: Request,
    refresh_token: Optional[str] = None,
):
    """Выход из системы с ревокацией refresh токена."""
    username = "unknown"

    if refresh_token:
        try:
            from api.routes.auth_routes.helpers import JWT_ALGORITHM, JWT_SECRET

            payload = jwt.decode(
                refresh_token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            username = payload.get("sub", "unknown")

            if jti:
                revoke_refresh_token(jti)

                log_audit_event(
                    AuditEventType.TOKEN_REVOKED,
                    username=username,
                    request=request,
                    extra={"jti": jti, "reason": "logout"},
                )
        except jwt.PyJWTError:
            pass

    log_audit_event(AuditEventType.LOGOUT, username=username, request=request)

    return {"message": "Успешный выход. Удалите токены на клиенте."}


@router.get(
    "/rate-limit-status",
    summary="Статус rate limiting",
    description="Получение статуса rate limiting для текущего IP",
)
async def get_rate_limit_status(request: Request):
    """Получение статуса rate limiting."""
    from utils.security.rate_limiter import limiter

    client_ip = request.client.host
    login_key = f"login:{client_ip}"

    status = limiter.get_status(login_key, max_requests=5, window_seconds=300)
    return {
        "ip": client_ip,
        "endpoint": "login",
        **status,
    }
