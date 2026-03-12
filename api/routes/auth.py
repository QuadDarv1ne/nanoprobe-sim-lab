# -*- coding: utf-8 -*-
"""
API роуты для аутентификации
JWT токен, логин, регистрация
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional
import jwt
import os

from passlib.context import CryptContext

from api.schemas import (
    LoginRequest,
    LoginResponse,
    Token,
    ErrorResponse,
)
from utils.rate_limiter import rate_limit

router = APIRouter()
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60
JWT_REFRESH_EXPIRATION_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

USERS_DB = {
    "admin": {
        "id": 1,
        "username": "admin",
        "password_hash": pwd_context.hash("admin123"),
        "role": "admin",
        "created_at": "2026-03-11T00:00:00",
        "last_login": None,
    },
    "user": {
        "id": 2,
        "username": "user",
        "password_hash": pwd_context.hash("user123"),
        "role": "user",
        "created_at": "2026-03-11T00:00:00",
        "last_login": None,
    },
}


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Проверка надёжности пароля"""
    if len(password) < 8:
        return False, "Пароль должен быть не менее 8 символов"
    if not any(c.isupper() for c in password):
        return False, "Пароль должен содержать заглавную букву"
    if not any(c.isdigit() for c in password):
        return False, "Пароль должен содержать цифру"
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
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Создание refresh токена"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_EXPIRATION_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Получение текущего пользователя из токена"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
        
        user = USERS_DB.get(username)
        if not user:
            raise credentials_exception
        
        return user
    except jwt.PyJWTError:
        raise credentials_exception


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
async def login(request: Request, login_data: LoginRequest):
    """Вход в систему"""
    user = USERS_DB.get(login_data.username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
        )

    if not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
        )

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
    description="Обновление access токена с помощью refresh токена",
)
async def refresh_token(refresh_token: str):
    """Обновление токена"""
    try:
        payload = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный refresh токен",
            )
        
        user = USERS_DB.get(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Пользователь не найден",
            )
        
        # Создание нового access токена
        new_access_token = create_access_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )
        
        return Token(
            access_token=new_access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=JWT_EXPIRATION_MINUTES * 60,
        )
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный refresh токен",
        )


@router.get(
    "/me",
    summary="Текущий пользователь",
    description="Получение информации о текущем пользователе",
)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Получение информации о текущем пользователе"""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"],
        "created_at": current_user["created_at"],
    }


@router.post(
    "/logout",
    summary="Выход из системы",
    description="Выход из системы (на клиенте удалить токены)",
)
async def logout():
    """Выход из системы"""
    # На клиенте нужно удалить токены
    return {"message": "Успешный выход. Удалите токены на клиенте."}
