# -*- coding: utf-8 -*-
"""
Централизованная обработка ошибок для FastAPI API
Унифицированная обработка исключений с категоризацией по severity
"""

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import logging
import traceback
import os

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Уровни важности ошибок"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class APIError(Exception):
    """Базовое исключение API"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.severity = severity
        self.error_code = error_code or f"ERR_{status_code}"
        self.details = details or {}
        super().__init__(message)


class ValidationError(APIError):
    """Ошибка валидации данных"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            severity=ErrorSeverity.WARNING,
            error_code="ERR_VALIDATION",
            details=details
        )


class NotFoundError(APIError):
    """Ресурс не найден"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=404,
            severity=ErrorSeverity.WARNING,
            error_code="ERR_NOT_FOUND",
            details={"resource_type": resource_type} if resource_type else {}
        )


class AuthenticationError(APIError):
    """Ошибка аутентификации"""
    
    def __init__(self, message: str = "Неверные учетные данные"):
        super().__init__(
            message=message,
            status_code=401,
            severity=ErrorSeverity.WARNING,
            error_code="ERR_AUTH",
            details={}
        )


class AuthorizationError(APIError):
    """Ошибка авторизации (нет прав)"""
    
    def __init__(self, message: str = "Недостаточно прав"):
        super().__init__(
            message=message,
            status_code=403,
            severity=ErrorSeverity.WARNING,
            error_code="ERR_FORBIDDEN",
            details={}
        )


class RateLimitError(APIError):
    """Превышен лимит запросов"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Слишком много запросов",
            status_code=429,
            severity=ErrorSeverity.WARNING,
            error_code="ERR_RATE_LIMIT",
            details={"retry_after": retry_after}
        )


class DatabaseError(APIError):
    """Ошибка базы данных"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=503,
            severity=ErrorSeverity.CRITICAL,
            error_code="ERR_DATABASE",
            details=details
        )


class ExternalServiceError(APIError):
    """Ошибка внешнего сервиса"""
    
    def __init__(self, service_name: str, message: str):
        super().__init__(
            message=f"Ошибка сервиса {service_name}: {message}",
            status_code=503,
            severity=ErrorSeverity.ERROR,
            error_code="ERR_EXTERNAL_SERVICE",
            details={"service_name": service_name}
        )


def create_error_response(
    error: Exception,
    status_code: int,
    request: Request,
    severity: ErrorSeverity = ErrorSeverity.ERROR
) -> JSONResponse:
    """Создание унифицированного ответа об ошибке"""
    
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.error(
        f"[{severity.value.upper()}] {error_id} - {type(error).__name__}: {str(error)}",
        extra={
            "error_id": error_id,
            "path": str(request.url.path),
            "method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "error_id": error_id,
            "status_code": status_code,
            "error_code": getattr(error, "error_code", f"ERR_{status_code}"),
            "message": str(error),
            "severity": severity.value,
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
            **getattr(error, "details", {})
        }
    )


async def api_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTP исключений"""
    severity = ErrorSeverity.WARNING if exc.status_code < 500 else ErrorSeverity.ERROR

    return create_error_response(
        error=exc,
        status_code=exc.status_code,
        request=request,
        severity=severity
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTPException с добавлением severity"""
    severity = ErrorSeverity.WARNING if exc.status_code < 500 else ErrorSeverity.ERROR

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"ERR_{exc.status_code}",
            "severity": severity.value,
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
        }
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Обработчик ошибок валидации"""
    
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error.get("loc", [])),
            "message": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
    logger.warning(
        f"Validation error: {errors}",
        extra={
            "path": str(request.url.path),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "error_code": "ERR_VALIDATION",
            "message": "Ошибка валидации данных",
            "status_code": 422,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
    )


async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик общих исключений"""
    
    error_trace = traceback.format_exc()
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.error(
        f"[CRITICAL] {error_id} - {type(exc).__name__}: {str(exc)}\n{error_trace}",
        extra={
            "error_id": error_id,
            "path": str(request.url.path),
            "method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    # В production не показываем детали ошибки клиенту
    show_details = os.getenv("API_DEBUG", "false").lower() == "true"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_id": error_id,
            "error_code": "ERR_INTERNAL",
            "message": "Внутренняя ошибка сервера" if not show_details else str(exc),
            "status_code": 500,
            "severity": "critical",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
            **({"traceback": error_trace} if show_details else {})
        }
    )


async def api_error_handler_wrapper(request: Request, exc: APIError) -> JSONResponse:
    """Обработчик кастомных API ошибок"""
    return create_error_response(
        error=exc,
        status_code=exc.status_code,
        request=request,
        severity=exc.severity
    )


def register_error_handlers(app):
    """Регистрация обработчиков ошибок в FastAPI приложении"""

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, general_error_handler)
    app.add_exception_handler(APIError, api_error_handler_wrapper)
    app.add_exception_handler(ValidationError, api_error_handler_wrapper)
    app.add_exception_handler(NotFoundError, api_error_handler_wrapper)
    app.add_exception_handler(AuthenticationError, api_error_handler_wrapper)
    app.add_exception_handler(AuthorizationError, api_error_handler_wrapper)
    app.add_exception_handler(RateLimitError, api_error_handler_wrapper)
    app.add_exception_handler(DatabaseError, api_error_handler_wrapper)
    app.add_exception_handler(ExternalServiceError, api_error_handler_wrapper)
    
    logger.info("Error handlers registered")


def handle_errors(func):
    """Декоратор для автоматической обработки ошибок в endpoint'ах"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except APIError:
            raise
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper
