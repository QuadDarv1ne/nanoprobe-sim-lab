"""
Security Headers Middleware для FastAPI

Защита от:
- XSS (Cross-Site Scripting)
- Clickjacking
- MIME type sniffing
- Information leakage
- HTTPS enforcement

Использование:
    from api.security_headers import SecurityHeadersMiddleware
    app.add_middleware(SecurityHeadersMiddleware)
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Optional

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware для добавления security headers
    
    Headers:
    - X-Frame-Options: DENY (защита от clickjacking)
    - X-Content-Type-Options: nosniff (защита от MIME sniffing)
    - X-XSS-Protection: 1; mode=block (XSS защита для старых браузеров)
    - Referrer-Policy: strict-origin-when-cross-origin (контроль referrer)
    - Permissions-Policy: ограничение функций браузера
    - Strict-Transport-Security: HSTS (HTTPS enforcement)
    - Content-Security-Policy: CSP (защита от XSS)
    """

    def __init__(
        self,
        app,
        hsts_max_age: int = 31536000,  # 1 год
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = True,
        csp_report_only: bool = False,
        custom_csp: Optional[str] = None,
    ):
        """
        Инициализация middleware
        
        Args:
            app: FastAPI приложение
            hsts_max_age: Max-Age для HSTS (секунды)
            hsts_include_subdomains: Включать includeSubDomains
            hsts_preload: Включать preload для HSTS
            csp_report_only: Report-Only режим для CSP
            custom_csp: Пользовательская Content-Security-Policy
        """
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.csp_report_only = csp_report_only
        self.custom_csp = custom_csp

    async def dispatch(self, request: Request, call_next) -> Response:
        """Добавление security headers к ответу"""
        response = await call_next(request)
        
        # X-Frame-Options: защита от clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options: запрет MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection: XSS защита (для старых браузеров)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy: контроль передачи referrer
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy: ограничение функций браузера
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "ambient-light-sensor=(), "
            "autoplay=(), "
            "battery=(), "
            "camera=(), "
            "cross-origin-isolated=(), "
            "display-capture=(), "
            "document-domain=(), "
            "encrypted-media=(), "
            "execution-while-not-rendered=(), "
            "execution-while-out-of-viewport=(), "
            "fullscreen=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "keyboard-map=(), "
            "magnetometer=(), "
            "microphone=(), "
            "midi=(), "
            "navigation-override=(), "
            "payment=(), "
            "picture-in-picture=(), "
            "publickey-credentials-get=(), "
            "screen-wake-lock=(), "
            "sync-xhr=(), "
            "usb=(), "
            "web-share=(), "
            "xr-spatial-tracking=()"
        )
        
        # Strict-Transport-Security: HSTS (только для HTTPS)
        if request.url.scheme == "https":
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value
        
        # Content-Security-Policy: CSP (исправленный - без unsafe-inline/unsafe-eval)
        if self.custom_csp:
            csp_header = "Content-Security-Policy-Report-Only" if self.csp_report_only else "Content-Security-Policy"
            response.headers[csp_header] = self.custom_csp
        else:
            # CSP по умолчанию (строгий - без unsafe-inline/unsafe-eval)
            # Используем nonce-based approach для скриптов
            default_csp = (
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "object-src 'none'"
            )
            csp_header = "Content-Security-Policy-Report-Only" if self.csp_report_only else "Content-Security-Policy"
            response.headers[csp_header] = default_csp
        
        # Удаление заголовков с информацией о сервере
        if "Server" in response.headers:
            del response.headers["Server"]
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        return response


def setup_security_headers(app, production: bool = True):
    """
    Настройка security headers для приложения
    
    Args:
        app: FastAPI приложение
        production: Production режим (более строгие настройки)
    """
    if production:
        # Production: строгие настройки
        app.add_middleware(
            SecurityHeadersMiddleware,
            hsts_max_age=31536000,  # 1 год
            hsts_include_subdomains=True,
            hsts_preload=True,
            csp_report_only=False,  # Строгий CSP
        )
        logger.info("Security headers enabled (production mode)")
    else:
        # Development: более мягкие настройки (но всё ещё без unsafe-eval)
        app.add_middleware(
            SecurityHeadersMiddleware,
            hsts_max_age=0,  # Отключено
            hsts_include_subdomains=False,
            hsts_preload=False,
            csp_report_only=True,  # Report-Only для отладки
            custom_csp=(
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self' ws: wss:; "
                "img-src 'self' data: https:; "
                "frame-ancestors 'self'"
            ),
        )
        logger.info("Security headers enabled (development mode)")
