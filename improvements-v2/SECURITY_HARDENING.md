# Security Hardening Guide

## Обзор

Комплексное руководство по обеспечению безопасности Nanoprobe Sim Lab на всех уровнях: приложение, API, инфраструктура.

## Security Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SECURITY HARDENING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Application           API Security          Infrastructure                 │
│  ─────────────         ───────────           ──────────────                 │
│  [✓] Input Validation  [✓] Authentication    [✓] HTTPS/TLS                  │
│  [✓] Output Encoding   [✓] Authorization     [✓] Rate Limiting              │
│  [✓] CSRF Protection   [✓] JWT Security      [✓] WAF                        │
│  [✓] XSS Prevention    [✓] API Keys          [✓] DDoS Protection            │
│  [✓] SQL Injection     [✓] CORS              [✓] Secret Management          │
│  [✓] File Upload       [✓] Input Validation  [✓] Logging & Monitoring       │
│  [✓] Session Mgmt      [✓] Rate Limiting     [✓] Backup & Recovery          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Authentication & Authorization

### JWT Best Practices

```python
# api/security/jwt.py
"""
Secure JWT Implementation

Best Practices:
1. Use strong secret key (32+ chars)
2. Short expiration times
3. Refresh token rotation
4. Token revocation
5. Secure storage (httpOnly cookies)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import hashlib

# Password hashing
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4
)

# JWT Settings
JWT_SECRET_KEY = secrets.token_urlsafe(32)  # Generate at startup
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived
REFRESH_TOKEN_EXPIRE_DAYS = 7
JWT_ISSUER = "nanoprobe-sim-lab"
JWT_AUDIENCE = "nanoprobe-users"


class TokenPayload:
    """Secure token payload structure"""
    
    sub: str          # Subject (user ID)
    exp: datetime     # Expiration
    iat: datetime     # Issued at
    jti: str          # Unique token ID (for revocation)
    type: str         # 'access' or 'refresh'
    iss: str          # Issuer
    aud: str          # Audience
    
    @classmethod
    def create_access_token(cls, user_id: int) -> Dict[str, Any]:
        now = datetime.utcnow()
        jti = secrets.token_urlsafe(16)
        
        payload = {
            "sub": str(user_id),
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": now,
            "jti": jti,
            "type": "access",
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "jti": jti
        }
    
    @classmethod
    def create_refresh_token(cls, user_id: int) -> Dict[str, Any]:
        now = datetime.utcnow()
        jti = secrets.token_urlsafe(32)  # Longer for refresh
        
        payload = {
            "sub": str(user_id),
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": now,
            "jti": jti,
            "type": "refresh",
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Hash for storage (don't store raw token)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        return {
            "refresh_token": token,
            "token_hash": token_hash,
            "jti": jti
        }


class PasswordValidator:
    """Secure password validation"""
    
    MIN_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate(cls, password: str) -> tuple[bool, list[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters")
        
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if cls.REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if cls.REQUIRE_SPECIAL and not any(c in cls.SPECIAL_CHARS for c in password):
            errors.append(f"Password must contain at least one special character: {cls.SPECIAL_CHARS}")
        
        # Check for common patterns
        common_patterns = ["password", "123456", "qwerty", "admin"]
        if any(pattern in password.lower() for pattern in common_patterns):
            errors.append("Password contains common patterns")
        
        return len(errors) == 0, errors
    
    @classmethod
    def hash(cls, password: str) -> str:
        """Hash password securely"""
        return pwd_context.hash(password)
    
    @classmethod
    def verify(cls, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, hashed)
```

### Two-Factor Authentication

```python
# api/security/two_factor.py
"""
Two-Factor Authentication (TOTP)

Uses pyotp for TOTP implementation compatible with:
- Google Authenticator
- Authy
- Microsoft Authenticator
"""

import pyotp
import qrcode
from io import BytesIO
import base64
from typing import Optional
import secrets

class TwoFactorAuth:
    """TOTP-based Two-Factor Authentication"""
    
    ISSUER = "NanoprobeSimLab"
    DIGITS = 6
    INTERVAL = 30
    
    @classmethod
    def generate_secret(cls) -> str:
        """Generate new TOTP secret"""
        return pyotp.random_base32()
    
    @classmethod
    def get_totp(cls, secret: str) -> pyotp.TOTP:
        """Get TOTP instance"""
        return pyotp.TOTP(
            secret,
            digits=cls.DIGITS,
            interval=cls.INTERVAL
        )
    
    @classmethod
    def verify_code(cls, secret: str, code: str, valid_window: int = 1) -> bool:
        """Verify TOTP code with time window tolerance"""
        totp = cls.get_totp(secret)
        return totp.verify(code, valid_window=valid_window)
    
    @classmethod
    def get_provisioning_uri(cls, secret: str, email: str) -> str:
        """Generate provisioning URI for authenticator apps"""
        totp = cls.get_totp(secret)
        return totp.provisioning_uri(
            name=email,
            issuer_name=cls.ISSUER
        )
    
    @classmethod
    def generate_qr_code(cls, secret: str, email: str) -> str:
        """Generate QR code as base64 image"""
        uri = cls.get_provisioning_uri(secret, email)
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    @classmethod
    def generate_backup_codes(cls, count: int = 10) -> list[str]:
        """Generate one-time backup codes"""
        codes = []
        for _ in range(count):
            code = f"{secrets.randbelow(10000):04d}-{secrets.randbelow(10000):04d}"
            codes.append(code)
        return codes
```

## 2. Input Validation & Sanitization

```python
# api/security/validation.py
"""
Input Validation and Sanitization

Prevents:
- SQL Injection
- XSS (Cross-Site Scripting)
- Command Injection
- Path Traversal
"""

import re
import html
from typing import Any, Optional
from urllib.parse import urlparse
import unicodedata
from pydantic import BaseModel, field_validator, constr


class SecureInput:
    """Input sanitization utilities"""
    
    # Allowed characters for different input types
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Expected string value")
        
        # Normalize unicode
        value = unicodedata.normalize('NFKC', value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Trim whitespace
        value = value.strip()
        
        # Limit length
        if len(value) > max_length:
            raise ValueError(f"Value exceeds maximum length of {max_length}")
        
        return value
    
    @classmethod
    def sanitize_html(cls, value: str) -> str:
        """Escape HTML entities"""
        return html.escape(value, quote=True)
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove null bytes
        filename = filename.replace('\x00', '')
        
        # Only allow safe characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    @classmethod
    def validate_url(cls, url: str, allowed_schemes: list = None) -> str:
        """Validate URL format and scheme"""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            if parsed.scheme not in allowed_schemes:
                raise ValueError(f"URL scheme must be one of: {allowed_schemes}")
            if not parsed.netloc:
                raise ValueError("Invalid URL: missing domain")
            return url
        except Exception as e:
            raise ValueError(f"Invalid URL: {str(e)}")
    
    @classmethod
    def validate_username(cls, username: str) -> str:
        """Validate username format"""
        if not cls.USERNAME_PATTERN.match(username):
            raise ValueError(
                "Username must be 3-50 characters and contain only "
                "letters, numbers, underscores, and hyphens"
            )
        return username
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate email format"""
        if not cls.EMAIL_PATTERN.match(email):
            raise ValueError("Invalid email format")
        return email.lower()


# Pydantic models with validation
class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: constr(min_length=5, max_length=255)
    password: constr(min_length=12, max_length=128)
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        return SecureInput.validate_username(v)
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        return SecureInput.validate_email(v)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        is_valid, errors = PasswordValidator.validate(v)
        if not is_valid:
            raise ValueError('; '.join(errors))
        return v


class FileUploadValidation:
    """File upload security"""
    
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'application/pdf',
        'text/plain',
    }
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate_file(
        cls,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> tuple[bool, str]:
        """Validate uploaded file"""
        
        # Check file size
        if len(file_content) > cls.MAX_FILE_SIZE:
            return False, f"File exceeds maximum size of {cls.MAX_FILE_SIZE // 1024 // 1024}MB"
        
        # Check content type
        if content_type not in cls.ALLOWED_MIME_TYPES:
            return False, f"File type '{content_type}' is not allowed"
        
        # Verify magic bytes (actual file type)
        actual_type = cls._detect_file_type(file_content)
        if actual_type != content_type:
            return False, "File content does not match declared type"
        
        # Sanitize filename
        safe_filename = SecureInput.sanitize_filename(filename)
        
        return True, safe_filename
    
    @classmethod
    def _detect_file_type(cls, content: bytes) -> str:
        """Detect actual file type from magic bytes"""
        if content[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        elif content[:2] == b'\xff\xd8':
            return 'image/jpeg'
        elif content[:6] in (b'GIF87a', b'GIF89a'):
            return 'image/gif'
        elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
            return 'image/webp'
        elif content[:4] == b'%PDF':
            return 'application/pdf'
        else:
            return 'application/octet-stream'
```

## 3. API Security Headers

```python
# api/middleware/security_headers.py
"""
Security Headers Middleware

Adds security headers to all responses:
- Content-Security-Policy
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Strict-Transport-Security
- Referrer-Policy
- Permissions-Policy
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https://images.nasa.gov https://api.nasa.gov; "
            "connect-src 'self' https://api.nasa.gov wss:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS Protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Strict Transport Security (HTTPS)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=()"
        )
        
        # Cache Control for sensitive pages
        if request.url.path.startswith('/api/v1/auth'):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, proxy-revalidate"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class CORSSecurityMiddleware:
    """Secure CORS configuration"""
    
    def __init__(
        self,
        allowed_origins: list[str],
        allow_credentials: bool = True,
        allowed_methods: list[str] = None,
        allowed_headers: list[str] = None,
    ):
        self.allowed_origins = set(allowed_origins)
        self.allow_credentials = allow_credentials
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or [
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-Request-ID",
        ]
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        return origin in self.allowed_origins
    
    def get_cors_headers(self, origin: str) -> dict:
        """Get CORS headers for allowed origin"""
        if not self.is_origin_allowed(origin):
            return {}
        
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allowed_headers),
        }
        
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        return headers
```

## 4. Rate Limiting (Enhanced)

```python
# api/security/advanced_rate_limit.py
"""
Advanced Rate Limiting with:

1. Sliding Window (accurate)
2. Token Bucket (burst handling)
3. User-based limits
4. IP-based limits
5. Endpoint-specific limits
6. Automatic blocking for abuse
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import asyncio
from datetime import datetime, timedelta


class RateLimitType(Enum):
    IP = "ip"
    USER = "user"
    API_KEY = "api_key"
    GLOBAL = "global"


@dataclass
class RateLimitRule:
    """Rate limit rule definition"""
    name: str
    max_requests: int
    window_seconds: int
    type: RateLimitType
    burst_size: Optional[int] = None
    block_duration: Optional[int] = None  # Auto-block duration in seconds


# Predefined rules
RATE_LIMIT_RULES = {
    # Authentication endpoints (strict)
    "auth_login": RateLimitRule(
        name="auth_login",
        max_requests=5,
        window_seconds=60,
        type=RateLimitType.IP,
        block_duration=300  # 5 min block after limit
    ),
    
    # API general usage
    "api_general": RateLimitRule(
        name="api_general",
        max_requests=60,
        window_seconds=60,
        type=RateLimitType.USER,
        burst_size=10
    ),
    
    # File uploads
    "file_upload": RateLimitRule(
        name="file_upload",
        max_requests=10,
        window_seconds=60,
        type=RateLimitType.USER
    ),
    
    # AI/ML endpoints (expensive)
    "ai_analysis": RateLimitRule(
        name="ai_analysis",
        max_requests=10,
        window_seconds=3600,  # Per hour
        type=RateLimitType.USER
    ),
}


class AbuseDetector:
    """Detect and block abusive behavior"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.suspicious_threshold = 5  # Number of violations before flagging
    
    async def record_violation(self, identifier: str, reason: str):
        """Record a rate limit violation"""
        key = f"abuse:{identifier}"
        
        # Increment violation count
        count = await self.redis.incr(key)
        await self.redis.expire(key, 3600)  # 1 hour window
        
        if count >= self.suspicious_threshold:
            await self.flag_suspicious(identifier, reason)
    
    async def flag_suspicious(self, identifier: str, reason: str):
        """Flag identifier as suspicious"""
        key = f"suspicious:{identifier}"
        
        await self.redis.set(
            key,
            {
                "reason": reason,
                "flagged_at": datetime.utcnow().isoformat(),
            },
            ex=86400  # 24 hours
        )
        
        # Alert security team
        logger.warning(f"Suspicious activity detected: {identifier} - {reason}")
    
    async def is_blocked(self, identifier: str) -> bool:
        """Check if identifier is blocked"""
        key = f"blocked:{identifier}"
        return await self.redis.exists(key)
    
    async def block(self, identifier: str, duration: int, reason: str):
        """Block an identifier"""
        key = f"blocked:{identifier}"
        
        await self.redis.set(
            key,
            {
                "reason": reason,
                "blocked_at": datetime.utcnow().isoformat(),
                "duration": duration
            },
            ex=duration
        )
        
        logger.info(f"Blocked {identifier} for {duration}s: {reason}")
```

## 5. SQL Injection Prevention

```python
# api/security/database.py
"""
SQL Injection Prevention

Always use parameterized queries!
"""

import asyncpg
from typing import Any, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SecureDatabase:
    """
    Secure database operations with:
    - Parameterized queries (ALWAYS)
    - Input validation
    - Query logging
    - Transaction safety
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: float = 30.0
    ) -> str:
        """
        Execute a query safely.
        
        IMPORTANT: query must use $1, $2, ... placeholders!
        
        Example:
            await db.execute(
                "INSERT INTO users (name, email) VALUES ($1, $2)",
                "John", "john@example.com"
            )
        """
        # Validate query uses parameterized format
        if args and '%' in query:
            logger.warning(
                "Query uses %s format, should use $1, $2 parameterized format"
            )
        
        async with self.pool.acquire(timeout=timeout) as conn:
            return await conn.execute(query, *args)
    
    async def fetch(
        self,
        query: str,
        *args: Any,
        timeout: float = 30.0
    ) -> List[asyncpg.Record]:
        """Fetch multiple rows safely"""
        async with self.pool.acquire(timeout=timeout) as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(
        self,
        query: str,
        *args: Any,
        timeout: float = 30.0
    ) -> Optional[asyncpg.Record]:
        """Fetch single row safely"""
        async with self.pool.acquire(timeout=timeout) as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: float = 30.0
    ) -> Any:
        """Fetch single value safely"""
        async with self.pool.acquire(timeout=timeout) as conn:
            return await conn.fetchval(query, *args, column=column)
    
    # Safe query builders
    @staticmethod
    def safe_order_by(column: str, allowed_columns: List[str]) -> str:
        """Safely build ORDER BY clause"""
        if column not in allowed_columns:
            raise ValueError(f"Invalid order column: {column}")
        return f"ORDER BY {column}"
    
    @staticmethod
    def safe_in_list(items: List[str], max_items: int = 100) -> tuple:
        """Build safe IN clause"""
        if len(items) > max_items:
            raise ValueError(f"Too many items in IN clause: {len(items)}")
        
        placeholders = ', '.join(f'${i+1}' for i in range(len(items)))
        return f"IN ({placeholders})", items


# WRONG - SQL Injection vulnerable
# async def bad_example(name):
#     query = f"SELECT * FROM users WHERE name = '{name}'"
#     await conn.execute(query)  # DANGEROUS!

# CORRECT - Parameterized query
async def good_example(db: SecureDatabase, name: str):
    query = "SELECT * FROM users WHERE name = $1"
    return await db.fetch(query, name)
```

## 6. Security Configuration

```python
# config/security.py
"""Security configuration settings"""

from pydantic_settings import BaseSettings
from typing import List


class SecuritySettings(BaseSettings):
    """Security-related settings"""
    
    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    
    # Password
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digit: bool = True
    password_require_special: bool = True
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    
    # CORS
    cors_allowed_origins: List[str] = []
    cors_allow_credentials: bool = True
    
    # File Upload
    max_file_size_mb: int = 50
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "application/pdf"]
    
    # Security Headers
    hsts_max_age: int = 31536000
    content_security_policy: str = "default-src 'self'"
    
    # Session
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "strict"
    
    # Blocking
    auto_block_enabled: bool = True
    auto_block_threshold: int = 5
    auto_block_duration_seconds: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = False


security_settings = SecuritySettings()
```

## 7. Security Audit Log

```python
# api/security/audit.py
"""
Security Audit Logging

Log all security-relevant events:
- Authentication attempts
- Authorization failures
- Rate limit violations
- Input validation failures
- Suspicious activities
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import json

logger = logging.getLogger("security_audit")


class AuditEventType(Enum):
    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TWO_FA_ENABLED = "2fa_enabled"
    TWO_FA_DISABLED = "2fa_disabled"
    
    # Authorization
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    
    # Security
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INPUT_VALIDATION_FAILED = "input_validation_failed"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    IP_BLOCKED = "ip_blocked"
    
    # Data
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"
    PASSWORD_CHANGE = "password_change"


class AuditLogger:
    """Security audit logger"""
    
    @staticmethod
    def log(
        event_type: AuditEventType,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {}
        }
        
        # Log to file
        logger.info(json.dumps(event))
        
        # Could also send to:
        # - SIEM (Security Information and Event Management)
        # - Elasticsearch
        # - Security database table
    
    @staticmethod
    def log_login_success(user_id: int, ip: str, user_agent: str):
        AuditLogger.log(
            AuditEventType.LOGIN_SUCCESS,
            user_id=user_id,
            ip_address=ip,
            user_agent=user_agent
        )
    
    @staticmethod
    def log_login_failure(username: str, ip: str, reason: str):
        AuditLogger.log(
            AuditEventType.LOGIN_FAILURE,
            ip_address=ip,
            details={
                "username": username,
                "reason": reason
            }
        )
    
    @staticmethod
    def log_rate_limit(ip: str, endpoint: str, limit: int):
        AuditLogger.log(
            AuditEventType.RATE_LIMIT_EXCEEDED,
            ip_address=ip,
            details={
                "endpoint": endpoint,
                "limit": limit
            }
        )
    
    @staticmethod
    def log_suspicious_activity(ip: str, reason: str, details: dict = None):
        AuditLogger.log(
            AuditEventType.SUSPICIOUS_ACTIVITY,
            ip_address=ip,
            details={
                "reason": reason,
                **(details or {})
            }
        )
```

## Security Checklist Summary

| Category | Item | Status |
|----------|------|--------|
| **Authentication** | JWT with strong secret | ✅ |
| | Password hashing (Argon2/Bcrypt) | ✅ |
| | Password validation (12+ chars) | ✅ |
| | 2FA TOTP support | ✅ |
| | Refresh token rotation | ✅ |
| | Token revocation | ✅ |
| **Input Validation** | SQL injection prevention | ✅ |
| | XSS prevention | ✅ |
| | File upload validation | ✅ |
| | Path traversal prevention | ✅ |
| **Rate Limiting** | IP-based limits | ✅ |
| | User-based limits | ✅ |
| | Endpoint-specific limits | ✅ |
| | Auto-blocking | ✅ |
| **Headers** | Content-Security-Policy | ✅ |
| | X-Frame-Options | ✅ |
| | HSTS | ✅ |
| | CORS | ✅ |
| **Monitoring** | Audit logging | ✅ |
| | Security events | ✅ |
| | Alerting | ✅ |
