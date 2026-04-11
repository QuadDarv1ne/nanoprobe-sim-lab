# Comprehensive Rate Limiting Guide

## Обзор

Rate limiting защищает API от злоупотреблений, обеспечивает справедливое распределение ресурсов и предотвращает DDoS атаки.

## Архитектура Rate Limiting

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Flow                              │
├─────────────────────────────────────────────────────────────────┤
│  Client → [Rate Limit Middleware] → [Auth] → [Endpoint]         │
│              │                    │           │                  │
│              ▼                    ▼           ▼                  │
│         Redis Counter      User/Token    Business Logic         │
│         (Sliding Window)   Extraction                         │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Установка зависимостей

```bash
pip install slowapi redis aioredis
```

## 2. Конфигурация Rate Limits

```python
# config/rate_limit_config.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RateLimitConfig:
    """Конфигурация rate limits для разных типов пользователей и эндпоинтов"""

    # Global limits (по IP)
    GLOBAL_PER_MINUTE: int = 60
    GLOBAL_PER_HOUR: int = 1000

    # Anonymous users
    ANONYMOUS_PER_MINUTE: int = 20
    ANONYMOUS_PER_HOUR: int = 200
    ANONYMOUS_PER_DAY: int = 500

    # Authenticated users
    AUTHENTICATED_PER_MINUTE: int = 60
    AUTHENTICATED_PER_HOUR: int = 1000
    AUTHENTICATED_PER_DAY: int = 10000

    # Premium/API Key users
    PREMIUM_PER_MINUTE: int = 120
    PREMIUM_PER_HOUR: int = 5000
    PREMIUM_PER_DAY: int = 50000

    # Endpoint-specific limits
    ENDPOINT_LIMITS: Dict[str, str] = None

    def __post_init__(self):
        self.ENDPOINT_LIMITS = {
            # Authentication endpoints (stricter)
            "/api/v1/auth/login": "5/minute;20/hour",
            "/api/v1/auth/refresh": "10/minute;50/hour",
            "/api/v1/auth/2fa/*": "3/minute;10/hour",

            # NASA API (respect upstream limits)
            "/api/v1/nasa/*": "30/minute;500/hour",

            # SSTV endpoints
            "/api/v1/sstv/*": "30/minute;500/hour",

            # AI/ML endpoints (expensive)
            "/api/v1/analysis": "10/minute;100/hour",
            "/api/v1/ml/*": "5/minute;50/hour",

            # GraphQL
            "/api/v1/graphql": "30/minute;500/hour",

            # WebSocket connections
            "/ws/*": "10/minute",

            # Default
            "default": "60/minute;1000/hour"
        }

    # Whitelist (no rate limiting)
    WHITELIST_PATHS: list = None

    def __post_init__(self):
        if self.WHITELIST_PATHS is None:
            self.WHITELIST_PATHS = [
                "/health",
                "/health/detailed",
                "/metrics",
                "/docs",
                "/redoc",
                "/openapi.json",
                "/favicon.ico",
            ]

    # Whitelist IPs (internal services)
    WHITELIST_IPS: list = None

    def __post_init__(self):
        if self.WHITELIST_IPS is None:
            self.WHITELIST_IPS = [
                "127.0.0.1",
                "::1",
                "10.0.0.0/8",      # Internal network
                "172.16.0.0/12",   # Docker
                "192.168.0.0/16",  # Local network
            ]

rate_limit_config = RateLimitConfig()
```

## 3. Redis-based Rate Limiter

```python
# utils/rate_limiter.py
import time
import asyncio
import hashlib
from typing import Optional, Callable, Tuple
from datetime import datetime
import logging
from functools import wraps
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Sliding Window Rate Limiter с Redis backend.

    Features:
    - Sliding window algorithm (точный подсчёт)
    - Multiple time windows (minute, hour, day)
    - User-based и IP-based limiting
    - Automatic cleanup
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "ratelimit:",
        key_func: Optional[Callable] = None
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, identifier: str, window: str) -> str:
        """Генерация ключа для Redis"""
        return f"{self.prefix}{identifier}:{window}"

    async def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int,
        window_name: str = "default"
    ) -> Tuple[bool, dict]:
        """
        Проверка и запись запроса.

        Returns:
            (is_allowed, info) где info содержит:
            - remaining: остаток запросов
            - reset_at: timestamp сброса
            - retry_after: секунд до повторной попытки (если заблокирован)
        """
        key = self._default_key_func(identifier, window_name)
        now = time.time()
        window_start = now - window_seconds

        pipe = self.redis.pipeline()

        # Удаляем старые записи (за пределами окна)
        pipe.zremrangebyscore(key, 0, window_start)

        # Считаем текущие запросы
        pipe.zcard(key)

        # Добавляем текущий запрос
        pipe.zadd(key, {str(now): now})

        # Устанавливаем TTL
        pipe.expire(key, window_seconds + 1)

        results = await pipe.execute()
        current_count = results[1]

        remaining = max(0, max_requests - current_count - 1)
        reset_at = int(now + window_seconds)

        if current_count >= max_requests:
            # Получаем самый старый запрос для расчёта retry_after
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = int(oldest[0][1] + window_seconds - now) + 1
            else:
                retry_after = window_seconds

            return False, {
                "remaining": 0,
                "reset_at": reset_at,
                "retry_after": retry_after,
                "limit": max_requests,
                "window": window_name
            }

        return True, {
            "remaining": remaining,
            "reset_at": reset_at,
            "retry_after": 0,
            "limit": max_requests,
            "window": window_name
        }

    async def check_multiple_windows(
        self,
        identifier: str,
        limits: list  # [(max_requests, window_seconds, window_name), ...]
    ) -> Tuple[bool, list]:
        """
        Проверка нескольких окон одновременно.

        Args:
            limits: [(60, 60, "minute"), (1000, 3600, "hour")]
        """
        results = []
        is_allowed = True

        for max_requests, window_seconds, window_name in limits:
            allowed, info = await self.is_allowed(
                identifier, max_requests, window_seconds, window_name
            )
            results.append(info)
            if not allowed:
                is_allowed = False

        return is_allowed, results

    async def reset(self, identifier: str):
        """Сброс всех лимитов для идентификатора"""
        pattern = self._default_key_func(identifier, "*")
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)


class TokenBucketRateLimiter:
    """
    Token Bucket алгоритм для smoother rate limiting.
    Лучше подходит для burst-трафика.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "tokenbucket:"
    ):
        self.redis = redis_client
        self.prefix = prefix

    async def consume(
        self,
        identifier: str,
        bucket_size: int,
        refill_rate: float,  # tokens per second
        tokens_requested: int = 1
    ) -> Tuple[bool, dict]:
        """
        Попытка потребить токены из bucket.

        Args:
            bucket_size: Максимальное количество токенов
            refill_rate: Скорость пополнения (токенов/сек)
            tokens_requested: Сколько токенов нужно
        """
        key = f"{self.prefix}{identifier}"
        now = time.time()

        # Получаем текущее состояние
        data = await self.redis.hgetall(key)

        if data:
            tokens = float(data.get(b'tokens', bucket_size))
            last_update = float(data.get(b'last_update', now))
        else:
            tokens = bucket_size
            last_update = now

        # Рассчитываем пополнение
        elapsed = now - last_update
        tokens = min(bucket_size, tokens + elapsed * refill_rate)

        if tokens >= tokens_requested:
            tokens -= tokens_requested

            # Сохраняем состояние
            await self.redis.hset(key, mapping={
                'tokens': tokens,
                'last_update': now
            })
            await self.redis.expire(key, 86400)  # 24h TTL

            return True, {
                "remaining": int(tokens),
                "limit": bucket_size,
                "refill_rate": refill_rate
            }
        else:
            retry_after = (tokens_requested - tokens) / refill_rate

            return False, {
                "remaining": 0,
                "limit": bucket_size,
                "retry_after": int(retry_after) + 1
            }


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None
_token_bucket: Optional[TokenBucketRateLimiter] = None

async def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        from utils.redis_client import get_redis
        redis_client = await get_redis()
        _rate_limiter = RateLimiter(redis_client)
    return _rate_limiter

async def get_token_bucket() -> TokenBucketRateLimiter:
    global _token_bucket
    if _token_bucket is None:
        from utils.redis_client import get_redis
        redis_client = await get_redis()
        _token_bucket = TokenBucketRateLimiter(redis_client)
    return _token_bucket
```

## 4. FastAPI Middleware

```python
# api/middleware/rate_limit_middleware.py
from fastapi import Request, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import logging
from datetime import datetime

from config.rate_limit_config import rate_limit_config
from utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive Rate Limiting Middleware.

    Features:
    - IP-based limiting
    - User-based limiting (after auth)
    - Endpoint-specific limits
    - Graceful handling with headers
    """

    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.config = rate_limit_config

    def _get_client_ip(self, request: Request) -> str:
        """Извлечение IP клиента с учётом прокси"""
        # X-Forwarded-For (через nginx/CDN)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # X-Real-IP (nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def _get_identifier(self, request: Request) -> str:
        """
        Получение идентификатора для rate limiting.
        Приоритет: User ID > API Key > IP
        """
        # Authenticated user
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"

        # API Key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key[:16]}"  # Truncate for security

        # IP address
        ip = self._get_client_ip(request)
        return f"ip:{ip}"

    def _get_limits_for_path(self, path: str) -> list:
        """Получение лимитов для конкретного пути"""
        # Check specific endpoint limits
        for pattern, limit_str in self.config.ENDPOINT_LIMITS.items():
            if self._match_pattern(path, pattern):
                return self._parse_limit_string(limit_str)

        # Default limits
        return self._parse_limit_string(self.config.ENDPOINT_LIMITS.get("default", "60/minute;1000/hour"))

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching with wildcards"""
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        return path == pattern

    def _parse_limit_string(self, limit_str: str) -> list:
        """
        Parse limit string like "60/minute;1000/hour"
        Returns: [(60, 60, "minute"), (1000, 3600, "hour")]
        """
        limits = []
        windows = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }

        for part in limit_str.split(";"):
            part = part.strip()
            if "/" in part:
                count, window = part.split("/")
                count = int(count)
                window_seconds = windows.get(window.lower(), 60)
                limits.append((count, window_seconds, window.lower()))

        return limits

    def _is_whitelisted(self, request: Request) -> bool:
        """Check if request should bypass rate limiting"""
        path = request.url.path

        # Whitelisted paths
        if path in self.config.WHITELIST_PATHS:
            return True

        # Whitelisted IPs
        client_ip = self._get_client_ip(request)
        if client_ip in self.config.WHITELIST_IPS:
            return True

        return False

    async def dispatch(self, request: Request, call_next):
        # Skip whitelisted
        if self._is_whitelisted(request):
            return await call_next(request)

        # Get rate limiter
        limiter = await get_rate_limiter()
        identifier = self._get_identifier(request)
        limits = self._get_limits_for_path(request.url.path)

        # Check rate limits
        is_allowed, results = await limiter.check_multiple_windows(identifier, limits)

        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(results[0]["limit"]),
            "X-RateLimit-Remaining": str(min(r["remaining"] for r in results)),
            "X-RateLimit-Reset": str(max(r["reset_at"] for r in results)),
        }

        if not is_allowed:
            # Find the most restrictive limit
            blocked = next(r for r in results if r["remaining"] == 0)

            headers["Retry-After"] = str(blocked["retry_after"])
            headers["X-RateLimit-Window"] = blocked["window"]

            logger.warning(
                f"Rate limit exceeded for {identifier} "
                f"path={request.url.path} "
                f"window={blocked['window']}"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests",
                        "details": {
                            "retry_after": blocked["retry_after"],
                            "window": blocked["window"],
                            "limit": blocked["limit"]
                        }
                    }
                },
                headers=headers
            )

        # Process request
        response = await call_next(request)

        # Add headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response
```

## 5. Decorator-based Rate Limiting

```python
# utils/decorators/rate_limit.py
from functools import wraps
from fastapi import HTTPException, Request
from typing import Optional, Callable
import logging

from utils.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

def rate_limit(
    max_requests: int = 60,
    window_seconds: int = 60,
    key_func: Optional[Callable] = None,
    error_message: str = "Too many requests"
):
    """
    Decorator for rate limiting specific endpoints.

    Usage:
        @rate_limit(max_requests=10, window_seconds=60)
        async def my_endpoint(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                request = kwargs.get('request')

            if request is None:
                return await func(*args, **kwargs)

            # Get identifier
            if key_func:
                identifier = key_func(request)
            else:
                identifier = f"ip:{request.client.host if request.client else 'unknown'}"

            # Check rate limit
            limiter = await get_rate_limiter()
            allowed, info = await limiter.is_allowed(
                identifier, max_requests, window_seconds, "endpoint"
            )

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": error_message,
                        "retry_after": info["retry_after"]
                    },
                    headers={
                        "Retry-After": str(info["retry_after"]),
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0"
                    }
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def rate_limit_by_user(
    max_requests: int = 100,
    window_seconds: int = 60
):
    """Rate limit by authenticated user ID"""
    def key_func(request: Request) -> str:
        user = getattr(request.state, "user", None)
        if user:
            return f"user:{user.id}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    return rate_limit(max_requests, window_seconds, key_func)


def rate_limit_by_api_key(
    max_requests: int = 1000,
    window_seconds: int = 60
):
    """Rate limit by API key"""
    def key_func(request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key[:16]}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    return rate_limit(max_requests, window_seconds, key_func)
```

## 6. Usage in API Routes

```python
# api/routes/auth.py
from fastapi import APIRouter, Request, Depends
from utils.decorators.rate_limit import rate_limit, rate_limit_by_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login")
@rate_limit(max_requests=5, window_seconds=60, error_message="Too many login attempts")
async def login(request: Request, credentials: LoginCredentials):
    """Login endpoint with strict rate limiting"""
    # ... login logic
    pass

@router.post("/refresh")
@rate_limit(max_requests=10, window_seconds=60)
async def refresh_token(request: Request, refresh_token: str):
    """Token refresh with moderate rate limiting"""
    # ... refresh logic
    pass

# api/routes/analysis.py
@router.post("/analyze")
@rate_limit_by_user(max_requests=10, window_seconds=60)
async def analyze_image(
    request: Request,
    file: UploadFile,
    current_user: User = Depends(get_current_user)
):
    """AI analysis endpoint with user-based rate limiting"""
    # ... analysis logic
    pass
```

## 7. Rate Limit Monitoring Dashboard

```python
# api/routes/rate_limit_admin.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from datetime import datetime, timedelta

from utils.rate_limiter import get_rate_limiter
from api.dependencies import get_admin_user

router = APIRouter(prefix="/admin/rate-limits", tags=["Admin"])

@router.get("/stats")
async def get_rate_limit_stats(admin = Depends(get_admin_user)):
    """Get rate limiting statistics"""
    limiter = await get_rate_limiter()

    # Get all rate limit keys
    keys = await limiter.redis.keys("ratelimit:*")

    stats = {
        "total_keys": len(keys),
        "by_window": {},
        "top_consumers": []
    }

    # Aggregate by window
    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        parts = key_str.split(":")

        if len(parts) >= 3:
            identifier = parts[1]
            window = parts[2]

            count = await limiter.redis.zcard(key)

            if window not in stats["by_window"]:
                stats["by_window"][window] = 0
            stats["by_window"][window] += count

    return stats

@router.get("/violations")
async def get_rate_limit_violations(
    hours: int = 24,
    admin = Depends(get_admin_user)
):
    """Get rate limit violations log"""
    # This would typically be stored in a separate log
    # For now, return a placeholder
    return {
        "period_hours": hours,
        "violations": [
            # Would contain actual violation data
        ]
    }

@router.delete("/reset/{identifier}")
async def reset_rate_limit(
    identifier: str,
    admin = Depends(get_admin_user)
):
    """Reset rate limits for a specific identifier"""
    limiter = await get_rate_limiter()
    await limiter.reset(identifier)
    return {"status": "reset", "identifier": identifier}
```

## 8. Frontend Rate Limit Handling

```typescript
// frontend/src/lib/apiClient.ts
import axios, { AxiosError, AxiosInstance } from 'axios';

interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
  retryAfter?: number;
}

class ApiClient {
  private client: AxiosInstance;
  private rateLimitInfo: RateLimitInfo | null = null;
  private requestQueue: Array<() => void> = [];
  private isPaused = false;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Response interceptor for rate limit handling
    this.client.interceptors.response.use(
      (response) => {
        // Extract rate limit headers
        this.rateLimitInfo = {
          limit: parseInt(response.headers['x-ratelimit-limit'] || '0'),
          remaining: parseInt(response.headers['x-ratelimit-remaining'] || '0'),
          reset: parseInt(response.headers['x-ratelimit-reset'] || '0'),
        };

        return response;
      },
      async (error: AxiosError) => {
        if (error.response?.status === 429) {
          const retryAfter = parseInt(
            error.response.headers['retry-after'] || '60'
          );

          console.warn(`Rate limited. Retrying after ${retryAfter}s`);

          // Emit event for UI
          window.dispatchEvent(new CustomEvent('rate-limited', {
            detail: { retryAfter }
          }));

          // Auto-retry after delay
          if (retryAfter <= 60) {
            await this.delay(retryAfter * 1000);
            return this.client.request(error.config!);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getRateLimitInfo(): RateLimitInfo | null {
    return this.rateLimitInfo;
  }

  async get<T>(url: string, params?: object): Promise<T> {
    const response = await this.client.get<T>(url, { params });
    return response.data;
  }

  async post<T>(url: string, data?: object): Promise<T> {
    const response = await this.client.post<T>(url, data);
    return response.data;
  }
}

export const apiClient = new ApiClient();
```

## 9. Rate Limit UI Component

```typescript
// frontend/src/components/RateLimitIndicator.tsx
'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/apiClient';
import { AlertCircle, Clock } from 'lucide-react';

export function RateLimitIndicator() {
  const [rateLimited, setRateLimited] = useState(false);
  const [retryAfter, setRetryAfter] = useState(0);

  useEffect(() => {
    const handleRateLimit = (e: CustomEvent) => {
      setRateLimited(true);
      setRetryAfter(e.detail.retryAfter);
    };

    window.addEventListener('rate-limited', handleRateLimit as EventListener);

    return () => {
      window.removeEventListener('rate-limited', handleRateLimit as EventListener);
    };
  }, []);

  useEffect(() => {
    if (retryAfter > 0) {
      const timer = setInterval(() => {
        setRetryAfter(prev => {
          if (prev <= 1) {
            setRateLimited(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [retryAfter]);

  if (!rateLimited) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50 flex items-center gap-3">
      <AlertCircle className="h-5 w-5 flex-shrink-0" />
      <div className="flex-1">
        <p className="font-medium">Превышен лимит запросов</p>
        <p className="text-sm opacity-90">
          Попробуйте через {retryAfter} сек.
        </p>
      </div>
      <Clock className="h-5 w-5 opacity-50" />
    </div>
  );
}
```

## 10. Testing Rate Limiting

```python
# tests/test_rate_limiting.py
import pytest
import asyncio
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_rate_limit_enforced():
    """Test that rate limiting is enforced"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Make requests up to the limit
        for i in range(65):  # Default is 60/minute
            response = await client.get("/api/v1/health")

            if i < 60:
                assert response.status_code == 200
            else:
                assert response.status_code == 429
                assert "Retry-After" in response.headers
                break

@pytest.mark.asyncio
async def test_rate_limit_headers():
    """Test that rate limit headers are present"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

@pytest.mark.asyncio
async def test_rate_limit_reset():
    """Test that rate limits reset after window"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # This test would need a mock Redis with time manipulation
        pass
```

## Best Practices

| Практика | Описание |
|----------|----------|
| **Graceful degradation** | Не блокировать критичные операции |
| **Clear headers** | Всегда возвращать X-RateLimit-* headers |
| **Retry-After** | Указывать время до следующей попытки |
| **Logging** | Логировать violations для анализа |
| **Monitoring** | Отслеживать паттерны злоупотребления |
| **Whitelisting** | Whitelist для внутренних сервисов |
| **Burst handling** | Token bucket для burst-трафика |
| **User feedback** | Показывать пользователю статус лимитов |
