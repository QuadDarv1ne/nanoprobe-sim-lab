# -*- coding: utf-8 -*-
"""
Rate Limiter для Nanoprobe Sim Lab API
Защита от brute force и DDoS атак
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from functools import wraps
from fastapi import HTTPException, status, Request


class RateLimiter:
    """Rate limiter с скользящим окном"""

    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        window_start = now - window_seconds

        self.requests[key] = [
            ts for ts in self.requests[key]
            if ts > window_start
        ]

        if len(self.requests[key]) >= max_requests:
            return False

        self.requests[key].append(now)
        return True

    def get_retry_after(self, key: str, max_requests: int, window_seconds: int) -> int:
        if not self.requests[key]:
            return 0
        
        oldest = min(self.requests[key])
        retry_after = int(oldest + window_seconds - time.time())
        return max(0, retry_after)


limiter = RateLimiter()


def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    Декоратор для rate limiting.
    
    Args:
        max_requests: Максимум запросов в окно
        window_seconds: Размер окна в секундах
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            key = f"{func.__name__}:{client_ip}"
            
            if not limiter.is_allowed(key, max_requests, window_seconds):
                retry_after = limiter.get_retry_after(key, max_requests, window_seconds)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Слишком много запросов",
                    headers={"Retry-After": str(retry_after)},
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
