"""
Circuit Breaker pattern для Nanoprobe Sim Lab
Защита от каскадных сбоев при работе с внешними сервисами
"""

import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Состояния circuit breaker"""

    CLOSED = "closed"  # Нормальная работа
    OPEN = "open"  # Сбой, запросы блокируются
    HALF_OPEN = "half_open"  # Проверка восстановления


class CircuitBreakerError(Exception):
    """Исключение circuit breaker"""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker открыт"""


class CircuitBreaker:
    """
    Circuit Breaker для защиты от каскадных сбоев

    Паттерн:
    - CLOSED: Нормальная работа, отслеживаем ошибки
    - OPEN: Превышен порог ошибок, блокируем запросы на timeout
    - HALF_OPEN: Разрешаем один тестовый запрос
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
        name: str = "default",
    ):
        """
        Инициализация circuit breaker

        Args:
            failure_threshold: Порог ошибок для открытия
            recovery_timeout: Таймаут восстановления (секунды)
            half_open_max_calls: Максимум запросов в half-open
            name: Имя circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: datetime = datetime.now(timezone.utc)
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Получение текущего состояния"""
        with self._lock:
            # Проверка автоматического перехода из OPEN в HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._last_state_change = datetime.now(timezone.utc)
                    logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Проверка возможности сброса"""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Вызов функции через circuit breaker

        Args:
            func: Функция для вызова
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы

        Returns:
            Результат вызова функции

        Raises:
            CircuitBreakerOpenError: Если circuit breaker открыт
        """
        with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Blocking request to {func.__name__}"
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Retry after {self.recovery_timeout}s"
                )

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    logger.warning(f"Circuit breaker '{self.name}' HALF_OPEN max calls reached")
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1

        # Выполнение вызова
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            logger.warning(f"Circuit breaker call failed for {self._name}: {e}")
            self._on_failure()
            raise

    def _on_success(self):
        """Обработка успешного вызова"""
        with self._lock:
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Успешный вызов в half-open — закрываем circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._last_state_change = datetime.now(timezone.utc)
                logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")

            elif self._state == CircuitState.CLOSED:
                # Сброс счетчика ошибок при успехе
                self._failure_count = 0

    def _on_failure(self):
        """Обработка неудачного вызова"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                # Провал в half-open — снова открываем circuit
                self._state = CircuitState.OPEN
                self._last_state_change = datetime.now(timezone.utc)
                logger.warning(
                    f"Circuit breaker '{self.name}' transitioned to OPEN "
                    f"(failure in half-open state)"
                )

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    # Превышен порог ошибок — открываем circuit
                    self._state = CircuitState.OPEN
                    self._last_state_change = datetime.now(timezone.utc)
                    logger.warning(
                        f"Circuit breaker '{self.name}' transitioned to OPEN "
                        f"(failure threshold {self.failure_threshold} reached)"
                    )

    def get_stats(self) -> dict:
        """Получение статистики"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": (
                    self._last_failure_time.isoformat() if self._last_failure_time else None
                ),
                "last_state_change": self._last_state_change.isoformat(),
            }

    def reset(self):
        """Сброс circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._last_state_change = datetime.now(timezone.utc)
            self._half_open_calls = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")


# Глобальное хранилище circuit breakers
_circuit_breakers: dict = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str = "default", failure_threshold: int = 5, recovery_timeout: int = 60
) -> CircuitBreaker:
    """
    Получение или создание circuit breaker

    Args:
        name: Имя circuit breaker
        failure_threshold: Порог ошибок
        recovery_timeout: Таймаут восстановления

    Returns:
        CircuitBreaker экземпляр
    """
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold, recovery_timeout=recovery_timeout, name=name
            )
        return _circuit_breakers[name]


def circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    fallback: Optional[Any] = None,
):
    """
    Декоратор circuit breaker

    Args:
        name: Имя circuit breaker
        failure_threshold: Порог ошибок
        recovery_timeout: Таймаут восстановления
        fallback: Значение по умолчанию при ошиббе

    Usage:
        @circuit_breaker(name="external_api", failure_threshold=3)
        def call_external_api():
            ...
    """

    def decorator(func: Callable) -> Callable:
        breaker = get_circuit_breaker(name, failure_threshold, recovery_timeout)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError:
                if fallback is not None:
                    logger.warning(
                        f"Circuit breaker '{name}' open, using fallback for {func.__name__}"
                    )
                    return fallback
                raise
            except Exception as e:
                logger.error(f"Circuit breaker '{name}' caught exception in {func.__name__}: {e}")
                if fallback is not None:
                    return fallback
                raise

        wrapper.circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator


def get_all_circuit_breakers_stats() -> dict:
    """Получение статистики всех circuit breakers"""
    with _breakers_lock:
        return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}


def reset_all_circuit_breakers():
    """Сброс всех circuit breakers"""
    with _breakers_lock:
        for cb in _circuit_breakers.values():
            cb.reset()


def close_all_circuit_breakers():
    """Очистка всех circuit breakers при shutdown"""
    with _breakers_lock:
        _circuit_breakers.clear()
        logger.info("All circuit breakers closed")
