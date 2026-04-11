#!/usr/bin/env python
"""
Unit тесты для Circuit Breaker

Тестирование паттерна Circuit Breaker для защиты от каскадных сбоев
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.caching.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreakerInit:
    """Тесты инициализации Circuit Breaker"""

    def test_init_default_values(self):
        """Тест инициализации со значениями по умолчанию"""
        cb = CircuitBreaker()

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.half_open_max_calls == 1
        assert cb.name == "default"
        assert cb.state == CircuitState.CLOSED
        print("  [PASS] Init default values")

    def test_init_custom_values(self):
        """Тест инициализации с кастомными значениями"""
        cb = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30, half_open_max_calls=2, name="test_breaker"
        )

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.half_open_max_calls == 2
        assert cb.name == "test_breaker"
        print("  [PASS] Init custom values")


class TestCircuitBreakerStateTransitions:
    """Тесты переходов состояний Circuit Breaker"""

    def test_initial_state_closed(self):
        """Тест начального состояния - CLOSED"""
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        print("  [PASS] Initial state CLOSED")

    def test_state_transitions_to_open_after_failures(self):
        """Тест перехода в OPEN после превышения порога ошибок"""
        cb = CircuitBreaker(failure_threshold=3)

        # Симуляция ошибок
        for i in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN
        print("  [PASS] State transitions to OPEN after failures")

    def test_state_transitions_to_half_open_after_timeout(self):
        """Тест перехода в HALF_OPEN после таймаута"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Вызываем ошибки для открытия
        for i in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

        # Ждём таймаут
        time.sleep(1.1)

        # Проверяем переход в HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        print("  [PASS] State transitions to HALF_OPEN after timeout")

    def test_state_transitions_to_closed_after_success(self):
        """Тест перехода в CLOSED после успешного вызова в HALF_OPEN"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Открываем circuit breaker
        for i in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        # Ждём таймаут
        time.sleep(1.1)

        # Успешный вызов в HALF_OPEN
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        print("  [PASS] State transitions to CLOSED after success")


class TestCircuitBreakerCall:
    """Тесты вызовов через Circuit Breaker"""

    def test_call_success(self):
        """Тест успешного вызова"""
        cb = CircuitBreaker()

        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb._success_count == 1
        print("  [PASS] Call success")

    def test_call_with_args(self):
        """Тест вызова с аргументами"""
        cb = CircuitBreaker()

        def add(a, b):
            return a + b

        result = cb.call(add, 2, 3)
        assert result == 5
        print("  [PASS] Call with args")

    def test_call_with_kwargs(self):
        """Тест вызова с именованными аргументами"""
        cb = CircuitBreaker()

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = cb.call(greet, "World", greeting="Hi")
        assert result == "Hi, World!"
        print("  [PASS] Call with kwargs")

    def test_call_failure(self):
        """Тест неудачного вызова"""
        cb = CircuitBreaker()

        def fail_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError) as exc_info:
            cb.call(fail_func)

        assert str(exc_info.value) == "Test error"
        assert cb._failure_count == 1
        print("  [PASS] Call failure")

    def test_call_open_circuit_breaker(self):
        """Тест вызова при открытом circuit breaker"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)

        # Открываем circuit breaker
        for i in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        # Попытка вызова при открытом
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "should not execute")
        print("  [PASS] Call open circuit breaker")


class TestCircuitBreakerDecorator:
    """Тесты декоратора circuit_breaker"""

    def test_decorator_success(self):
        """Тест успешного вызова через декоратор"""

        @circuit_breaker(name="test_success", failure_threshold=3)
        def success_func():
            return "decorated success"

        result = success_func()
        assert result == "decorated success"
        print("  [PASS] Decorator success")

    def test_decorator_failure(self):
        """Тест неудачного вызова через декоратор"""

        @circuit_breaker(name="test_fail", failure_threshold=2)
        def fail_func():
            raise RuntimeError("Decorated error")

        with pytest.raises(RuntimeError):
            fail_func()
        print("  [PASS] Decorator failure")


class TestCircuitBreakerStats:
    """Тесты статистики Circuit Breaker"""

    def test_get_stats(self):
        """Тест получения статистики"""
        cb = CircuitBreaker(failure_threshold=5)

        # Несколько успешных вызовов
        for i in range(3):
            cb.call(lambda: "success")

        # Несколько неудачных
        for i in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        stats = cb.get_stats()

        assert stats["name"] == "default"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2
        assert stats["success_count"] == 3
        assert "failure_threshold" in stats
        assert "recovery_timeout" in stats
        print("  [PASS] Get stats")

    def test_reset(self):
        """Тест сброса статистики"""
        cb = CircuitBreaker()

        # Вызываем ошибки
        for i in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        # Сброс
        cb.reset()

        assert cb._failure_count == 0
        assert cb._success_count == 0
        assert cb.state == CircuitState.CLOSED
        print("  [PASS] Reset")


class TestCircuitBreakerEdgeCases:
    """Тесты граничных случаев"""

    def test_rapid_success_failures(self):
        """Тест быстрых последовательных ошибок"""
        cb = CircuitBreaker(failure_threshold=5)

        # Быстрые ошибки
        for i in range(5):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN
        print("  [PASS] Rapid success failures")

    def test_half_open_single_success(self):
        """Тест одного успеха в HALF_OPEN"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Открываем
        for i in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        # Ждём
        time.sleep(1.1)

        # Один успех
        cb.call(lambda: "success")

        # Должен закрыться
        assert cb.state == CircuitState.CLOSED
        print("  [PASS] Half open single success")

    def test_concurrent_access(self):
        """Тест конкурентного доступа"""
        import threading

        cb = CircuitBreaker(failure_threshold=10)
        success_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count
            for i in range(10):
                try:
                    cb.call(lambda: "success")
                    with lock:
                        success_count += 1
                except Exception:
                    pass

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count == 50
        print("  [PASS] Concurrent access")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Circuit Breaker Unit Tests")
    print("=" * 60)

    test_classes = [
        TestCircuitBreakerInit,
        TestCircuitBreakerStateTransitions,
        TestCircuitBreakerCall,
        TestCircuitBreakerDecorator,
        TestCircuitBreakerStats,
        TestCircuitBreakerEdgeCases,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: {e}")
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("  ✅ All tests passed!")
    else:
        print(f"  ❌ {total_tests - passed_tests} tests failed")
    print("=" * 60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
