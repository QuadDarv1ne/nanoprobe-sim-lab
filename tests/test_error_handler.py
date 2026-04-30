#!/usr/bin/env python3
"""
Тесты для улучшенной системы обработки ошибок
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.core.error_handler import ErrorInfo, ErrorSeverity


def test_error_info_serialization():
    """Тест сериализации ErrorInfo"""
    print("Тест сериализации ErrorInfo...")

    error = ErrorInfo(
        timestamp=datetime.now(timezone.utc),
        severity=ErrorSeverity.WARNING,
        message="Test message",
        exception_type="TestException",
        exception_message="Test",
        traceback_info="Traceback info",
        component="test",
        user_context={"user": "test_user"},
        error_id="test_123",
    )

    # Сериализация
    data = error.to_dict()

    assert "timestamp" in data
    assert "severity" in data
    assert "message" in data
    assert "error_id" in data
    assert "resolved" in data

    # Десериализация
    restored = ErrorInfo.from_dict(data)

    assert restored.message == error.message
    assert restored.error_id == error.error_id
    assert restored.severity == error.severity

    print("✓ Сериализация ErrorInfo: PASS")


def test_error_severity_enum():
    """Тест Enum уровней ошибок"""
    print("Тест ErrorSeverity Enum...")

    assert ErrorSeverity.DEBUG.value == 10
    assert ErrorSeverity.INFO.value == 20
    assert ErrorSeverity.WARNING.value == 30
    assert ErrorSeverity.ERROR.value == 40
    assert ErrorSeverity.CRITICAL.value == 50

    print("✓ ErrorSeverity Enum: PASS")


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ СИСТЕМЫ ОБРАБОТКИ ОШИБОК")
    print("=" * 60)

    tests = [
        test_error_info_serialization,
        test_error_severity_enum,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAIL - {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"ИТОГИ: {passed}/{len(tests)} тестов пройдено ({passed/len(tests)*100:.1f}%)")

    if passed == len(tests):
        print("🎉 Все тесты пройдены!")
        return 0
    else:
        print(f"❌ {failed} тест(а) провалено")
        return 1


if __name__ == "__main__":
    sys.exit(main())
