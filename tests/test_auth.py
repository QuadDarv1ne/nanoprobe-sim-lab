#!/usr/bin/env python
"""
Unit тесты для Authentication модуля

Тестирование JWT аутентификации, refresh token rotation, password validation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes.auth import (
    validate_password_strength,
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    _store_refresh_token,
    _is_token_valid,
    _revoke_refresh_token,
)


class TestPasswordValidation:
    """Тесты валидации пароля"""

    def test_validate_password_too_short(self):
        """Тест: пароль слишком короткий"""
        is_valid, message = validate_password_strength("Short1!")
        assert is_valid is False
        assert "at least 8 characters" in message
        print("  [PASS] Password too short")

    def test_validate_password_too_long(self):
        """Тест: пароль слишком длинный"""
        long_password = "a" * 129
        is_valid, message = validate_password_strength(long_password)
        assert is_valid is False
        assert "exceed 128 characters" in message
        print("  [PASS] Password too long")

    def test_validate_password_no_uppercase(self):
        """Тест: нет заглавной буквы"""
        is_valid, message = validate_password_strength("password1!")
        assert is_valid is False
        assert "uppercase letter" in message
        print("  [PASS] Password no uppercase")

    def test_validate_password_no_lowercase(self):
        """Тест: нет строчной буквы"""
        is_valid, message = validate_password_strength("PASSWORD1!")
        assert is_valid is False
        assert "lowercase letter" in message
        print("  [PASS] Password no lowercase")

    def test_validate_password_no_digit(self):
        """Тест: нет цифры"""
        is_valid, message = validate_password_strength("Password!")
        assert is_valid is False
        assert "digit" in message
        print("  [PASS] Password no digit")

    def test_validate_password_no_special(self):
        """Тест: нет специального символа"""
        is_valid, message = validate_password_strength("Password1")
        assert is_valid is False
        assert "special character" in message
        print("  [PASS] Password no special")

    def test_validate_password_strong(self):
        """Тест: надёжный пароль"""
        is_valid, message = validate_password_strength("StrongPass123!")
        assert is_valid is True
        assert message == ""
        print("  [PASS] Password strong")

    def test_validate_password_edge_cases(self):
        """Тест: граничные случаи"""
        # Ровно 8 символов
        is_valid, _ = validate_password_strength("Pass123!")
        assert is_valid is True
        
        # Минимальный специальный символ
        is_valid, _ = validate_password_strength("Password1!")
        assert is_valid is True
        print("  [PASS] Password edge cases")


class TestPasswordHashing:
    """Тесты хеширования пароля"""

    def test_hash_password_returns_string(self):
        """Тест: хеш пароля - строка"""
        password_hash = hash_password("TestPass123!")
        assert isinstance(password_hash, str)
        assert len(password_hash) > 0
        print("  [PASS] Hash password returns string")

    def test_hash_password_different_hashes(self):
        """Тест: разные хеши для одного пароля"""
        hash1 = hash_password("SamePassword123!")
        hash2 = hash_password("SamePassword123!")
        # bcrypt использует случайную соль
        assert hash1 != hash2
        print("  [PASS] Hash password different hashes")

    def test_verify_password_correct(self):
        """Тест: проверка правильного пароля"""
        password = "CorrectPassword123!"
        password_hash = hash_password(password)
        assert verify_password(password, password_hash) is True
        print("  [PASS] Verify password correct")

    def test_verify_password_incorrect(self):
        """Тест: проверка неправильного пароля"""
        password = "Password123!"
        password_hash = hash_password(password)
        assert verify_password("WrongPassword123!", password_hash) is False
        print("  [PASS] Verify password incorrect")


class TestJWTToken:
    """Тесты JWT токенов"""

    def test_create_access_token(self):
        """Тест: создание access токена"""
        token = create_access_token(data={"sub": "testuser", "user_id": 1})

        assert isinstance(token, str)
        assert len(token) > 0

        # Декодируем и проверяем
        payload = decode_token(token)
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 1
        assert "exp" in payload
        assert payload["type"] == "access"
        print("  [PASS] Create access token")

    def test_create_refresh_token(self):
        """Тест: создание refresh токена"""
        token = create_refresh_token(data={"sub": "testuser", "user_id": 1})

        assert isinstance(token, str)

        # Декодируем и проверяем
        payload = decode_token(token)
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 1
        assert "exp" in payload
        assert "jti" in payload  # Уникальный ID для rotation
        assert payload["type"] == "refresh"
        print("  [PASS] Create refresh token")

    def test_decode_token_expired(self):
        """Тест: декодирование истёкшего токена"""
        # Создаём токен с прошлым временем
        import jwt
        from datetime import datetime, timedelta, timezone

        expired_payload = {
            "sub": "testuser",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
            "type": "access"
        }
        from api.routes.auth import JWT_SECRET, JWT_ALGORITHM
        expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        payload = decode_token(expired_token)
        # Истёкший токен декодируется с verify_exp=False
        assert payload["sub"] == "testuser"
        assert payload["type"] == "access"
        print("  [PASS] Decode token expired")

    def test_decode_token_invalid(self):
        """Тест: декодирование невалидного токена"""
        invalid_token = "invalid.token.here"
        payload = decode_token(invalid_token)
        assert payload == {"error": "invalid_token"}
        print("  [PASS] Decode token invalid")

    def test_access_token_expiration(self):
        """Тест: время жизни access токена"""
        token = create_access_token(data={"sub": "testuser", "user_id": 1})
        payload = decode_token(token)

        # Проверяем, что время экспирации установлено
        assert "exp" in payload
        print("  [PASS] Access token expiration")


class TestRefreshTokenRotation:
    """Тесты rotation refresh токенов"""

    def test_store_and_check_refresh_token(self):
        """Тест: сохранение и проверка refresh токена"""
        jti = "test_jti_12345"
        username = "testuser"
        
        # Сохраняем
        _store_refresh_token(jti, username)
        
        # Проверяем
        assert _is_token_valid(jti) is True
        print("  [PASS] Store and check refresh token")

    def test_revoke_refresh_token(self):
        """Тест: отзыв refresh токена"""
        jti = "test_jti_revoke"
        username = "testuser"
        
        # Сохраняем
        _store_refresh_token(jti, username)
        assert _is_token_valid(jti) is True
        
        # Отозываем
        _revoke_refresh_token(jti)
        
        # Проверяем - должен быть невалиден
        assert _is_token_valid(jti) is False
        print("  [PASS] Revoke refresh token")

    def test_nonexistent_token_invalid(self):
        """Тест: несуществующий токен невалиден"""
        assert _is_token_valid("nonexistent_jti") is False
        print("  [PASS] Nonexistent token invalid")


class TestTokenTypes:
    """Тесты типов токенов"""

    def test_access_token_type(self):
        """Тест: тип access токена"""
        token = create_access_token("testuser")
        payload = decode_token(token)
        assert payload["type"] == "access"
        print("  [PASS] Access token type")

    def test_refresh_token_type(self):
        """Тест: тип refresh токена"""
        token = create_refresh_token("testuser")
        payload = decode_token(token)
        assert payload["type"] == "refresh"
        print("  [PASS] Refresh token type")


class TestJWTClaims:
    """Тесты claims JWT"""

    def test_access_token_has_subject(self):
        """Тест: access токен имеет subject"""
        token = create_access_token("myuser")
        payload = decode_token(token)
        assert payload["sub"] == "myuser"
        print("  [PASS] Access token has subject")

    def test_refresh_token_has_jti(self):
        """Тест: refresh токен имеет уникальный jti"""
        token1 = create_refresh_token("user1")
        token2 = create_refresh_token("user1")
        
        payload1 = decode_token(token1)
        payload2 = decode_token(token2)
        
        # jti должен быть уникальным для каждого токена
        assert payload1["jti"] != payload2["jti"]
        print("  [PASS] Refresh token has unique jti")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Authentication Module Unit Tests")
    print("=" * 60)
    
    test_classes = [
        TestPasswordValidation,
        TestPasswordHashing,
        TestJWTToken,
        TestRefreshTokenRotation,
        TestTokenTypes,
        TestJWTClaims,
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
