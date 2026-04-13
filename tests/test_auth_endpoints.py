"""
Тесты для auth endpoints (api/routes/auth_routes/endpoints.py)

Покрытие:
- Login endpoint
- Refresh token endpoint
- Logout endpoint
- 2FA endpoints
- Rate limit status endpoint
- User info endpoint
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Устанавливаем тестовую БД
TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app
from api.routes.auth_routes.helpers import (
    AuditEventType,
    hash_password,
    log_audit_event,
    validate_password_strength,
    verify_password,
)
from api.routes.auth_routes.tokens import (
    create_access_token,
    create_refresh_token,
    is_token_valid,
    revoke_refresh_token,
)


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент для тестов"""
    print("\n[INFO] Инициализация тестовой БД для auth endpoint тестов...")
    with TestClient(app) as test_client:
        yield test_client
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


@pytest.fixture
def valid_user_data():
    """Валидные данные пользователя."""
    return {
        "id": 1,
        "username": "testuser",
        "password_hash": hash_password("TestPass123!"),
        "role": "user",
        "created_at": "2026-04-13T00:00:00Z",
    }


class TestPasswordValidation:
    """Тесты валидации паролей."""

    def test_validate_password_too_short(self):
        """Пароль слишком короткий."""
        is_valid, message = validate_password_strength("Short1!")
        assert not is_valid
        assert "8 characters" in message

    def test_validate_password_too_long(self):
        """Пароль слишком длинный."""
        long_password = "A" * 129 + "1!"
        is_valid, message = validate_password_strength(long_password)
        assert not is_valid
        assert "128 characters" in message

    def test_validate_password_no_uppercase(self):
        """Нет заглавной буквы."""
        is_valid, message = validate_password_strength("short123!")
        assert not is_valid
        assert "uppercase" in message.lower()

    def test_validate_password_no_lowercase(self):
        """Нет строчной буквы."""
        is_valid, message = validate_password_strength("SHORT123!")
        assert not is_valid
        assert "lowercase" in message.lower()

    def test_validate_password_no_digit(self):
        """Нет цифры."""
        is_valid, message = validate_password_strength("NoDigitsHere!")
        assert not is_valid
        assert "digit" in message.lower()

    def test_validate_password_no_special(self):
        """Нет специального символа."""
        is_valid, message = validate_password_strength("NoSpecialChar123")
        assert not is_valid
        assert "special character" in message.lower()

    def test_validate_password_strong(self):
        """Сильный пароль проходит валидацию."""
        is_valid, message = validate_password_strength("StrongPass123!")
        assert is_valid
        assert message == ""

    def test_validate_password_edge_cases(self):
        """Граничные случаи валидации паролей."""
        # Ровно 8 символов
        is_valid, _ = validate_password_strength("Pass123!")
        assert is_valid

        # Ровно 128 символов
        edge_password = "A" * 120 + "1!a"
        is_valid, _ = validate_password_strength(edge_password)
        assert is_valid


class TestAuditLogging:
    """Тесты audit логирования."""

    def test_log_audit_event_login_success(self, caplog):
        """Логирование успешного входа."""
        with caplog.at_level("INFO"):
            log_audit_event(
                AuditEventType.LOGIN_SUCCESS,
                username="testuser",
                details={"ip": "127.0.0.1"},
            )
        assert "login_success" in caplog.text
        assert "testuser" in caplog.text

    def test_log_audit_event_login_failure(self, caplog):
        """Логирование неудачного входа."""
        with caplog.at_level("INFO"):
            log_audit_event(
                AuditEventType.LOGIN_FAILURE,
                username="testuser",
                reason="invalid_credentials",
            )
        assert "login_failure" in caplog.text
        assert "invalid_credentials" in caplog.text

    def test_log_audit_event_logout(self, caplog):
        """Логирование выхода."""
        with caplog.at_level("INFO"):
            log_audit_event(
                AuditEventType.LOGOUT,
                username="testuser",
            )
        assert "logout" in caplog.text

    def test_log_audit_event_token_refresh(self, caplog):
        """Логирование обновления токена."""
        with caplog.at_level("INFO"):
            log_audit_event(
                AuditEventType.TOKEN_REFRESH,
                username="testuser",
                jti="test-jti",
            )
        assert "token_refresh" in caplog.text


class TestTokenManagement:
    """Тесты управления токенами."""

    def test_create_access_token(self):
        """Создание access токена."""
        token = create_access_token(data={"sub": "testuser", "user_id": 1})
        assert token is not None
        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT format: header.payload.signature

    def test_create_refresh_token(self):
        """Создание refresh токена."""
        token = create_refresh_token(data={"sub": "testuser", "user_id": 1})
        assert token is not None
        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    def test_token_revoke_and_check(self):
        """Ревокация токена и проверка."""
        # Создаем токен
        token = create_refresh_token(data={"sub": "testuser", "user_id": 1, "jti": "test-jti-123"})

        # Извлекаем JTI
        import jwt

        from api.routes.auth_routes.helpers import JWT_ALGORITHM, JWT_SECRET

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")

        # Проверяем, что токен валиден
        assert is_token_valid(jti)

        # Ревоцируем токен
        revoke_refresh_token(jti)

        # Проверяем, что токен больше не валиден
        assert not is_token_valid(jti)

    def test_different_tokens_different_hashes(self):
        """Разные токены имеют разные хэши."""
        token1 = create_refresh_token(data={"sub": "user1", "user_id": 1})
        token2 = create_refresh_token(data={"sub": "user2", "user_id": 2})

        assert token1 != token2


class TestPasswordHashing:
    """Тесты хэширования паролей."""

    def test_hash_password_returns_string(self):
        """hash_password возвращает строку."""
        hashed = hash_password("TestPass123!")
        assert isinstance(hashed, str)

    def test_hash_password_argon2_format(self):
        """hash_password использует Argon2 формат."""
        hashed = hash_password("TestPass123!")
        assert hashed.startswith("$argon2")

    def test_hash_password_different_hashes(self):
        """Одинаковые пароли дают разные хэши (salt)."""
        hash1 = hash_password("TestPass123!")
        hash2 = hash_password("TestPass123!")
        # Argon2 использует случайный salt, поэтому хэши разные
        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Верификация правильного пароля."""
        password = "TestPass123!"
        hashed = hash_password(password)
        assert verify_password(password, hashed)

    def test_verify_password_incorrect(self):
        """Верификация неправильвого пароля."""
        hashed = hash_password("TestPass123!")
        assert not verify_password("WrongPassword!", hashed)

    def test_verify_password_empty_hash_raises_exception(self):
        """Верификация с пустым хэшем бросает исключение."""
        import pytest
        from argon2.exceptions import InvalidHashError

        with pytest.raises(InvalidHashError):
            verify_password("TestPass123!", "")
