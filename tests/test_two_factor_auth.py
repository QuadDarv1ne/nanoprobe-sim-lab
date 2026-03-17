#!/usr/bin/env python
"""
Unit тесты для Two-Factor Authentication

Тестирование 2FA TOTP (Google Authenticator)
"""

import sys
import os
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security.two_factor_auth import TwoFactorAuth
import pyotp


class TestTwoFactorAuthInit:
    """Тесты инициализации 2FA"""

    def test_init_default_storage(self):
        """Тест инициализации с хранилищем по умолчанию"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            assert totp.storage_path == Path(storage_path)
            assert totp._secrets == {}
        print("  [PASS] Init default storage")

    def test_init_creates_directory(self):
        """Тест создания директории при инициализации"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "subdir", "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            assert totp.storage_path.parent.exists()
        print("  [PASS] Init creates directory")


class TestTwoFactorAuthSetup:
    """Тесты настройки 2FA"""

    def test_setup_2fa_returns_secret_and_uri(self):
        """Тест настройки 2FA возвращает secret и URI"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            secret, uri = totp.setup_2fa("testuser", "test@example.com")
            
            assert isinstance(secret, str)
            assert len(secret) > 0
            assert isinstance(uri, str)
            assert "test@example.com" in uri
            assert "Nanoprobe Sim Lab" in uri
        print("  [PASS] Setup 2FA returns secret and URI")

    def test_setup_2fa_generates_unique_secrets(self):
        """Тест генерации уникальных секретов для разных пользователей"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            secret1, _ = totp.setup_2fa("user1", "user1@example.com")
            secret2, _ = totp.setup_2fa("user2", "user2@example.com")
            
            assert secret1 != secret2
        print("  [PASS] Setup 2FA generates unique secrets")

    def test_setup_2fa_persists_secret(self):
        """Тест сохранения секрета в хранилище"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            
            # Проверяем, что секрет сохранён
            assert "testuser" in totp._secrets
            assert "secret" in totp._secrets["testuser"]
        print("  [PASS] Setup 2FA persists secret")


class TestTwoFactorAuthVerification:
    """Тесты верификации 2FA"""

    def test_verify_2fa_valid_code(self):
        """Тест верификации правильного кода"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            # Настраиваем 2FA
            secret, _ = totp.setup_2fa("testuser", "test@example.com")
            
            # Генерируем правильный код
            totp_obj = pyotp.TOTP(secret)
            code = totp_obj.now()
            
            # Верифицируем
            is_valid = totp.verify_2fa("testuser", code)
            
            assert is_valid is True
        print("  [PASS] Verify 2FA valid code")

    def test_verify_2fa_invalid_code(self):
        """Тест верификации неправильного кода"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            
            # Неправильный код
            is_valid = totp.verify_2fa("testuser", "000000")
            
            assert is_valid is False
        print("  [PASS] Verify 2FA invalid code")

    def test_verify_2fa_nonexistent_user(self):
        """Тест верификации для несуществующего пользователя"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            is_valid = totp.verify_2fa("nonexistent", "123456")
            
            assert is_valid is False
        print("  [PASS] Verify 2FA nonexistent user")

    def test_verify_2fa_expired_code(self):
        """Тест верификации устаревшего кода"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            secret, _ = totp.setup_2fa("testuser", "test@example.com")
            
            # Генерируем код
            totp_obj = pyotp.TOTP(secret)
            code = totp_obj.now()
            
            # Ждём 31 секунду (код устаревает)
            time.sleep(31)
            
            # Код должен быть невалидным
            is_valid = totp.verify_2fa("testuser", code)
            
            assert is_valid is False
        print("  [PASS] Verify 2FA expired code")


class TestTwoFactorAuthBackupCodes:
    """Тесты резервных кодов 2FA"""

    def test_generate_backup_codes(self):
        """Тест генерации резервных кодов"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            codes = totp.generate_backup_codes("testuser")
            
            assert isinstance(codes, list)
            assert len(codes) == 10
            assert all(isinstance(code, str) for code in codes)
            assert all(len(code) == 8 for code in codes)
        print("  [PASS] Generate backup codes")

    def test_generate_backup_codes_unique(self):
        """Тест уникальности резервных кодов"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            codes = totp.generate_backup_codes("testuser")
            
            # Все коды уникальны
            assert len(codes) == len(set(codes))
        print("  [PASS] Generate backup codes unique")

    def test_verify_backup_code(self):
        """Тест верификации резервного кода"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            codes = totp.generate_backup_codes("testuser")
            
            # Верифицируем резервный код
            is_valid = totp.verify_backup_code("testuser", codes[0])
            
            assert is_valid is True
        print("  [PASS] Verify backup code")

    def test_verify_backup_code_consumed(self):
        """Тест использования резервного кода (одноразовый)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            codes = totp.generate_backup_codes("testuser")
            
            # Первое использование - успешно
            assert totp.verify_backup_code("testuser", codes[0]) is True
            
            # Повторное использование - неудачно
            assert totp.verify_backup_code("testuser", codes[0]) is False
        print("  [PASS] Verify backup code consumed")

    def test_verify_backup_code_invalid(self):
        """Тест верификации неправильного резервного кода"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            
            is_valid = totp.verify_backup_code("testuser", "INVALID8")
            
            assert is_valid is False
        print("  [PASS] Verify backup code invalid")


class TestTwoFactorAuthDisable:
    """Тесты отключения 2FA"""

    def test_disable_2fa(self):
        """Тест отключения 2FA"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            
            # Отключаем
            result = totp.disable_2fa("testuser")
            
            assert result is True
            assert "testuser" not in totp._secrets
        print("  [PASS] Disable 2FA")

    def test_disable_2fa_nonexistent_user(self):
        """Тест отключения 2FA для несуществующего пользователя"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            result = totp.disable_2fa("nonexistent")
            
            assert result is False
        print("  [PASS] Disable 2FA nonexistent user")


class TestTwoFactorAuthStatus:
    """Тесты статуса 2FA"""

    def test_is_2fa_enabled_true(self):
        """Тест статуса 2FA включён"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            
            assert totp.is_2fa_enabled("testuser") is True
        print("  [PASS] Is 2FA enabled true")

    def test_is_2fa_enabled_false(self):
        """Тест статуса 2FA выключен"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            assert totp.is_2fa_enabled("nonexistent") is False
        print("  [PASS] Is 2FA enabled false")

    def test_get_2fa_status(self):
        """Тест получения полной информации о статусе"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            totp = TwoFactorAuth(storage_path)
            
            totp.setup_2fa("testuser", "test@example.com")
            
            status = totp.get_2fa_status("testuser")
            
            assert status["enabled"] is True
            assert "secret" in status
            assert "backup_codes" in status or "has_backup_codes" in status
        print("  [PASS] Get 2FA status")


class TestTwoFactorAuthPersistence:
    """Тесты сохранения/загрузки 2FA"""

    def test_persistence_save_and_load(self):
        """Тест сохранения и загрузки секретов"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "2fa_secrets.json")
            
            # Создаём и настраиваем
            totp1 = TwoFactorAuth(storage_path)
            totp1.setup_2fa("testuser", "test@example.com")
            
            # Создаём новый экземпляр (должен загрузить сохранённые данные)
            totp2 = TwoFactorAuth(storage_path)
            
            assert "testuser" in totp2._secrets
            assert totp2._secrets["testuser"]["secret"] == totp1._secrets["testuser"]["secret"]
        print("  [PASS] Persistence save and load")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Two-Factor Authentication Unit Tests")
    print("=" * 60)
    
    test_classes = [
        TestTwoFactorAuthInit,
        TestTwoFactorAuthSetup,
        TestTwoFactorAuthVerification,
        TestTwoFactorAuthBackupCodes,
        TestTwoFactorAuthDisable,
        TestTwoFactorAuthStatus,
        TestTwoFactorAuthPersistence,
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
