"""
2FA (Two-Factor Authentication) для Nanoprobe Sim Lab
TOTP (Time-based One-Time Password) интеграция с Google Authenticator
"""

import pyotp
import qrcode
import base64
import secrets
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class TwoFactorAuth:
    """
    Менеджер двухфакторной аутентификации
    Поддерживает TOTP (Google Authenticator, Authy, etc.)
    """

    def __init__(self, storage_path: str = "data/2fa_secrets.json"):
        """
        Инициализация 2FA менеджера

        Args:
            storage_path: Путь к файлу хранения секретов
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._secrets: Dict[str, dict] = {}
        self._load_secrets()

    def _load_secrets(self):
        """Загрузка секретов из файла"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self._secrets = json.load(f)
                logger.info(f"Loaded 2FA secrets for {len(self._secrets)} users")
            except Exception as e:
                logger.error(f"Failed to load 2FA secrets: {e}")
                self._secrets = {}

    def _save_secrets(self):
        """Сохранение секретов в файл"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._secrets, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save 2FA secrets: {e}")

    def setup_2fa(self, username: str, user_email: str) -> Tuple[str, str]:
        """
        Настройка 2FA для пользователя

        Args:
            username: Имя пользователя
            user_email: Email пользователя

        Returns:
            Tuple(secret, provisioning_uri)
        """
        # Генерация случайного секрета
        secret = pyotp.random_base32()

        # Создание TOTP объекта
        totp = pyotp.TOTP(secret)

        # Создание provisioning URI для QR кода
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name="Nanoprobe Sim Lab"
        )

        # Сохранение секрета
        self._secrets[username] = {
            'secret': secret,
            'enabled': False,  # Включается после верификации
            'created_at': str(pyotp.datetime.now())
        }
        self._save_secrets()

        logger.info(f"2FA setup initiated for user: {username}")

        return secret, provisioning_uri

    def verify_2fa_setup(self, username: str, otp_code: str) -> bool:
        """
        Верификация настройки 2FA

        Args:
            username: Имя пользователя
            otp_code: OTP код из приложения

        Returns:
            bool: Успешность верификации
        """
        if username not in self._secrets:
            logger.warning(f"2FA setup not found for user: {username}")
            return False

        secret = self._secrets[username]['secret']
        totp = pyotp.TOTP(secret)

        # Проверка OTP кода
        if totp.verify(otp_code, valid_window=1):
            self._secrets[username]['enabled'] = True
            self._secrets[username]['verified_at'] = str(pyotp.datetime.now())
            self._save_secrets()
            logger.info(f"2FA verified and enabled for user: {username}")
            return True

        logger.warning(f"2FA verification failed for user: {username}")
        return False

    def verify_2fa(self, username: str, otp_code: str) -> bool:
        """
        Верификация 2FA кода при входе

        Args:
            username: Имя пользователя
            otp_code: OTP код из приложения

        Returns:
            bool: Успешность верификации
        """
        if username not in self._secrets:
            # 2FA не настроена для пользователя
            return True  # Разрешаем вход без 2FA

        if not self._secrets[username].get('enabled', False):
            # 2FA настроена но не включена
            return True

        secret = self._secrets[username]['secret']
        totp = pyotp.TOTP(secret)

        # Проверка OTP кода с окном в 1 период (30 секунд)
        if totp.verify(otp_code, valid_window=1):
            logger.info(f"2FA verified for user: {username}")
            return True

        logger.warning(f"2FA verification failed for user: {username}")
        return False

    def is_2fa_enabled(self, username: str) -> bool:
        """
        Проверка включена ли 2FA для пользователя

        Args:
            username: Имя пользователя

        Returns:
            bool: Статус 2FA
        """
        if username not in self._secrets:
            return False
        return self._secrets[username].get('enabled', False)

    def disable_2fa(self, username: str, otp_code: str) -> bool:
        """
        Отключение 2FA

        Args:
            username: Имя пользователя
            otp_code: Текущий OTP код для подтверждения

        Returns:
            bool: Успешность отключения
        """
        if username not in self._secrets:
            return False

        # Требуется верификация перед отключением
        if not self.verify_2fa(username, otp_code):
            logger.warning(f"Failed to disable 2FA for {username}: invalid OTP")
            return False

        del self._secrets[username]
        self._save_secrets()
        logger.info(f"2FA disabled for user: {username}")
        return True

    def generate_backup_codes(self, username: str, count: int = 10) -> list:
        """
        Генерация резервных кодов

        Args:
            username: Имя пользователя
            count: Количество кодов

        Returns:
            list: Список резервных кодов
        """
        if username not in self._secrets:
            return []

        backup_codes = [secrets.token_hex(4) for _ in range(count)]

        if 'backup_codes' not in self._secrets[username]:
            self._secrets[username]['backup_codes'] = []

        self._secrets[username]['backup_codes'] = backup_codes
        self._secrets[username]['backup_codes_generated_at'] = str(pyotp.datetime.now())
        self._save_secrets()

        logger.info(f"Generated {count} backup codes for user: {username}")
        return backup_codes

    def verify_backup_code(self, username: str, backup_code: str) -> bool:
        """
        Верификация резервного кода

        Args:
            username: Имя пользователя
            backup_code: Резервный код

        Returns:
            bool: Успешность верификации
        """
        if username not in self._secrets:
            return False

        backup_codes = self._secrets[username].get('backup_codes', [])

        if backup_code in backup_codes:
            # Удаление использованного кода
            backup_codes.remove(backup_code)
            self._secrets[username]['backup_codes'] = backup_codes
            self._secrets[username]['last_backup_code_used_at'] = str(pyotp.datetime.now())
            self._save_secrets()
            logger.info(f"Backup code used for user: {username}")
            return True

        return False

    def get_qr_code_image(self, provisioning_uri: str) -> bytes:
        """
        Генерация QR кода для настройки 2FA

        Args:
            provisioning_uri: URI для provisioning

        Returns:
            bytes: PNG изображение QR кода
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Конвертация в bytes
        import io
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()


# Singleton instance
_2fa_instance: Optional[TwoFactorAuth] = None


def get_2fa_manager() -> TwoFactorAuth:
    """Получение экземпляра 2FA менеджера"""
    global _2fa_instance
    if _2fa_instance is None:
        _2fa_instance = TwoFactorAuth()
    return _2fa_instance
