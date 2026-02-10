# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль шифрования данных для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для шифрования и защиты чувствительных данных проекта.
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import os
from typing import Union, Tuple, Optional
from pathlib import Path
import json

class DataEncryption:
    """
    Класс для шифрования данных
    Обеспечивает шифрование и дешифрование чувствительных данных проекта с использованием современных криптографических методов.
    """


    def __init__(self, key: bytes = None):
        """
        Инициализирует шифровальщик данных

        Args:
            key: Ключ шифрования (если None, генерируется новый)
        """
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key

        self.cipher = Fernet(self.key)


    def encrypt_string(self, plaintext: str) -> str:
        """
        Шифрует строку

        Args:
            plaintext: Открытый текст для шифрования

        Returns:
            Зашифрованная строка в формате base64
        """
        encrypted_bytes = self.cipher.encrypt(plaintext.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')


    def decrypt_string(self, encrypted_data: str) -> str:
        """
        Дешифрует строку

        Args:
            encrypted_data: Зашифрованные данные в формате base64

        Returns:
            Расшифрованная строка
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')


    def encrypt_bytes(self, data: bytes) -> bytes:
        """
        Шифрует байты

        Args:
            data: Байты для шифрования

        Returns:
            Зашифрованные байты
        """
        return self.cipher.encrypt(data)


    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        """
        Дешифрует байты

        Args:
            encrypted_data: Зашифрованные байты

        Returns:
            Расшифрованные байты
        """
        return self.cipher.decrypt(encrypted_data)


    def encrypt_file(self, input_file: str, output_file: str) -> bool:
        """
        Шифрует содержимое файла

        Args:
            input_file: Входной файл
            output_file: Выходной файл (зашифрованный)

        Returns:
            True если шифрование успешно, иначе False
        """
        try:
            with open(input_file, 'rb') as infile:
                data = infile.read()

            encrypted_data = self.encrypt_bytes(data)

            with open(output_file, 'wb') as outfile:
                outfile.write(encrypted_data)

            return True
        except Exception as e:
            print(f"Ошибка шифрования файла: {e}")
            return False


    def decrypt_file(self, input_file: str, output_file: str) -> bool:
        """
        Дешифрует содержимое файла

        Args:
            input_file: Входной файл (зашифрованный)
            output_file: Выходной файл (расшифрованный)

        Returns:
            True если дешифрование успешно, иначе False
        """
        try:
            with open(input_file, 'rb') as infile:
                encrypted_data = infile.read()

            decrypted_data = self.decrypt_bytes(encrypted_data)

            with open(output_file, 'wb') as outfile:
                outfile.write(decrypted_data)

            return True
        except Exception as e:
            print(f"Ошибка дешифрования файла: {e}")
            return False


    def generate_key_from_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Генерирует ключ шифрования из пароля

        Args:
            password: Пароль
            salt: Соль (если None, генерируется новая)

        Returns:
            Кортеж (ключ, соль)
        """
        if salt is None:
            salt = os.urandom(16)  # 16 байт соли

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt

class SecureDataManager:
    """
    Класс безопасного управления данными
    Обеспечивает шифрование и безопасное хранение
    конфиденциальных данных проекта.
    """


    def __init__(self, encryption_key: bytes = None):
        """
        Инициализирует менеджер безопасных данных

        Args:
            encryption_key: Ключ шифрования
        """
        self.encryption = DataEncryption(encryption_key)
        self.secure_storage_path = Path("secure_data")
        self.secure_storage_path.mkdir(exist_ok=True)


    def store_secure_data(self, key: str, data: Union[str, dict, list], encrypt: bool = True) -> bool:
        """
        Сохраняет защищенные данные

        Args:
            key: Ключ для идентификации данных
            data: Данные для сохранения
            encrypt: Шифровать ли данные

        Returns:
            True если сохранение успешно, иначе False
        """
        try:
            # Подготавливаем данные для сохранения
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)

            # Шифруем если необходимо
            if encrypt:
                data_str = self.encryption.encrypt_string(data_str)

            # Сохраняем в файл
            file_path = self.secure_storage_path / f"{key}.enc"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data_str)

            return True
        except Exception as e:
            print(f"Ошибка сохранения защищенных данных: {e}")
            return False


    def retrieve_secure_data(self, key: str, decrypt: bool = True) -> Optional[Union[str, dict, list]]:
        """
        Получает защищенные данные

        Args:
            key: Ключ для идентификации данных
            decrypt: Дешифровать ли данные

        Returns:
            Данные или None если не найдены
        """
        try:
            file_path = self.secure_storage_path / f"{key}.enc"

            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data_str = f.read()

            # Дешифруем если необходимо
            if decrypt:
                data_str = self.encryption.decrypt_string(data_str)

            # Пытаемся десериализовать JSON
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                # Если не JSON, возвращаем как строку
                return data_str

        except Exception as e:
            print(f"Ошибка получения защищенных данных: {e}")
            return None


    def secure_delete(self, key: str) -> bool:
        """
        Безопасно удаляет защищенные данные

        Args:
            key: Ключ для идентификации данных

        Returns:
            True если удаление успешно, иначе False
        """
        try:
            file_path = self.secure_storage_path / f"{key}.enc"

            if not file_path.exists():
                return True  # Файл уже удален

            # Перезаписываем файл случайными данными перед удалением
            file_size = file_path.stat().st_size
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))

            # Удаляем файл
            file_path.unlink()

            return True
        except Exception as e:
            print(f"Ошибка безопасного удаления: {e}")
            return False


    def store_sensitive_config(self, config_data: dict) -> bool:
        """
        Сохраняет конфигурацию с чувствительными данными

        Args:
            config_data: Данные конфигурации

        Returns:
            True если сохранение успешно, иначе False
        """
        return self.store_secure_data("sensitive_config", config_data)


    def get_sensitive_config(self) -> Optional[dict]:
        """
        Получает конфигурацию с чувствительными данными

        Returns:
            Конфигурация или None если не найдена
        """
        data = self.retrieve_secure_data("sensitive_config")
        if isinstance(data, dict):
            return data
        return None


    def store_encrypted_file(self, filename: str, data: bytes) -> bool:
        """
        Сохраняет зашифрованный файл

        Args:
            filename: Имя файла
            data: Данные для сохранения

        Returns:
            True если сохранение успешно, иначе False
        """
        try:
            encrypted_data = self.encryption.encrypt_bytes(data)
            file_path = self.secure_storage_path / f"{filename}.enc"

            with open(file_path, 'wb') as f:
                f.write(encrypted_data)

            return True
        except Exception as e:
            print(f"Ошибка сохранения зашифрованного файла: {e}")
            return False


    def retrieve_encrypted_file(self, filename: str) -> Optional[bytes]:
        """
        Получает зашифрованный файл

        Args:
            filename: Имя файла

        Returns:
            Данные файла или None если не найден
        """
        try:
            file_path = self.secure_storage_path / f"{filename}.enc"

            if not file_path.exists():
                return None

            with open(file_path, 'rb') as f:
                encrypted_data = f.read()

            return self.encryption.decrypt_bytes(encrypted_data)

        except Exception as e:
            print(f"Ошибка получения зашифрованного файла: {e}")
            return None

class SecurityValidator:
    """
    Класс валидации безопасности
    Обеспечивает проверку безопасности данных и
    защиту от распространенных угроз.
    """

    @staticmethod

    def validate_input(input_data: str, max_length: int = 1000) -> bool:
        """
        Проверяет безопасность входных данных

        Args:
            input_data: Входные данные для проверки
            max_length: Максимальная длина

        Returns:
            True если данные безопасны, иначе False
        """
        # Проверяем длину
        if len(input_data) > max_length:
            return False

        # Проверяем на потенциальные SQL-инъекции
        sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', '--', ';']
        for pattern in sql_patterns:
            if pattern.upper() in input_data.upper():
                return False

        # Проверяем на XSS-атаки
        xss_patterns = ['<script', 'javascript:', 'onerror=', 'onload=', '<iframe', '<object']
        for pattern in xss_patterns:
            if pattern.lower() in input_data.lower():
                return False

        return True

    @staticmethod

    def sanitize_input(input_data: str) -> str:
        """
        Санитизирует входные данные

        Args:
            input_data: Входные данные для санитизации

        Returns:
            Очищенные данные
        """
        # Удаляем потенциально опасные символы
        sanitized = input_data.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        sanitized = sanitized.replace('/', '&#x2F;').replace('\\', '&#x5C;')

        return sanitized

    @staticmethod

    def validate_file_type(filename: str, allowed_extensions: list) -> bool:
        """
        Проверяет тип файла

        Args:
            filename: Имя файла
            allowed_extensions: Разрешенные расширения

        Returns:
            True если тип файла допустим, иначе False
        """
        ext = Path(filename).suffix.lower()
        return ext in allowed_extensions

    @staticmethod

    def calculate_checksum(data: Union[str, bytes]) -> str:
        """
        Вычисляет контрольную сумму данных

        Args:
            data: Данные для вычисления контрольной суммы

        Returns:
            Контрольная сумма в формате hex
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(data)
        return digest.finalize().hex()

def main():
    """Главная функция для демонстрации возможностей шифрования"""
    print("=== МОДУЛЬ ШИФРОВАНИЯ ДАННЫХ ПРОЕКТА ===")

    # Создаем шифровальщик данных
    encryptor = DataEncryption()

    print("✓ Шифровальщик данных инициализирован")

    # Тестируем шифрование строки
    original_text = "Конфиденциальные данные проекта"
    encrypted_text = encryptor.encrypt_string(original_text)
    decrypted_text = encryptor.decrypt_string(encrypted_text)

    print(f"✓ Шифрование строки: '{original_text}' -> '{encrypted_text[:30]}...'")
    print(f"✓ Дешифрование строки: '{decrypted_text}'")
    print(f"✓ Совпадение: {original_text == decrypted_text}")

    # Тестируем шифрование файла
    test_file = "test_data.txt"
    encrypted_file = "test_data.txt.enc"

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Тестовые данные для шифрования")

    success_encrypt = encryptor.encrypt_file(test_file, encrypted_file)
    success_decrypt = encryptor.decrypt_file(encrypted_file, "decrypted_test.txt")

    print(f"✓ Шифрование файла: {'Успешно' if success_encrypt else 'Ошибка'}")
    print(f"✓ Дешифрование файла: {'Успешно' if success_decrypt else 'Ошибка'}")

    # Тестируем безопасное хранение данных
    secure_manager = SecureDataManager()

    test_config = {
        "api_key": "secret123",
        "database_url": "postgresql://user:pass@localhost/db",
        "encryption_enabled": True
    }

    config_saved = secure_manager.store_sensitive_config(test_config)
    retrieved_config = secure_manager.get_sensitive_config()

    print(f"✓ Сохранение конфигурации: {'Успешно' if config_saved else 'Ошибка'}")
    print(f"✓ Получение конфигурации: {'Успешно' if retrieved_config is not None else 'Ошибка'}")

    # Тестируем валидацию
    validator = SecurityValidator()
    is_safe = validator.validate_input("<script>alert('xss')</script>")
    sanitized = validator.sanitize_input("<script>alert('xss')</script>")

    print(f"✓ Валидация XSS: {'Безопасно' if not is_safe else 'Небезопасно'}")
    print(f"✓ Санитизация: '{sanitized}'")

    # Вычисляем контрольную сумму
    checksum = SecurityValidator.calculate_checksum("тестовые данные")
    print(f"✓ Контрольная сумма: {checksum[:16]}...")

    # Удаляем временные файлы
    for temp_file in [test_file, encrypted_file, "decrypted_test.txt"]:
        if Path(temp_file).exists():
            Path(temp_file).unlink()

    print("Модуль шифрования данных успешно протестирован")

if __name__ == "__main__":
    main()

