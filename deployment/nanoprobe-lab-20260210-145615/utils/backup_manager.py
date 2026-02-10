#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления резервным копированием для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для создания, 
управления и восстановления резервных копий данных проекта.
"""

import os
import shutil
import zipfile
import tarfile
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import subprocess
from cryptography.fernet import Fernet
from utils.logger import setup_project_logging
from utils.config_manager import ConfigManager


class BackupManager:
    """
    Класс управления резервным копированием
    Обеспечивает создание, хранение и восстановление 
    резервных копий проекта и его данных.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Инициализирует менеджер резервного копирования
        
        Args:
            config_manager: Менеджер конфигурации (опционально)
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger_manager = setup_project_logging(self.config_manager)
        
        # Получаем пути из конфигурации
        self.backup_dir = Path(self.config_manager.get("paths.backup_dir", "backups"))
        self.data_dir = Path(self.config_manager.get("paths.data_dir", "data"))
        self.output_dir = Path(self.config_manager.get("paths.output_dir", "output"))
        
        # Создаем директорию резервных копий
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Файл для хранения метаданных резервных копий
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Загружает метаданные резервных копий"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger_manager.log_system_event(f"Ошибка загрузки метаданных: {e}", "WARNING")
                return {}
        return {}
    
    def _save_metadata(self):
        """Сохраняет метаданные резервных копий"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка сохранения метаданных: {e}", "ERROR")
    
    def create_backup(self, backup_name: str = None, include_outputs: bool = True, 
                     compress: bool = True, encrypt: bool = False, 
                     encryption_key: bytes = None) -> Optional[str]:
        """
        Создает резервную копию проекта
        
        Args:
            backup_name: Имя резервной копии (если None, генерируется автоматически)
            include_outputs: Включать ли выходные данные
            compress: Сжимать ли резервную копию
            encrypt: Шифровать ли резервную копию
            encryption_key: Ключ шифрования (если None, генерируется новый)
            
        Returns:
            Путь к созданной резервной копии или None при ошибке
        """
        try:
            # Генерируем имя резервной копии
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"nanoprobe_backup_{timestamp}"
            
            # Создаем временный каталог для резервной копии
            temp_dir = Path(tempfile.mkdtemp(prefix=f"nanoprobe_backup_{backup_name}_"))
            
            # Список директорий для резервного копирования
            dirs_to_backup = [
                self.data_dir,
                Path("cpp-spm-hardware-sim"),
                Path("py-surface-image-analyzer"),
                Path("py-sstv-groundstation"),
                Path("utils"),
                Path("api"),
                Path("security"),
                Path("tests"),
                Path("docs")
            ]
            
            if include_outputs:
                dirs_to_backup.append(self.output_dir)
            
            # Копируем директории
            for src_dir in dirs_to_backup:
                if src_dir.exists():
                    dest_dir = temp_dir / src_dir.name
                    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
            
            # Копируем конфигурационные файлы
            config_files = ["config.json", "requirements.txt", "CMakeLists.txt", "README.md"]
            for config_file in config_files:
                src_file = Path(config_file)
                if src_file.exists():
                    shutil.copy2(src_file, temp_dir / src_file.name)
            
            # Создаем файл индекса
            index_file = temp_dir / "backup_index.json"
            backup_info = {
                "name": backup_name,
                "timestamp": datetime.now().isoformat(),
                "includes_outputs": include_outputs,
                "compressed": compress,
                "encrypted": encrypt,
                "directories_backed_up": [str(d) for d in dirs_to_backup if d.exists()],
                "files_backed_up": [str(f) for f in config_files if Path(f).exists()]
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False, default=str)
            
            # Определяем путь для сохранения
            if compress:
                if encrypt:
                    # Сначала сжимаем, потом шифруем
                    archive_path = self.backup_dir / f"{backup_name}.zip"
                    self._create_zip_archive(temp_dir, archive_path)
                    
                    # Шифруем архив
                    encrypted_path = archive_path.with_suffix('.enc.zip')
                    if self._encrypt_file(archive_path, encrypted_path, encryption_key):
                        archive_path.unlink()  # Удаляем незашифрованный архив
                        final_path = encrypted_path
                    else:
                        self.logger_manager.log_system_event("Ошибка шифрования архива", "ERROR")
                        shutil.rmtree(temp_dir)
                        return None
                else:
                    # Просто сжимаем
                    archive_path = self.backup_dir / f"{backup_name}.zip"
                    self._create_zip_archive(temp_dir, archive_path)
                    final_path = archive_path
            else:
                # Сохраняем как директорию
                backup_path = self.backup_dir / backup_name
                shutil.copytree(temp_dir, backup_path)
                final_path = backup_path
            
            # Удаляем временный каталог
            shutil.rmtree(temp_dir)
            
            # Обновляем метаданные
            self.metadata[backup_name] = {
                "path": str(final_path),
                "timestamp": backup_info["timestamp"],
                "size": self._get_file_size(final_path),
                "includes_outputs": include_outputs,
                "compressed": compress,
                "encrypted": encrypt
            }
            self._save_metadata()
            
            self.logger_manager.log_system_event(f"Создана резервная копия: {backup_name}", "INFO")
            return str(final_path)
            
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка создания резервной копии: {e}", "ERROR")
            return None
    
    def _create_zip_archive(self, source_dir: Path, archive_path: Path):
        """
        Создает ZIP архив
        
        Args:
            source_dir: Исходная директория
            archive_path: Путь к архиву
        """
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(source_dir)
                    zipf.write(file_path, arc_path)
    
    def _encrypt_file(self, input_file: Path, output_file: Path, key: bytes = None) -> bool:
        """
        Шифрует файл
        
        Args:
            input_file: Входной файл
            output_file: Выходной файл
            key: Ключ шифрования
            
        Returns:
            True если шифрование успешно, иначе False
        """
        try:
            if key is None:
                key = Fernet.generate_key()
            
            cipher = Fernet(key)
            
            with open(input_file, 'rb') as f:
                data = f.read()
            
            encrypted_data = cipher.encrypt(data)
            
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка шифрования файла: {e}", "ERROR")
            return False
    
    def restore_backup(self, backup_name: str, restore_path: str = None, 
                      decrypt_key: bytes = None) -> bool:
        """
        Восстанавливает резервную копию
        
        Args:
            backup_name: Имя резервной копии
            restore_path: Путь для восстановления (по умолчанию текущая директория)
            decrypt_key: Ключ для дешифрования (если резервная копия зашифрована)
            
        Returns:
            True если восстановление успешно, иначе False
        """
        try:
            if backup_name not in self.metadata:
                self.logger_manager.log_system_event(f"Резервная копия не найдена: {backup_name}", "ERROR")
                return False
            
            backup_info = self.metadata[backup_name]
            backup_path = Path(backup_info["path"])
            
            if not backup_path.exists():
                self.logger_manager.log_system_event(f"Файл резервной копии не найден: {backup_path}", "ERROR")
                return False
            
            # Определяем путь восстановления
            if restore_path is None:
                restore_path = Path.cwd() / f"restored_{backup_name}"
            else:
                restore_path = Path(restore_path)
            
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Проверяем, зашифрована ли резервная копия
            if backup_info.get("encrypted", False):
                if decrypt_key is None:
                    self.logger_manager.log_system_event("Требуется ключ для дешифрования", "ERROR")
                    return False
                
                # Создаем временный файл для дешифрования
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                # Дешифруем файл
                if not self._decrypt_file(backup_path, temp_path, decrypt_key):
                    self.logger_manager.log_system_event("Ошибка дешифрования", "ERROR")
                    return False
                
                # Извлекаем архив
                extracted_path = restore_path
                with zipfile.ZipFile(temp_path, 'r') as zipf:
                    zipf.extractall(extracted_path)
                
                # Удаляем временный файл
                temp_path.unlink()
            elif backup_path.suffix == '.zip':
                # Извлекаем ZIP архив
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    zipf.extractall(restore_path)
            else:
                # Копируем директорию
                shutil.copytree(backup_path, restore_path, dirs_exist_ok=True)
            
            self.logger_manager.log_system_event(f"Восстановлена резервная копия: {backup_name}", "INFO")
            return True
            
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка восстановления резервной копии: {e}", "ERROR")
            return False
    
    def _decrypt_file(self, input_file: Path, output_file: Path, key: bytes) -> bool:
        """
        Дешифрует файл
        
        Args:
            input_file: Входной файл
            output_file: Выходной файл
            key: Ключ шифрования
            
        Returns:
            True если дешифрование успешно, иначе False
        """
        try:
            cipher = Fernet(key)
            
            with open(input_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = cipher.decrypt(encrypted_data)
            
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            
            return True
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка дешифрования файла: {e}", "ERROR")
            return False
    
    def list_backups(self) -> List[Dict]:
        """
        Возвращает список резервных копий
        
        Returns:
            Список словарей с информацией о резервных копиях
        """
        backups = []
        for name, info in self.metadata.items():
            backups.append({
                "name": name,
                "timestamp": info["timestamp"],
                "size": info["size"],
                "includes_outputs": info.get("includes_outputs", False),
                "compressed": info.get("compressed", False),
                "encrypted": info.get("encrypted", False),
                "path": info["path"]
            })
        
        # Сортируем по времени создания (новые первыми)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Удаляет резервную копию
        
        Args:
            backup_name: Имя резервной копии
            
        Returns:
            True если удаление успешно, иначе False
        """
        try:
            if backup_name not in self.metadata:
                return False
            
            backup_info = self.metadata[backup_name]
            backup_path = Path(backup_info["path"])
            
            if backup_path.exists():
                if backup_path.is_file():
                    backup_path.unlink()
                else:
                    shutil.rmtree(backup_path)
            
            del self.metadata[backup_name]
            self._save_metadata()
            
            self.logger_manager.log_system_event(f"Удалена резервная копия: {backup_name}", "INFO")
            return True
            
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка удаления резервной копии: {e}", "ERROR")
            return False
    
    def verify_backup_integrity(self, backup_name: str) -> Tuple[bool, str]:
        """
        Проверяет целостность резервной копии
        
        Args:
            backup_name: Имя резервной копии
            
        Returns:
            Кортеж (успешно, сообщение)
        """
        try:
            if backup_name not in self.metadata:
                return False, f"Резервная копия не найдена: {backup_name}"
            
            backup_info = self.metadata[backup_name]
            backup_path = Path(backup_info["path"])
            
            if not backup_path.exists():
                return False, f"Файл резервной копии не найден: {backup_path}"
            
            # Проверяем размер
            actual_size = self._get_file_size(backup_path)
            expected_size = backup_info["size"]
            
            if actual_size != expected_size:
                return False, f"Несоответствие размера: ожидаемый {expected_size}, фактический {actual_size}"
            
            # Если это архив, проверяем его целостность
            if backup_path.suffix == '.zip':
                try:
                    with zipfile.ZipFile(backup_path, 'r') as zipf:
                        bad_file = zipf.testzip()
                        if bad_file:
                            return False, f"Поврежденный файл в архиве: {bad_file}"
                except zipfile.BadZipFile:
                    return False, "Файл архива поврежден"
            
            return True, "Целостность подтверждена"
            
        except Exception as e:
            return False, f"Ошибка проверки целостности: {str(e)}"
    
    def _get_file_size(self, path: Path) -> int:
        """
        Получает размер файла или директории
        
        Args:
            path: Путь к файлу или директории
            
        Returns:
            Размер в байтах
        """
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    total_size += filepath.stat().st_size
            return total_size
        else:
            return 0
    
    def cleanup_old_backups(self, keep_days: int = 30, keep_count: int = 5) -> int:
        """
        Удаляет старые резервные копии
        
        Args:
            keep_days: Хранить резервные копии не менее указанного количества дней
            keep_count: Оставлять не менее указанного количества резервных копий
            
        Returns:
            Количество удаленных резервных копий
        """
        try:
            # Получаем список резервных копий, отсортированный по дате
            backups = self.list_backups()
            
            # Удаляем лишние копии
            deleted_count = 0
            cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
            
            for i, backup in enumerate(backups):
                if i >= keep_count:  # Сохраняем не менее keep_count копий
                    backup_datetime = datetime.fromisoformat(backup["timestamp"])
                    if backup_datetime.timestamp() < cutoff_date:
                        if self.delete_backup(backup["name"]):
                            deleted_count += 1
            
            self.logger_manager.log_system_event(f"Очистка старых резервных копий: удалено {deleted_count}", "INFO")
            return deleted_count
            
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка очистки резервных копий: {e}", "ERROR")
            return 0


def main():
    """Главная функция для демонстрации возможностей менеджера резервного копирования"""
    print("=== МЕНЕДЖЕР РЕЗЕРВНОГО КОПИРОВАНИЯ ПРОЕКТА ===")
    
    # Создаем менеджер резервного копирования
    backup_manager = BackupManager()
    
    print("✓ Менеджер резервного копирования инициализирован")
    print(f"✓ Директория резервных копий: {backup_manager.backup_dir}")
    
    # Показываем существующие резервные копии
    backups = backup_manager.list_backups()
    print(f"Найдено резервных копий: {len(backups)}")
    
    if backups:
        print("Последние резервные копии:")
        for backup in backups[:3]:  # Показываем последние 3
            print(f"  - {backup['name']} ({backup['timestamp']}) - {backup['size']} байт")
    
    # Пример создания резервной копии (закомментировано для безопасности)
    # backup_path = backup_manager.create_backup(include_outputs=True, compress=True)
    # if backup_path:
    #     print(f"✓ Создана резервная копия: {backup_path}")
    # else:
    #     print("✗ Ошибка создания резервной копии")
    
    print("Менеджер резервного копирования готов к работе")


if __name__ == "__main__":
    main()