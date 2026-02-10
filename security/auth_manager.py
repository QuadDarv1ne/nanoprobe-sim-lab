#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления аутентификацией для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему аутентификации и авторизации для API и компонентов проекта.
"""

import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import sqlite3
from functools import wraps
from flask import request, jsonify, g


class AuthManager:
    """
    Класс управления аутентификацией
    Обеспечивает аутентификацию пользователей, генерацию токенов и проверку прав доступа к ресурсам.
    """
    
    def __init__(self, db_path: str = "auth.db", secret_key: str = None):
        """
        Инициализирует менеджер аутентификации
        
        Args:
            db_path: Путь к базе данных аутентификации
            secret_key: Секретный ключ для подписи токенов
        """
        self.db_path = db_path
        self.secret_key = secret_key or secrets.token_hex(32)
        self.init_database()
    
    def init_database(self):
        """Инициализирует базу данных аутентификации"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу пользователей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Создаем таблицу токенов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Создаем таблицу прав доступа
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                resource TEXT NOT NULL,
                action TEXT NOT NULL,
                granted BOOLEAN DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Создаем администратора по умолчанию
        self.create_default_admin()
    
    def create_default_admin(self):
        """Создает пользователя администратора по умолчанию"""
        if not self.user_exists("admin"):
            self.register_user(
                username="admin",
                email="admin@nanoprobe-sim-lab.org",
                password="SecurePass123!",
                role="admin"
            )
            print("Создан пользователь администратора по умолчанию")
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """
        Хэширует пароль с солью
        
        Args:
            password: Пароль для хэширования
            salt: Соль (если None, генерируется новая)
            
        Returns:
            Кортеж (хэш_пароля, соль)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Комбинируем пароль и соль
        password_salt = password + salt
        # Хэшируем с использованием SHA-256
        password_hash = hashlib.sha256(password_salt.encode()).hexdigest()
        # Дополнительно используем bcrypt для усиления безопасности
        bcrypt_hash = bcrypt.hashpw(password_hash.encode(), bcrypt.gensalt())
        
        return bcrypt_hash.decode(), salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        Проверяет пароль
        
        Args:
            password: Введенный пароль
            stored_hash: Сохраненный хэш
            salt: Соль
            
        Returns:
            True если пароль верен, иначе False
        """
        password_salt = password + salt
        password_hash = hashlib.sha256(password_salt.encode()).hexdigest()
        
        return bcrypt.checkpw(password_hash.encode(), stored_hash.encode())
    
    def register_user(self, username: str, email: str, password: str, role: str = "user") -> bool:
        """
        Регистрирует нового пользователя
        
        Args:
            username: Имя пользователя
            email: Email пользователя
            password: Пароль пользователя
            role: Роль пользователя
            
        Returns:
            True если регистрация успешна, иначе False
        """
        try:
            password_hash, salt = self.hash_password(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, role))
            
            conn.commit()
            conn.close()
            
            return True
        except sqlite3.IntegrityError:
            # Пользователь с таким именем или email уже существует
            return False
        except Exception as e:
            print(f"Ошибка регистрации пользователя: {e}")
            return False
    
    def user_exists(self, username: str) -> bool:
        """
        Проверяет существование пользователя
        
        Args:
            username: Имя пользователя
            
        Returns:
            True если пользователь существует, иначе False
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Аутентифицирует пользователя
        
        Args:
            username: Имя пользователя
            password: Пароль
            
        Returns:
            Словарь с информацией о пользователе или None если аутентификация не удалась
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, salt, role, is_active
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user and user[6]:  # Проверяем is_active
            user_id, username, email, password_hash, salt, role, is_active = user
            if self.verify_password(password, password_hash, salt):
                # Обновляем время последнего входа
                self.update_last_login(user_id)
                return {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'role': role
                }
        
        return None
    
    def update_last_login(self, user_id: int):
        """
        Обновляет время последнего входа
        
        Args:
            user_id: ID пользователя
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def generate_token(self, user_id: int, expires_in: int = 3600) -> str:
        """
        Генерирует токен аутентификации
        
        Args:
            user_id: ID пользователя
            expires_in: Время жизни токена в секундах
            
        Returns:
            Строка токена
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Сохраняем токен в базе данных
        self.store_token(user_id, token, expires_in)
        
        return token
    
    def store_token(self, user_id: int, token: str, expires_in: int):
        """
        Сохраняет токен в базе данных
        
        Args:
            user_id: ID пользователя
            token: Токен
            expires_in: Время жизни в секундах
        """
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO tokens (user_id, token, expires_at)
            VALUES (?, ?, ?)
        ''', (user_id, token, expires_at))
        
        conn.commit()
        conn.close()
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Проверяет токен аутентификации
        
        Args:
            token: Токен для проверки
            
        Returns:
            Словарь с информацией о пользователе или None если токен недействителен
        """
        try:
            # Проверяем токен с помощью JWT
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            user_id = payload['user_id']
            
            # Проверяем, что токен существует в базе данных
            if self.token_exists_in_db(token):
                # Получаем информацию о пользователе
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, username, email, role
                    FROM users WHERE id = ?
                ''', (user_id,))
                
                user = cursor.fetchone()
                conn.close()
                
                if user:
                    return {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'role': user[3]
                    }
        
        except jwt.ExpiredSignatureError:
            # Токен истек
            self.remove_expired_token(token)
        except jwt.InvalidTokenError:
            # Неверный токен
            pass
        
        return None
    
    def token_exists_in_db(self, token: str) -> bool:
        """
        Проверяет существование токена в базе данных
        
        Args:
            token: Токен для проверки
            
        Returns:
            True если токен существует, иначе False
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM tokens WHERE token = ?', (token,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def remove_expired_token(self, token: str):
        """
        Удаляет истекший токен из базы данных
        
        Args:
            token: Токен для удаления
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM tokens WHERE token = ?', (token,))
        
        conn.commit()
        conn.close()
    
    def get_user_permissions(self, user_id: int) -> List[Dict[str, str]]:
        """
        Получает права доступа пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список прав доступа
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Сначала получаем роль пользователя
        cursor.execute('SELECT role FROM users WHERE id = ?', (user_id,))
        role_result = cursor.fetchone()
        
        if not role_result:
            conn.close()
            return []
        
        user_role = role_result[0]
        
        # Получаем права доступа для роли
        cursor.execute('''
            SELECT resource, action
            FROM permissions
            WHERE role = ? AND granted = 1
        ''', (user_role,))
        
        permissions = [{'resource': row[0], 'action': row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return permissions
    
    def has_permission(self, user_id: int, resource: str, action: str) -> bool:
        """
        Проверяет права доступа пользователя к ресурсу
        
        Args:
            user_id: ID пользователя
            resource: Ресурс
            action: Действие
            
        Returns:
            True если доступ разрешен, иначе False
        """
        permissions = self.get_user_permissions(user_id)
        for perm in permissions:
            if perm['resource'] == resource and perm['action'] == action:
                return True
        return False
    
    def require_auth(self, resource: str = None, action: str = None):
        """
        Декоратор для защиты маршрутов аутентификацией
        
        Args:
            resource: Ресурс для проверки прав
            action: Действие для проверки прав
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                token = None
                
                # Проверяем токен в заголовке
                if 'Authorization' in request.headers:
                    auth_header = request.headers['Authorization']
                    if auth_header.startswith('Bearer '):
                        token = auth_header.split(' ')[1]
                
                # Проверяем токен в параметрах запроса
                if not token and 'token' in request.args:
                    token = request.args.get('token')
                
                if not token:
                    return jsonify({'status': 'error', 'message': 'Токен аутентификации отсутствует'}), 401
                
                user_info = self.verify_token(token)
                if not user_info:
                    return jsonify({'status': 'error', 'message': 'Неверный или истекший токен'}), 401
                
                # Проверяем права доступа если указаны
                if resource and action:
                    if not self.has_permission(user_info['id'], resource, action):
                        return jsonify({'status': 'error', 'message': 'Недостаточно прав доступа'}), 403
                
                # Сохраняем информацию о пользователе в глобальной переменной
                g.current_user = user_info
                g.auth_token = token
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator


def main():
    """Главная функция для демонстрации возможностей аутентификации"""
    print("=== МЕНЕДЖЕР АУТЕНТИФИКАЦИИ ПРОЕКТА ===")
    
    # Создаем менеджер аутентификации
    auth_manager = AuthManager()
    
    print("✓ Менеджер аутентификации инициализирован")
    print("✓ База данных аутентификации создана")
    print("✓ Пользователь администратора по умолчанию создан")
    
    # Тестируем регистрацию пользователя
    success = auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="TestPassword123!"
    )
    print(f"✓ Регистрация тестового пользователя: {'Успешна' if success else 'Ошибка'}")
    
    # Тестируем аутентификацию
    user_info = auth_manager.authenticate_user("testuser", "TestPassword123!")
    if user_info:
        print(f"✓ Аутентификация успешна для пользователя: {user_info['username']}")
        
        # Генерируем токен
        token = auth_manager.generate_token(user_info['id'])
        print(f"✓ Токен сгенерирован: {token[:20]}...")
        
        # Проверяем токен
        verified_user = auth_manager.verify_token(token)
        print(f"✓ Проверка токена: {'Успешна' if verified_user else 'Ошибка'}")
    
    # Проверяем права доступа
    permissions = auth_manager.get_user_permissions(user_info['id']) if user_info else []
    print(f"✓ Права доступа пользователя: {permissions}")
    
    print("Менеджер аутентификации успешно протестирован")


if __name__ == "__main__":
    main()