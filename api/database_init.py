# -*- coding: utf-8 -*-
"""
Применение миграций Alembic при запуске приложения
"""

import logging
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.script import Script
from sqlalchemy import create_engine, text
import os

logger = logging.getLogger(__name__)


def get_current_revision(db_path: str = "data/nanoprobe.db") -> str:
    """Получение текущей ревизии БД"""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current = context.get_current_revision()
        return current or "None"


def get_head_revision() -> str:
    """Получение последней ревизии"""
    alembic_cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(alembic_cfg)
    return script.get_current_head()


def run_migrations(db_path: str = "data/nanoprobe.db") -> bool:
    """
    Применение всех доступных миграций
    
    Returns:
        bool: True если миграции применены успешно
    """
    try:
        # Создаём директорию data если нет
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        
        current = get_current_revision(db_path)
        head = get_head_revision()
        
        if current == head:
            logger.info(f"Database is up to date (revision {current})")
            return True
        
        logger.info(f"Upgrading database from {current} to {head}...")
        command.upgrade(alembic_cfg, "head")
        
        logger.info(f"Database upgraded successfully to {head}")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def init_database(db_path: str = "data/nanoprobe.db") -> bool:
    """
    Инициализация БД через DatabaseManager
    
    Returns:
        bool: True если БД инициализирована успешно
    """
    try:
        # DatabaseManager автоматически создаёт схему в конструкторе
        from utils.database import DatabaseManager
        
        db = DatabaseManager(db_path)
        
        # Проверяем что таблицы созданы
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
            )
            result = cursor.fetchone()
            if result:
                logger.info(f"Database initialized at {db_path}")
                return True
        
        logger.warning(f"Database at {db_path} may not be fully initialized")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def ensure_database(db_path: str = "data/nanoprobe.db") -> bool:
    """
    Гарантия наличия БД (миграции или fallback на direct init)
    
    Returns:
        bool: True если БД готова
    """
    # Пробуем миграции
    if run_migrations(db_path):
        return True
    
    # Fallback на прямую инициализацию
    logger.warning("Migration failed, trying direct initialization...")
    return init_database(db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    db_path = "data/nanoprobe.db"
    success = ensure_database(db_path)
    
    if success:
        print(f"✅ Database ready at {db_path}")
    else:
        print(f"❌ Database initialization failed")
        exit(1)
