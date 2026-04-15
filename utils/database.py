"""
Модуль базы данных для проекта Nanoprobe Simulation Lab
Хранение результатов сканирований, истории симуляций, метаданных

Refactored: ре-экспорт из utils.db для обратной совместимости.
"""

from utils.db import DatabaseManager, get_database

__all__ = ["DatabaseManager", "get_database"]
