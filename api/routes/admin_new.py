"""
Панель администратора для Nanoprobe Sim Lab
Управление пользователями, системные настройки, мониторинг
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Администрирование"])
