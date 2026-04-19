"""
Панель администратора для Nanoprobe Sim Lab
Управление пользователями, системные настройки, мониторинг
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import APIRouter, Depends

from api.dependencies import get_current_user, get_redis_cache, require_admin
from api.error_handlers import AuthorizationError, NotFoundError, ValidationError, get_error_metrics
from api.state import get_db_manager, get_system_disk_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Администрирование"])
