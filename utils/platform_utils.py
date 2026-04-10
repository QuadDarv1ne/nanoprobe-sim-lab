"""
Кроссплатформенные утилиты для определения путей
"""

import os
import platform


def get_system_drive() -> str:
    """
    Получение пути системного диска для текущей ОС

    Returns:
        Путь системного диска: "C:\" для Windows, "/" для Linux/Mac

    Examples:
        >>> get_system_drive()
        'C:\\'  # на Windows
        '/'     # на Linux/Mac
    """
    if platform.system() == "Windows":
        return os.environ.get("SYSTEMDRIVE", "C:\\")
    else:
        return "/"


def get_system_disk_usage():
    """
    Получение информации об использовании системного диска

    Returns:
        Именованный кортеж с полями: total, used, free, percent
        Или None в случае ошибки

    Examples:
        >>> usage = get_system_disk_usage()
        >>> if usage:
        ...     print(f"Total: {usage.total}, Used: {usage.percent}%")
    """
    try:
        import psutil

        return psutil.disk_usage(get_system_drive())
    except (OSError, ImportError) as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to get disk usage: {e}")
        return None
