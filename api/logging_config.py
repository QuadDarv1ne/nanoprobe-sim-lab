"""
Логирование для FastAPI приложения
Интеграция production_logger с uvicorn/gunicorn
"""

import logging
from pathlib import Path
from datetime import datetime

# Создание директорий для логов
LOG_DIR = Path('logs/api')
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Конфигурация логирования
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default': {
            '()': 'uvicorn.logging.DefaultFormatter',
            'fmt': '%(levelprefix)s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'access': {
            '()': 'uvicorn.logging.AccessFormatter',
            'fmt': '%(asctime)s | %(levelprefix)s %(client_addr)s | "%(request_line)s" %(status_code)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'json': {
            '()': 'utils.production_logger.JSONFormatter',
        },
        'detailed': {
            'format': '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },

    'handlers': {
        # Console handlers
        'console': {
            'formatter': 'default',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'level': 'INFO',
        },
        'console_debug': {
            'formatter': 'default',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
            'level': 'DEBUG',
        },

        # File handlers
        'file_info': {
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_DIR / 'nanoprobe_info.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 10,
            'encoding': 'utf-8',
            'level': 'INFO',
        },
        'file_debug': {
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_DIR / 'nanoprobe_debug.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50 MB
            'backupCount': 5,
            'encoding': 'utf-8',
            'level': 'DEBUG',
        },
        'file_errors': {
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_DIR / 'nanoprobe_errors.log'),
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 10,
            'encoding': 'utf-8',
            'level': 'ERROR',
        },
        'file_json': {
            'formatter': 'json',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_DIR / 'nanoprobe_structured.log'),
            'maxBytes': 20 * 1024 * 1024,
            'backupCount': 7,
            'encoding': 'utf-8',
            'level': 'INFO',
        },
        # Audit log handler - отдельный файл для security событий
        'file_audit': {
            'formatter': 'json',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_DIR / 'audit_security.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50 MB
            'backupCount': 30,  # Храним 30 файлов для аудита
            'encoding': 'utf-8',
            'level': 'INFO',
        },
    },

    'loggers': {
        # Uvicorn loggers
        'uvicorn': {
            'handlers': ['console', 'file_info', 'file_errors'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn.error': {
            'handlers': ['console', 'file_info', 'file_errors', 'file_debug'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn.access': {
            'handlers': ['console', 'file_info', 'file_json'],
            'level': 'INFO',
            'propagate': False,
        },

        # Application logger
        'nanoprobe': {
            'handlers': ['console', 'file_info', 'file_debug', 'file_errors', 'file_json'],
            'level': 'DEBUG',
            'propagate': False,
        },

        # Audit logger - security события
        'audit.security': {
            'handlers': ['file_audit'],
            'level': 'INFO',
            'propagate': False,
        },

        # SQLAlchemy logger
        'sqlalchemy': {
            'handlers': ['file_info', 'file_errors'],
            'level': 'WARNING',
            'propagate': False,
        },
    },

    'root': {
        'handlers': ['console', 'file_info', 'file_errors'],
        'level': 'INFO',
    },
}


def get_logger(name: str = 'nanoprobe') -> logging.Logger:
    """
    Получение настроенного логгера

    Args:
        name: Имя логгера

    Returns:
        Настроенный логгер
    """
    # Применяем конфигурацию при первом вызове
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(name)


def setup_logging():
    """
    Настройка логирования
    Вызывать при старте приложения
    """
    logging.config.dictConfig(LOGGING_CONFIG)

    logger = logging.getLogger('nanoprobe')
    logger.info(f"Логирование инициализировано: {datetime.now().isoformat()}")

    return logger


# Для использования с Gunicorn
def gunicorn_logger_config():
    """
    Конфигурация логирования для Gunicorn
    Использование: gunicorn --logger-class api.logging_config:gunicorn_logger_config
    """
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger('nanoprobe')
