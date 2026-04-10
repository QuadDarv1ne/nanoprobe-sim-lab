"""
Структурированное логирование для Nanoprobe Sim Lab

Поддерживает:
- JSON формат для продакшена (Grafana/Loki integration)
- Human-readable формат для разработки
- Автоматическое определение окружения
- Корреляция логов через request_id
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    JSON форматтер для продакшена.

    Совместим с Grafana/Loki стеком.
    """

    def __init__(self, service_name: str = "nanoprobe-sim-lab"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись в JSON"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Добавляем exception если есть
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Добавляем extra поля
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "scan_id"):
            log_data["scan_id"] = record.scan_id

        # Добавляем duration если есть
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        return json.dumps(log_data, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable форматтер для разработки.

    Цветной вывод с эмодзи для удобного чтения.
    """

    # Цвета для уровней
    LEVEL_COLORS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись в человекочитаемый вид"""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        emoji = self.LEVEL_COLORS.get(record.levelname, "📝")
        message = record.getMessage()

        # Формат: TIMESTAMP EMOJI LEVEL [LOGGER] MESSAGE
        log_line = f"{timestamp} {emoji} {record.levelname:8s} [{record.name}] {message}"

        # Добавляем exception если есть
        if record.exc_info and record.exc_info[0] is not None:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


def determine_log_format() -> str:
    """
    Определяет формат логов из окружения.

    Returns:
        'json' для production, 'human' для development
    """
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env in ("production", "prod", "staging"):
        return "json"

    return "human"


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    service_name: str = "nanoprobe-sim-lab",
) -> logging.Logger:
    """
    Настройка структурированного логирования.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Формат логов (json, human). Автоопределение если None.
        log_file: Путь к файлу логов (опционально)
        enable_console: Выводить ли в консоль
        service_name: Имя сервиса

    Returns:
        Настроенный root logger
    """
    # Определяем формат
    if log_format is None:
        log_format = determine_log_format()

    # Создаём formatter
    if log_format == "json":
        formatter = JSONFormatter(service_name=service_name)
    else:
        formatter = HumanReadableFormatter()

    # Настраиваем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Очищаем старые handlers
    root_logger.handlers.clear()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Подавляем шумные библиотеки
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Получение logger с корректным именем.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingMiddleware:
    """
    Middleware для логирования HTTP запросов.

    Добавляет request_id и логирует duration.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        import uuid

        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Генерируем request_id
        request_id = str(uuid.uuid4())

        # Логируем запрос
        logger = get_logger("api.request")
        path = scope["path"]
        method = scope.get("method", "UNKNOWN")

        extra = {
            "request_id": request_id,
            "path": path,
            "method": method,
        }

        logger.info(
            f"{method} {path} - started",
            extra=extra,
        )

        # Вызываем следующий middleware/handler
        await self.app(scope, receive, send)

        # Логируем ответ (duration будет добавлен другим middleware)
        logger.info(
            f"{method} {path} - completed",
            extra=extra,
        )


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **kwargs):
    """
    Логирование производительности.

    Args:
        logger: Logger instance
        operation: Название операции
        duration_ms: Длительность в миллисекундах
        **kwargs: Дополнительные метрики
    """
    extra = {
        "duration_ms": round(duration_ms, 2),
        **kwargs,
    }

    logger.info(
        f"Performance: {operation} completed in {duration_ms:.2f}ms",
        extra=extra,
    )


# Автоматическая настройка при импорте (если не настроено вручную)
if not logging.getLogger().handlers:
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
    )
