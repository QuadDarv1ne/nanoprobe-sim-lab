"""
Gunicorn конфигурация для production запуска FastAPI приложения
Использование: gunicorn -c gunicorn_conf.py api.main:app
"""

import multiprocessing
import os
from datetime import datetime, timezone

# ==================== Server Binding ====================

# Хост и порт для binding
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# ==================== Workers ====================

# Количество workers
# Формула: (2 x CPU cores) + 1 для CPU-bound задач
# Для I/O-bound задач можно больше: 4-16 workers
workers = os.getenv("GUNICORN_WORKERS", str(multiprocessing.cpu_count() * 2 + 1))
workers = int(workers)

# Класс workers для asyncio приложений
worker_class = "uvicorn.workers.UvicornWorker"

# Таймаут worker'а (секунды)
# Если worker не отвечает дольше, он убивается
timeout = os.getenv("GUNICORN_TIMEOUT", "120")
timeout = int(timeout)

# Graceful timeout - время на завершение запросов при перезагрузке
graceful_timeout = os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30")
graceful_timeout = int(graceful_timeout)

# Keepalive timeout для HTTP соединений
keepalive = os.getenv("GUNICORN_KEEPALIVE", "5")
keepalive = int(keepalive)

# ==================== Logging ====================

# Уровни логирования
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Access log
accesslog = os.getenv("GUNICORN_ACCESSLOG", "logs/gunicorn_access.log")

# Error log
errorlog = os.getenv("GUNICORN_ERRORLOG", "logs/gunicorn_error.log")

# Формат логирования
access_log_format = os.getenv(
    "GUNICORN_ACCESS_LOG_FORMAT",
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs',
)

# ==================== Process Naming ====================

# Имя процесса (для мониторинга)
proc_name = os.getenv("GUNICORN_PROC_NAME", "nanoprobe-fastapi")

# ==================== Security ====================

# Отключение передачи заголовков от клиента
limit_request_line = os.getenv("GUNICORN_LIMIT_REQUEST_LINE", "4094")
limit_request_line = int(limit_request_line)

limit_request_fields = os.getenv("GUNICORN_LIMIT_REQUEST_FIELDS", "100")
limit_request_fields = int(limit_request_fields)

limit_request_field_size = os.getenv("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190")
limit_request_field_size = int(limit_request_field_size)

# ==================== Performance ====================

# Максимальное количество соединений на worker
worker_connections = os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000")
worker_connections = int(worker_connections)

# Максимальное количество pending connections
backlog = os.getenv("GUNICORN_BACKLOG", "2048")
backlog = int(backlog)

# ==================== SSL/TLS (опционально) ====================

# Если нужно SSL termination на уровне Gunicorn
# keyfile = os.getenv('GUNICORN_KEYFILE', '/etc/ssl/private/server.key')
# certfile = os.getenv('GUNICORN_CERTFILE', '/etc/ssl/certs/server.crt')

# ==================== Hooks ====================


def on_starting(server):
    """Вызывается перед запуском master процесса"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] 🚀 Nanoprobe FastAPI starting...")
    print(f"  Workers: {workers}, Bind: {bind}, Log Level: {loglevel}")


def on_reload(server):
    """Вызывается при перезагрузке workers"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] 🔄 Workers reloading...")


def when_ready(server):
    """Вызывается когда server готов принимать соединения"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] ✅ Server ready on {bind}")

    # Создание директории для логов если не существует
    import os
    from pathlib import Path

    log_dirs = ["logs", "logs/gunicorn"]
    for log_dir in log_dirs:
        Path(log_dir).mkdir(parents=True, exist_ok=True)


def pre_fork(server, worker):
    """Вызывается перед fork'ом worker'а"""
    pass


def post_fork(server, worker):
    """Вызывается после fork'а worker'а"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] 👷 Worker spawned: {worker.pid}")


def worker_init(worker):
    """Инициализация worker'а"""
    pass


def worker_abort(worker):
    """Вызывается при abort worker'а"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ Worker {worker.pid} aborted")


def worker_exit(server, worker):
    """Вызывается при выходе worker'а"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] 🛑 Worker {worker.pid} exited")


def pre_exec(server):
    """Вызывается перед exec нового master процесса"""
    pass


def child_exit(server, worker):
    """Вызывается при выходе child процесса"""
    pass
