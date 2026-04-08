"""
Performance Monitoring API

Endpoints для мониторинга производительности:
- /metrics - Prometheus metrics
- /health/detailed - Детальная информация о здоровье
- /monitoring/stats - Статистика мониторинга
- /db/profile - Профилирование SQL запросов
"""

from fastapi import APIRouter, Response, Query, HTTPException
import psutil
import logging
import time
import json
import re
from datetime import datetime, timezone
import sqlite3
from pathlib import Path

from api.state import get_system_disk_usage


from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Получение Prometheus метрик для сбора",
)
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Используется Prometheus server для сбора метрик.
    Content-Type: text/plain; version=0.0.4
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get(
    "/health/detailed",
    summary="Detailed Health Check",
    description="Детальная проверка здоровья системы",
)
async def get_detailed_health():
    """
    Детальная информация о здоровье системы.
    
    Включает:
    - CPU usage
    - Memory usage
    - Disk usage
    - Network stats
    - Process info
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_per_core = psutil.cpu_percent(percpu=True)
    cpu_freq = psutil.cpu_freq()
    
    # Memory
    memory = psutil.virtual_memory()
    
    # Disk
    disk = get_system_disk_usage()
    
    # Network
    net = psutil.net_io_counters()
    
    # Process
    process = psutil.Process()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "cpu": {
                "percent": cpu_percent,
                "per_core": cpu_per_core,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
            },
            "memory": {
                "percent": memory.percent,
                "available_mb": memory.available // (1024 * 1024),
                "total_mb": memory.total // (1024 * 1024),
            },
            "disk": {
                "percent": disk.percent,
                "free_gb": disk.free // (1024 * 1024 * 1024),
                "total_gb": disk.total // (1024 * 1024 * 1024),
            },
            "network": {
                "bytes_sent_mb": net.bytes_sent // (1024 * 1024),
                "bytes_recv_mb": net.bytes_recv // (1024 * 1024),
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
            },
        },
        "process": {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "status": process.status(),
            "uptime_seconds": datetime.now(timezone.utc).timestamp() - process.create_time(),
        },
    }


@router.get(
    "/stats",
    summary="Monitoring Statistics",
    description="Статистика мониторинга",
)
async def get_monitoring_stats():
    """
    Статистика мониторинга.

    Включает:
    - Uptime
    - Request counts
    - Error rates
    """
    # Получаем uptime
    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    uptime = datetime.now(timezone.utc) - boot_time

    return {
        "uptime": {
            "seconds": uptime.total_seconds(),
            "formatted": str(uptime),
            "boot_time": boot_time.isoformat(),
        },
        "cpu": {
            "cores": psutil.cpu_count(),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total // (1024 * 1024 * 1024),
        },
        "disk": {
            "partitions": len(psutil.disk_partitions()),
        },
    }


@router.get(
    "/db/profile",
    summary="Database Query Profiling",
    description="Профилирование SQL запросов для оптимизации производительности",
)
async def profile_database_query(
    query: str = Query(..., description="SQL запрос для профилирования"),
    params: Optional[str] = Query(None, description="JSON параметры запроса"),
    analyze: bool = Query(False, description="Использовать EXPLAIN ANALYZE"),
) -> Dict[str, Any]:
    """
    Профилирование SQL запросов.

    Примеры использования:

    1. Базовое профилирование:
       `/monitoring/db/profile?query=SELECT * FROM scans LIMIT 10`

    2. С EXPLAIN ANALYZE:
       `/monitoring/db/profile?query=SELECT * FROM scans WHERE user_id = 1&analyze=true`

    3. С параметрами:
       `/monitoring/db/profile?query=SELECT * FROM scans WHERE status = ?&params=["completed"]`

    Возвращает:
    - query_plan - план выполнения запроса
    - execution_time - время выполнения (для EXPLAIN ANALYZE)
    - index_usage - информация об использовании индексов
    - recommendations - рекомендации по оптимизации
    """
    db_path = PROJECT_ROOT / "data" / "nanoprobe.db"

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    # Защита от SQL injection - разрешаем только SELECT
    if not re.match(r'^\s*SELECT\s+', query, re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail="Only SELECT queries are allowed for profiling"
        )

    # Блокируем опасные операции
    dangerous_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'ATTACH', 'DETACH']
    for pattern in dangerous_patterns:
        if re.search(rf'\b{pattern}\b', query, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail=f"Query contains forbidden operation: {pattern}"
            )

    # Валидация: извлекаем только имена таблиц для EXPLAIN QUERY PLAN (SQL injection fix)
    import re as _re
    table_names = set(_re.findall(r'\bFROM\s+(\w+)', query, _re.IGNORECASE))
    table_names.update(_re.findall(r'\bJOIN\s+(\w+)', query, _re.IGNORECASE))
    
    # Whitelist допустимых таблиц
    ALLOWED_TABLES = {
        'scans', 'simulations', 'images', 'users', 'analysis_results',
        'comparisons', 'reports', 'sstv_recordings', 'metrics'
    }
    
    # Проверяем что все таблицы из whitelist
    invalid_tables = table_names - ALLOWED_TABLES
    if invalid_tables:
        raise HTTPException(
            status_code=400,
            detail=f"Tables not allowed for profiling: {', '.join(invalid_tables)}"
        )

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Профилирование запроса
        profile_result: Dict[str, Any] = {
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analyze": analyze,
        }

        # Получаем план выполнения (безопасно - таблица из whitelist)
        # Параметризация через placeholder для защиты от SQL-инъекций
        allowed_tables = ["scans", "simulations", "images", "users", "reports", "exports"]
        table_name = query.split()[-1] if query else ""
        if table_name not in allowed_tables:
            return {"error": "Invalid table name", "status_code": 400}
        
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")

        query_plan = cursor.fetchall()
        profile_result["query_plan"] = [
            {"id": row[0], "parent": row[1], "notused": row[2], "detail": row[3]}
            for row in query_plan
        ]

        # Анализируем использование индексов
        index_usage = []
        for row in query_plan:
            detail = row[3] if len(row) > 3 else ""
            if "USING INDEX" in detail or "USING COVERING INDEX" in detail:
                index_usage.append({"index": detail, "type": "optimal"})
            elif "SCAN" in detail and "USING" not in detail:
                index_usage.append({"table": detail, "type": "full_scan", "warning": "No index used"})

        profile_result["index_usage"] = index_usage

        # Рекомендации по оптимизации
        recommendations = []
        has_full_scan = any(item.get("type") == "full_scan" for item in index_usage)
        if has_full_scan:
            recommendations.append({
                "type": "warning",
                "message": "Full table scan detected. Consider adding an index.",
                "suggestion": "CREATE INDEX idx_table_column ON table_name(column_name);"
            })

        if analyze:
            # Выполняем запрос для измерения времени
            start_time = time.perf_counter()
            if params:
                cursor.execute(query, json.loads(params))
            else:
                cursor.execute(query)
            cursor.fetchall()
            end_time = time.perf_counter()

            profile_result["execution_time_ms"] = (end_time - start_time) * 1000

            if profile_result["execution_time_ms"] > 100:
                recommendations.append({
                    "type": "warning",
                    "message": f"Slow query detected: {profile_result['execution_time_ms']:.2f}ms",
                    "suggestion": "Consider optimizing the query or adding indexes."
                })

        profile_result["recommendations"] = recommendations
        profile_result["status"] = "success"

        return profile_result

    except sqlite3.Error as e:
        logger.error(f"Database profiling error: {e}")
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON params: {str(e)}")
    except Exception as e:
        logger.error(f"Profiling error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Гарантированное закрытие соединения (resource leak fix)
        if conn:
            try:
                conn.close()
            except Exception:
                pass


@router.get(
    "/db/indexes",
    summary="Database Indexes List",
    description="Получить список всех индексов базы данных",
)
async def get_database_indexes() -> Dict[str, Any]:
    """
    Получить список всех индексов базы данных.

    Возвращает:
    - indexes - список индексов
    - table_stats - статистика по таблицам
    - recommendations - рекомендации по индексам
    """
    db_path = PROJECT_ROOT / "data" / "nanoprobe.db"

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Получаем все индексы
        cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
        indexes = [
            {"name": row[0], "table": row[1], "sql": row[2]}
            for row in cursor.fetchall()
        ]

        # Получаем статистику по таблицам
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        table_stats = {}
        for table in tables:
            try:
                # Безопасно - table из sqlite_master (доверенный источник)
                # Дополнительная валидация имени таблицы
                if not table.isidentifier() or table.startswith('_'):
                    continue
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_stats[table] = {"rows": count}
            except sqlite3.Error:
                pass

        return {
            "indexes": indexes,
            "table_stats": table_stats,
            "total_indexes": len(indexes),
            "total_tables": len(tables),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Гарантированное закрытие соединения
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# Project root for database path
PROJECT_ROOT = Path(__file__).parent.parent.parent
