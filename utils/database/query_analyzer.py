"""
Database Query Analyzer

Утилиты для анализа SQL запросов с использованием EXPLAIN ANALYZE.
Поддержка SQLite и PostgreSQL.

Features:
- EXPLAIN ANALYZE для запросов
- Выявление медленных запросов
- Рекомендации по оптимизации
- Статистика выполнения
"""

import sqlite3
import asyncio
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import timedelta
import time

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """План выполнения запроса"""
    query: str
    plan: List[Dict[str, Any]]
    execution_time_ms: float
    rows_affected: int
    uses_index: bool
    full_table_scan: bool
    recommendations: List[str]


@dataclass
class QueryStats:
    """Статистика запроса"""
    query: str
    total_time_ms: float
    rows_returned: int
    rows_scanned: int
    index_used: Optional[str]
    temp_tables: int
    sorts: int


class QueryAnalyzer:
    """
    Анализатор SQL запросов
    
    Поддерживает:
    - SQLite
    - PostgreSQL (через asyncpg)
    """

    def __init__(self, db_path: Optional[str] = None, dsn: Optional[str] = None):
        """
        Инициализация анализатора.
        
        Args:
            db_path: Путь к SQLite базе данных
            dsn: DSN для PostgreSQL подключения
        """
        self.db_path = db_path
        self.dsn = dsn
        self.db_type = "postgresql" if dsn else "sqlite"
        
        if db_path and not dsn:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
        else:
            self.conn = None

    def analyze_query(self, query: str, params: Optional[Tuple] = None) -> QueryPlan:
        """
        Анализ запроса с помощью EXPLAIN QUERY PLAN.
        
        Args:
            query: SQL запрос для анализа
            params: Параметры запроса
            
        Returns:
            QueryPlan с планом выполнения и рекомендациями
        """
        if self.db_type == "sqlite":
            return self._analyze_sqlite(query, params)
        else:
            raise NotImplementedError("PostgreSQL analysis requires asyncpg")

    def _analyze_sqlite(self, query: str, params: Optional[Tuple] = None) -> QueryPlan:
        """Анализ SQLite запроса"""
        plan = []
        recommendations = []
        uses_index = False
        full_table_scan = False
        
        # Получаем план выполнения
        try:
            cursor = self.conn.execute(
                f"EXPLAIN QUERY PLAN {query}",
                params or ()
            )
            
            for row in cursor.fetchall():
                plan_entry = {
                    "id": row[0],
                    "parent": row[1],
                    "notused": row[2],
                    "detail": row[3]
                }
                plan.append(plan_entry)
                
                detail = row[3].upper()
                
                # Анализ плана
                if "USING INDEX" in detail or "USING COVERING INDEX" in detail:
                    uses_index = True
                
                if "SCAN TABLE" in detail or "SCAN LIST":
                    full_table_scan = True
                    table_name = self._extract_table_name(detail)
                    recommendations.append(
                        f"Full table scan on '{table_name}'. Consider adding an index."
                    )
                
                if "TEMP B-TREE" in detail:
                    recommendations.append(
                        "Temporary B-tree used for sorting. Consider indexing ORDER BY columns."
                    )
            
            # Замер времени выполнения
            start = time.perf_counter()
            cursor = self.conn.execute(query, params or ())
            _ = cursor.fetchall()
            execution_time = (time.perf_counter() - start) * 1000
            
            rows_affected = cursor.rowcount
            
        except sqlite3.Error as e:
            logger.error(f"Query analysis error: {e}")
            raise
        
        # Генерация рекомендаций
        if not uses_index and full_table_scan:
            recommendations.append(
                "Query performs full table scan. Add appropriate indexes."
            )
        
        if execution_time > 100:
            recommendations.append(
                f"Slow query ({execution_time:.1f}ms). Consider optimization."
            )
        
        return QueryPlan(
            query=query,
            plan=plan,
            execution_time_ms=execution_time,
            rows_affected=rows_affected,
            uses_index=uses_index,
            full_table_scan=full_table_scan,
            recommendations=recommendations
        )

    def _extract_table_name(self, detail: str) -> str:
        """Извлечение имени таблицы из детали плана"""
        import re
        match = re.search(r'SCAN TABLE (\w+)', detail)
        if match:
            return match.group(1)
        return "unknown"

    def get_slow_queries(
        self,
        threshold_ms: float = 100,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Получение медленных запросов из логов.
        
        Args:
            threshold_ms: Порог медленного запроса (мс)
            limit: Максимальное количество запросов
            
        Returns:
            Список медленных запросов
        """
        # Для SQLite - анализ системных таблиц
        if self.db_type == "sqlite":
            try:
                cursor = self.conn.execute("""
                    SELECT query, avg_duration, executions
                    FROM query_stats
                    WHERE avg_duration > ?
                    ORDER BY avg_duration DESC
                    LIMIT ?
                """, (threshold_ms, limit))
                
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                # Таблица query_stats может не существовать
                return []
        
        return []

    def suggest_indexes(self, query: str) -> List[str]:
        """
        Предложения по индексам для запроса.
        
        Args:
            query: SQL запрос
            
        Returns:
            Список CREATE INDEX предложений
        """
        import re
        
        suggestions = []
        
        # Анализ WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', 
                               query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            
            # Поиск условий равенства
            eq_conditions = re.findall(r'(\w+)\s*=\s*[?\d\']', where_clause)
            for column in eq_conditions:
                suggestions.append(
                    f"CREATE INDEX idx_{column} ON table_name ({column});"
                )
            
            # Поиск условий диапазона
            range_conditions = re.findall(
                r'(\w+)\s*(?:>|<|>=|<=|BETWEEN)',
                where_clause
            )
            for column in range_conditions:
                suggestions.append(
                    f"CREATE INDEX idx_{column}_range ON table_name ({column});"
                )
        
        # Анализ ORDER BY
        order_match = re.search(r'ORDER BY\s+(.+?)(?:LIMIT|$)', query, re.IGNORECASE)
        if order_match:
            order_columns = order_match.group(1).split(',')
            for col in order_columns:
                col = col.strip().split()[0]
                suggestions.append(
                    f"CREATE INDEX idx_order_{col} ON table_name ({col});"
                )
        
        return suggestions

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Получение статистики таблицы.
        
        Args:
            table_name: Имя таблицы
            
        Returns:
            Статистика таблицы
        """
        if self.db_type == "sqlite":
            stats = {}
            
            # Количество строк
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats["row_count"] = cursor.fetchone()[0]
            
            # Размер таблицы
            cursor = self.conn.execute(
                "SELECT page_count * page_size as size "
                "FROM pragma_page_count(), pragma_page_size()"
            )
            stats["size_bytes"] = cursor.fetchone()[0]
            
            # Индексы
            cursor = self.conn.execute(
                f"PRAGMA index_list({table_name})"
            )
            indexes = cursor.fetchall()
            stats["index_count"] = len(indexes)
            stats["indexes"] = [idx[1] for idx in indexes]
            
            # Колонки
            cursor = self.conn.execute(
                f"PRAGMA table_info({table_name})"
            )
            stats["columns"] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "pk": bool(col[5])
                }
                for col in cursor.fetchall()
            ]
            
            return stats
        
        return {}

    def close(self):
        """Закрытие соединения"""
        if self.conn:
            self.conn.close()


# Async версия для PostgreSQL
class AsyncQueryAnalyzer:
    """Асинхронный анализатор запросов для PostgreSQL"""

    def __init__(self, dsn: str):
        """
        Инициализация с PostgreSQL DSN.
        
        Args:
            dsn: PostgreSQL DSN строка подключения
        """
        self.dsn = dsn
        self._pool = None

    async def connect(self):
        """Подключение к базе данных"""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.dsn)
        except ImportError:
            raise ImportError("asyncpg required: pip install asyncpg")

    async def close(self):
        """Закрытие подключения"""
        if self._pool:
            await self._pool.close()

    async def analyze_query(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> QueryPlan:
        """Анализ запроса с EXPLAIN ANALYZE"""
        if not self._pool:
            await self.connect()
        
        plan = []
        recommendations = []
        uses_index = False
        full_table_scan = False
        
        async with self._pool.acquire() as conn:
            # EXPLAIN ANALYZE
            explain_query = f"EXPLAIN ANALYZE {query}"
            
            try:
                rows = await conn.fetch(explain_query, *(params or ()))
                
                for row in rows:
                    plan_entry = {"Query Plan": row[0]}
                    plan.append(plan_entry)
                    
                    plan_text = row[0].upper()
                    
                    if "INDEX SCAN" in plan_text or "INDEX ONLY SCAN" in plan_text:
                        uses_index = True
                    
                    if "SEQ SCAN" in plan_text:
                        full_table_scan = True
                        recommendations.append(
                            "Sequential scan detected. Consider adding an index."
                        )
                    
                    if "SORT METHOD" in plan_text:
                        recommendations.append(
                            "External sort used. Consider indexing ORDER BY columns."
                        )
                
                # Замер времени выполнения уже включён в EXPLAIN ANALYZE
                execution_time = self._parse_execution_time(plan)
                
            except Exception as e:
                logger.error(f"Query analysis error: {e}")
                raise
        
        if not uses_index and full_table_scan:
            recommendations.append(
                "Query performs sequential scan. Add appropriate indexes."
            )
        
        return QueryPlan(
            query=query,
            plan=plan,
            execution_time_ms=execution_time,
            rows_affected=-1,  # Не доступно для PostgreSQL
            uses_index=uses_index,
            full_table_scan=full_table_scan,
            recommendations=recommendations
        )

    def _parse_execution_time(self, plan: List[Dict]) -> float:
        """Парсинг времени выполнения из плана"""
        import re
        
        for entry in plan:
            plan_text = entry.get("Query Plan", "")
            match = re.search(r'Execution Time: ([\d.]+) ms', plan_text)
            if match:
                return float(match.group(1))
        
        return 0.0


# Utility функции
def analyze_query(db_path: str, query: str, params: Optional[Tuple] = None) -> QueryPlan:
    """
    Быстрый анализ запроса.
    
    Args:
        db_path: Путь к SQLite базе
        query: SQL запрос
        params: Параметры запроса
        
    Returns:
        QueryPlan с результатами анализа
    """
    analyzer = QueryAnalyzer(db_path)
    try:
        return analyzer.analyze_query(query, params)
    finally:
        analyzer.close()


def print_query_plan(plan: QueryPlan):
    """Вывод плана запроса в консоль"""
    print("\n" + "=" * 60)
    print(f"Query: {plan.query}")
    print("=" * 60)
    
    print(f"\nExecution Time: {plan.execution_time_ms:.2f} ms")
    print(f"Rows Affected: {plan.rows_affected}")
    print(f"Uses Index: {'Yes' if plan.uses_index else 'No'}")
    print(f"Full Table Scan: {'Yes' if plan.full_table_scan else 'No'}")
    
    print("\nPlan:")
    for entry in plan.plan:
        print(f"  {entry}")
    
    if plan.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(plan.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("=" * 60 + "\n")
