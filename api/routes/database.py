"""
Database Query Analyzer API

Endpoints для анализа SQL запросов:
- EXPLAIN ANALYZE
- Index suggestions
- Table statistics
- Slow query log
"""

from fastapi import APIRouter, Query, Depends
from typing import Optional, List
from pydantic import BaseModel, Field
import logging

from utils.database.query_analyzer import (
    QueryAnalyzer,
    analyze_query,
    print_query_plan,
    QueryPlan,
)
from api.dependencies import get_current_user
from api.error_handlers import ValidationError, NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["Database"])


# ==================== Schemas ====================

class AnalyzeQueryRequest(BaseModel):
    query: str = Field(..., description="SQL запрос для анализа")
    params: Optional[List] = Field(None, description="Параметры запроса")


class AnalyzeQueryResponse(BaseModel):
    query: str
    execution_time_ms: float
    uses_index: bool
    full_table_scan: bool
    plan: list
    recommendations: List[str]


class IndexSuggestionResponse(BaseModel):
    query: str
    suggestions: List[str]


class TableStatsResponse(BaseModel):
    table_name: str
    row_count: int
    size_bytes: int
    index_count: int
    indexes: List[str]
    columns: List[dict]


# ==================== Endpoints ====================

@router.post(
    "/analyze",
    summary="Анализ SQL запроса",
    description="EXPLAIN ANALYZE для SQL запроса с рекомендациями по оптимизации",
    response_model=AnalyzeQueryResponse,
)
async def analyze_sql_query(
    request: AnalyzeQueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Анализ SQL запроса.
    
    - **query**: SQL запрос для анализа
    - **params**: Опциональные параметры запроса
    
    Возвращает план выполнения, время и рекомендации.
    """
    db_path = "data/nanoprobe.db"
    
    try:
        analyzer = QueryAnalyzer(db_path)
        plan = analyzer.analyze_query(request.query, tuple(request.params) if request.params else None)
        analyzer.close()
        
        return AnalyzeQueryResponse(
            query=plan.query,
            execution_time_ms=plan.execution_time_ms,
            uses_index=plan.uses_index,
            full_table_scan=plan.full_table_scan,
            plan=plan.plan,
            recommendations=plan.recommendations,
        )
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        raise ValidationError(f"Ошибка анализа запроса: {str(e)}")


@router.get(
    "/analyze",
    summary="Анализ SQL запроса (GET)",
    description="Быстрый анализ SQL запроса через GET",
)
async def analyze_sql_query_get(
    query: str = Query(..., description="SQL запрос"),
    current_user: dict = Depends(get_current_user),
):
    """Анализ SQL запроса через GET"""
    db_path = "data/nanoprobe.db"
    
    try:
        analyzer = QueryAnalyzer(db_path)
        plan = analyzer.analyze_query(query)
        analyzer.close()
        
        return {
            "query": plan.query,
            "execution_time_ms": plan.execution_time_ms,
            "uses_index": plan.uses_index,
            "full_table_scan": plan.full_table_scan,
            "recommendations": plan.recommendations,
        }
    except Exception as e:
        raise ValidationError(f"Ошибка анализа запроса: {str(e)}")


@router.get(
    "/index-suggestions",
    summary="Предложения по индексам",
    description="Генерация предложений CREATE INDEX для запроса",
    response_model=IndexSuggestionResponse,
)
async def get_index_suggestions(
    query: str = Query(..., description="SQL запрос"),
    current_user: dict = Depends(get_current_user),
):
    """Предложения по индексам для оптимизации запроса"""
    db_path = "data/nanoprobe.db"
    
    try:
        analyzer = QueryAnalyzer(db_path)
        suggestions = analyzer.suggest_indexes(query)
        analyzer.close()
        
        return IndexSuggestionResponse(
            query=query,
            suggestions=suggestions,
        )
    except Exception as e:
        raise ValidationError(f"Ошибка генерации индексов: {str(e)}")


@router.get(
    "/table-stats/{table_name}",
    summary="Статистика таблицы",
    description="Получение статистики таблицы (количество строк, размер, индексы)",
    response_model=TableStatsResponse,
)
async def get_table_stats(
    table_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Статистика таблицы"""
    db_path = "data/nanoprobe.db"
    
    try:
        analyzer = QueryAnalyzer(db_path)
        stats = analyzer.get_table_stats(table_name)
        analyzer.close()

        if not stats:
            raise NotFoundError(f"Таблица '{table_name}' не найдена")

        return TableStatsResponse(
            table_name=table_name,
            row_count=stats.get("row_count", 0),
            size_bytes=stats.get("size_bytes", 0),
            index_count=stats.get("index_count", 0),
            indexes=stats.get("indexes", []),
            columns=stats.get("columns", []),
        )
    except Exception as e:
        raise ValidationError(f"Ошибка получения статистики: {str(e)}")


@router.get(
    "/slow-queries",
    summary="Медленные запросы",
    description="Получение списка медленных запросов",
)
async def get_slow_queries(
    threshold_ms: float = Query(100, description="Порог медленного запроса (мс)"),
    limit: int = Query(10, description="Максимальное количество запросов"),
    current_user: dict = Depends(get_current_user),
):
    """Список медленных запросов"""
    db_path = "data/nanoprobe.db"
    
    try:
        analyzer = QueryAnalyzer(db_path)
        slow_queries = analyzer.get_slow_queries(threshold_ms, limit)
        analyzer.close()

        return {"slow_queries": slow_queries}
    except Exception as e:
        raise ValidationError(f"Ошибка получения медленных запросов: {str(e)}")
