# -*- coding: utf-8 -*-
"""
Pydantic схемы для Nanoprobe Sim Lab API
Валидация данных запросов и ответов
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ==================== Аутентификация ====================

class Token(BaseModel):
    """JWT токен"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # секунд


class TokenData(BaseModel):
    """Данные из токена"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    exp: Optional[datetime] = None


class LoginRequest(BaseModel):
    """Запрос на логин"""
    username: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_]+$')
    password: str = Field(..., min_length=8)

    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Пароль должен быть не менее 8 символов')
        if not any(c.isupper() for c in v):
            raise ValueError('Пароль должен содержать заглавную букву')
        if not any(c.isdigit() for c in v):
            raise ValueError('Пароль должен содержать цифру')
        return v


class LoginResponse(BaseModel):
    """Ответ на логин"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: Dict[str, Any]


# ==================== Сканирования ====================

class ScanType(str, Enum):
    """Типы сканирований"""
    SPM = "spm"
    IMAGE = "image"
    SSTV = "sstv"


class ScanCreate(BaseModel):
    """Создание сканирования"""
    scan_type: ScanType
    surface_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ScanResponse(BaseModel):
    """Ответ со сканированием"""
    id: int
    timestamp: str
    scan_type: str
    surface_type: Optional[str]
    width: Optional[int]
    height: Optional[int]
    file_path: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str

    class Config:
        from_attributes = True


class ScanListResponse(BaseModel):
    """Список сканирований"""
    items: List[ScanResponse]
    total: int
    limit: int
    offset: int


# ==================== Симуляции ====================

class SimulationStatus(str, Enum):
    """Статусы симуляции"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SimulationCreate(BaseModel):
    """Создание симуляции"""
    simulation_type: str
    parameters: Optional[Dict[str, Any]] = None


class SimulationResponse(BaseModel):
    """Ответ с симуляцией"""
    id: int
    simulation_id: str
    simulation_type: str
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: Optional[float]
    parameters: Optional[Dict[str, Any]]
    results_summary: Optional[Dict[str, Any]]
    created_at: str

    class Config:
        from_attributes = True


class SimulationListResponse(BaseModel):
    """Список симуляций"""
    items: List[SimulationResponse]
    total: int
    limit: int


# ==================== Анализ дефектов ====================

class DefectType(str, Enum):
    """Типы дефектов"""
    PIT = "pit"
    HILLOCK = "hillock"
    SCRATCH = "scratch"
    PARTICLE = "particle"
    CRACK = "crack"


class DefectInfo(BaseModel):
    """Информация о дефекте"""
    type: str
    x: float
    y: float
    width: float
    height: float
    area: int
    confidence: float


class DefectAnalysisRequest(BaseModel):
    """Запрос на анализ дефектов"""
    image_path: str
    model_name: Optional[str] = "isolation_forest"


class DefectAnalysisResponse(BaseModel):
    """Ответ анализа дефектов"""
    analysis_id: str
    image_path: str
    model_name: str
    defects_count: int
    defects: List[DefectInfo]
    confidence_score: float
    processing_time_ms: float
    timestamp: str


# ==================== Сравнение поверхностей ====================

class SurfaceComparisonRequest(BaseModel):
    """Запрос на сравнение поверхностей"""
    image1_path: str
    image2_path: str


class ComparisonMetrics(BaseModel):
    """Метрики сравнения"""
    ssim: float
    psnr: float
    mse: float
    similarity: float
    pearson: float


class SurfaceComparisonResponse(BaseModel):
    """Ответ сравнения поверхностей"""
    comparison_id: str
    image1_path: str
    image2_path: str
    similarity_score: float
    metrics: ComparisonMetrics
    difference_map_path: Optional[str]
    created_at: str


# ==================== PDF Отчёты ====================

class ReportType(str, Enum):
    """Типы отчётов"""
    SURFACE_ANALYSIS = "surface_analysis"
    DEFECT_ANALYSIS = "defect_analysis"
    COMPARISON = "comparison"
    SIMULATION = "simulation"
    BATCH = "batch"


class PDFReportRequest(BaseModel):
    """Запрос на генерацию PDF отчёта"""
    report_type: ReportType
    title: str
    author: Optional[str] = "Nanoprobe Simulation Lab"
    data: Dict[str, Any]
    images: Optional[List[str]] = None


class PDFReportResponse(BaseModel):
    """Ответ с PDF отчётом"""
    report_id: str
    report_path: str
    report_type: str
    title: str
    file_size_bytes: int
    pages_count: Optional[int]
    created_at: str


# ==================== Пакетная обработка ====================

class BatchJobCreate(BaseModel):
    """Создание задания пакетной обработки"""
    job_type: str
    items: List[Any]
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 0


class BatchJobResponse(BaseModel):
    """Ответ задания"""
    job_id: str
    job_type: str
    status: str
    total_items: int
    processed_items: int
    failed_items: int
    progress_percent: float
    created_at: str


# ==================== Общее ====================

class ErrorResponse(BaseModel):
    """Ошибка"""
    detail: str
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class StatisticsResponse(BaseModel):
    """Статистика базы данных"""
    total_scans: int
    total_simulations: int
    active_simulations: int
    total_images: int
    total_exports: int
    total_comparisons: int
    total_defect_analyses: int
    total_pdf_reports: int
    total_batch_jobs: int
    scans_by_type: Dict[str, int]
