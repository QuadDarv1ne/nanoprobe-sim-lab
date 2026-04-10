"""
Pydantic схемы для Nanoprobe Sim Lab API
Валидация данных запросов и ответов
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum
import re


# ==================== Типы для аннотаций (Python 3.10+) ====================
# Для совместимости с Python 3.8-3.9 используем typing
# Для Python 3.10+ можно использовать: dict, list, optional напрямую


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
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Валидация сложности пароля"""
        if len(v) < 8:
            raise ValueError('Пароль должен быть не менее 8 символов')
        if len(v) > 128:
            raise ValueError('Пароль не должен превышать 128 символов')
        if not re.search(r'[A-ZА-ЯЁ]', v):
            raise ValueError('Пароль должен содержать заглавную букву')
        if not re.search(r'\d', v):
            raise ValueError('Пароль должен содержать цифру')
        if not re.search(r'[a-zа-яё]', v):
            raise ValueError('Пароль должен содержать строчную букву')
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            raise ValueError('Пароль должен содержать специальный символ')
        return v


class RefreshTokenRequest(BaseModel):
    """Запрос на обновление токена"""
    refresh_token: str = Field(..., description="Refresh токен")


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
    severity: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    path: Optional[str] = None


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


# ==================== Пагинация ====================

class PaginationParams(BaseModel):
    """Параметры пагинации"""
    page: int = Field(1, ge=1, description="Номер страницы")
    page_size: int = Field(20, ge=1, le=100, description="Размер страницы")

    @property
    def offset(self) -> int:
        """Вычисляет смещение для пагинации"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Возвращает лимит записей на страницу"""
        return self.page_size


class PaginatedResponse(BaseModel):
    """Пагинированный ответ"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


# ==================== Дашборд ====================

class DashboardStats(BaseModel):
    """Статистика дашборда"""
    total_scans: int
    total_simulations: int
    active_simulations: int
    storage_used_mb: float
    storage_total_mb: float
    recent_scans_count: int
    recent_simulations_count: int
    success_rate: float
    # Расширенная статистика из БД
    total_images: int = 0
    total_exports: int = 0
    total_comparisons: int = 0
    total_defect_analyses: int = 0
    total_pdf_reports: int = 0
    total_batch_jobs: int = 0
    active_batch_jobs: int = 0
    scans_by_type: Dict[str, int] = {}
    db_size_mb: float = 0.0


class HealthStatus(BaseModel):
    """Статус здоровья"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: int
    services: Dict[str, str]


class SystemHealth(BaseModel):
    """Системное здоровье"""
    status: str
    timestamp: str
    version: str
    metrics: Dict[str, Any]
    issues: List[str]
    services: Dict[str, str]


class RealtimeMetrics(BaseModel):
    """Метрики в реальном времени"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_upload_mbps: float
    network_download_mbps: float


# ==================== Экспорт ====================

class ExportRequest(BaseModel):
    """Запрос на экспорт"""
    format: str = Field(..., pattern="^(json|csv|pdf|xlsx)$")
    scan_ids: Optional[List[int]] = None
    include_metadata: bool = True


class ExportResponse(BaseModel):
    """Ответ экспорта"""
    export_id: str
    format: str
    status: str
    file_path: Optional[str]
    download_url: Optional[str]
    file_size_bytes: Optional[int]
    created_at: str
    expires_at: str


# ==================== NASA API ====================

class APODResponse(BaseModel):
    """NASA APOD ответ"""
    date: str
    explanation: str
    title: str
    url: Optional[str] = None
    hdurl: Optional[str] = None
    media_type: Optional[str] = None
    service_version: Optional[str] = None
    copyright: Optional[str] = None


class MarsPhoto(BaseModel):
    """Фото с марсохода"""
    id: int
    sol: int
    camera: Dict[str, Any]
    img_src: str
    earth_date: str


class MarsPhotosResponse(BaseModel):
    """Ответ фото с Марса"""
    photos: List[MarsPhoto]
    total: int


class NEOCloseApproach(BaseModel):
    """Сближение с Землёй"""
    orbiting_body: str
    miss_distance: Dict[str, Any]
    relative_velocity: Dict[str, Any]


class NearEarthObject(BaseModel):
    """Околоземный объект"""
    id: str
    name: str
    diameter: Dict[str, Any]
    is_potentially_hazardous_asteroid: bool
    close_approach_data: List[NEOCloseApproach]


class NEOsResponse(BaseModel):
    """Ответ NEO"""
    near_earth_objects: List[NearEarthObject]
    total: int
