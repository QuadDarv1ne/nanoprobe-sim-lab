"""
Utilities for Nanoprobe Sim Lab

Реорганизованная структура (2026-03-15):
- monitoring/ - логирование, мониторинг системы
- performance/ - профилирование, бенчмарки, оптимизация
- security/ - безопасность, аутентификация, rate limiting
- data/ - работа с данными, БД, кэширование
- ai/ - AI/ML модели, анализ дефектов
- reporting/ - отчёты, документация
- config/ - конфигурация, валидация
- deployment/ - деплой, оркестрация

Для обратной совместимости все модули доступны напрямую:
    from utils import DatabaseManager, ConfigManager, SystemMonitor
"""

# Monitoring
from utils.monitoring.logger import NanoprobeLogger
from utils.monitoring.production_logger import ProductionLogger
from utils.monitoring.system_monitor import SystemMonitor
from utils.monitoring.system_health_monitor import SystemHealthMonitor
from utils.monitoring.enhanced_monitor import get_monitor, format_uptime
from utils.monitoring.advanced_logger_analyzer import LoggerAnalyzer
from utils.monitoring.realtime_dashboard import RealtimeDashboard
from utils.monitoring.performance_monitoring_center import PerformanceMonitoringCenter

# Performance
from utils.performance.performance_monitor import PerformanceMonitor
from utils.performance.profiler import Profiler
from utils.performance.performance_benchmark import run_benchmark
from utils.performance.performance_analytics_dashboard import PerformanceAnalyticsDashboard
from utils.performance.resource_optimizer import ResourceOptimizer
from utils.performance.ai_resource_optimizer import AIResourceOptimizer
from utils.performance.predictive_analytics_engine import PredictiveAnalyticsEngine
from utils.performance.performance_verification_framework import PerformanceVerificationFramework
from utils.performance.performance_profiler import PerformanceProfiler
from utils.performance.memory_tracker import MemoryTracker

# Security
from utils.security.error_handler import ErrorHandler, APIError, ValidationError, NotFoundError, AuthenticationError, AuthorizationError
from utils.security.two_factor_auth import TwoFactorAuth
from utils.security.circuit_breaker import CircuitBreaker, circuit_breaker
from utils.security.rate_limiter import limiter, auth_limit, api_limit, write_limit

# Data
from utils.data.database import DatabaseManager, get_database
from utils.data.redis_cache import RedisCache, cache, cached
from utils.data.data_manager import DataManager
from utils.data.data_exporter import DataExporter
from utils.data.data_validator import DataValidator
from utils.data.data_integrity import DataIntegrity
from utils.data.cache_manager import CacheManager
from utils.data.backup_manager import BackupManager
from utils.data.batch_processor import BatchProcessor
from utils.data.surface_comparator import compare_surfaces, SurfaceComparator

# AI/ML
from utils.ai.defect_analyzer import analyze_defects, DefectAnalyzer
from utils.ai.pretrained_defect_analyzer import PretrainedDefectAnalyzer
from utils.ai.machine_learning import MLModel, train_model, predict
from utils.ai.model_trainer import ModelTrainer
from utils.ai.code_analyzer import CodeAnalyzer
from utils.ai.visualizer import Visualizer
from utils.ai.spm_realtime_visualizer import SPMRealtimeVisualizer
from utils.ai.space_image_downloader import SpaceImageDownloader

# Reporting
from utils.reporting.report_generator import ReportGenerator
from utils.reporting.pdf_report_generator import PDFReportGenerator
from utils.reporting.documentation_generator import DocumentationGenerator
from utils.reporting.analytics import Analytics
from utils.reporting.enhanced_monitor import EnhancedMonitor

# Config
from utils.config.config_manager import ConfigManager
from utils.config.config_validator import ConfigValidator
from utils.config.config_optimizer import ConfigOptimizer
from utils.config.cli_utils import CLIUtils

# Deployment
from utils.deployment.deployment_manager import DeploymentManager
from utils.deployment.simulator_orchestrator import SimulatorOrchestrator
from utils.deployment.optimization_orchestrator import OptimizationOrchestrator
from utils.deployment.optimization_logging_manager import OptimizationLoggingManager
from utils.deployment.automated_optimization_scheduler import AutomatedOptimizationScheduler
from utils.deployment.self_healing_system import SelfHealingSystem
from utils.deployment.test_framework import TestFramework

# Legacy imports for backward compatibility
__all__ = [
    # Monitoring
    'NanoprobeLogger',
    'ProductionLogger',
    'SystemMonitor',
    'SystemHealthMonitor',
    'get_monitor',
    'format_uptime',
    'LoggerAnalyzer',
    'RealtimeDashboard',
    'PerformanceMonitoringCenter',
    'enhanced_monitor',
    
    # Performance
    'PerformanceMonitor',
    'Profiler',
    'run_benchmark',
    'PerformanceAnalyticsDashboard',
    'ResourceOptimizer',
    'AIResourceOptimizer',
    'PredictiveAnalyticsEngine',
    'PerformanceVerificationFramework',
    'PerformanceProfiler',
    'MemoryTracker',
    
    # Security
    'ErrorHandler',
    'APIError',
    'ValidationError',
    'NotFoundError',
    'AuthenticationError',
    'AuthorizationError',
    'TwoFactorAuth',
    'CircuitBreaker',
    'circuit_breaker',
    'limiter',
    'auth_limit',
    'api_limit',
    'write_limit',
    
    # Data
    'DatabaseManager',
    'get_database',
    'RedisCache',
    'cache',
    'cached',
    'DataManager',
    'DataExporter',
    'DataValidator',
    'DataIntegrity',
    'CacheManager',
    'BackupManager',
    'BatchProcessor',
    'compare_surfaces',
    'SurfaceComparator',
    
    # AI/ML
    'analyze_defects',
    'DefectAnalyzer',
    'PretrainedDefectAnalyzer',
    'MLModel',
    'train_model',
    'predict',
    'ModelTrainer',
    'CodeAnalyzer',
    'Visualizer',
    'SPMRealtimeVisualizer',
    'SpaceImageDownloader',
    
    # Reporting
    'ReportGenerator',
    'PDFReportGenerator',
    'DocumentationGenerator',
    'Analytics',
    'EnhancedMonitor',
    
    # Config
    'ConfigManager',
    'ConfigValidator',
    'ConfigOptimizer',
    'CLIUtils',
    
    # Deployment
    'DeploymentManager',
    'SimulatorOrchestrator',
    'OptimizationOrchestrator',
    'OptimizationLoggingManager',
    'AutomatedOptimizationScheduler',
    'SelfHealingSystem',
    'TestFramework',
]

# Модули которые остаются в корне utils (для обратной совместимости)
# Эти файлы физически остаются в utils/
from utils import machine_learning
from utils import deployment_manager
from utils import documentation_generator
from utils import analytics
from utils import enhanced_monitor
from utils import realtime_dashboard
from utils import performance_monitoring_center
from utils import advanced_logger_analyzer
from utils import ai_resource_optimizer
from utils import predictive_analytics_engine
from utils import performance_analytics_dashboard
from utils import performance_verification_framework
from utils import performance_profiler
from utils import profiler
from utils import memory_tracker
from utils import resource_optimizer
from utils import config_optimizer
from utils import config_validator
from utils import cli_utils
from utils import optimization_logging_manager
from utils import optimization_orchestrator
from utils import automated_optimization_scheduler
from utils import self_healing_system
from utils import test_framework
from utils import defect_analyzer
from utils import pretrained_defect_analyzer
from utils import model_trainer
from utils import code_analyzer
from utils import visualizer
from utils import spm_realtime_visualizer
from utils import space_image_downloader
from utils import report_generator
from utils import pdf_report_generator
from utils import data_integrity
from utils import data_validator
from utils import cache_manager
from utils import backup_manager
from utils import batch_processor
from utils import surface_comparator
