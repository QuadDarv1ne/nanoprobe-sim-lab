"""
Utils Package

Реорганизованная структура утилит (2026-03-15):

core/           - Базовые утилиты (CLI, errors)
api/            - API клиенты (NASA, external)
database/       - Database utilities
security/       - Security (auth, rate limiting, 2FA)
monitoring/     - Monitoring (system, performance, health)
performance/    - Performance optimization
caching/        - Caching (Redis, cache manager)
batch/          - Batch processing
logging/        - Logging utilities
visualization/  - Visualization components
simulator/      - Simulator orchestration
testing/        - Testing frameworks
dev/            - Development tools
ai/             - AI/ML utilities
config/         - Configuration management
data/           - Data management
reporting/      - Report generation
deployment/     - Deployment utilities
"""

__version__ = "2.0.0"

# Core exports
from utils.core.cli_utils import *
from utils.core.error_handler import *

# API exports
from utils.api.nasa_api_client import get_nasa_client, NASAAPIClient

# Database exports
from utils.database.database import DatabaseManager

# Security exports
from utils.security.rate_limiter import limiter
from utils.security.two_factor_auth import get_2fa_manager

# Caching exports
from utils.caching.redis_cache import cache, cached, cached_sync
from utils.caching.cache_manager import CacheManager
from utils.caching.circuit_breaker import circuit_breaker

# Monitoring exports
from utils.monitoring.system_monitor import SystemMonitor
from utils.monitoring.enhanced_monitor import EnhancedMonitor

# Config exports
from utils.config.config_manager import ConfigManager

# Data exports
from utils.data.data_manager import DataManager

# Reporting exports
from utils.reporting.report_generator import ReportGenerator

# Deployment exports
from utils.deployment.deployment_manager import DeploymentManager

__all__ = [
    # Core
    'cli_utils',
    'error_handler',
    
    # API
    'get_nasa_client',
    'NASAAPIClient',
    
    # Database
    'DatabaseManager',
    
    # Security
    'limiter',
    'get_2fa_manager',
    
    # Caching
    'cache',
    'cached',
    'cached_sync',
    'CacheManager',
    'circuit_breaker',
    
    # Monitoring
    'SystemMonitor',
    'EnhancedMonitor',
    
    # Config
    'ConfigManager',
    
    # Data
    'DataManager',
    
    # Reporting
    'ReportGenerator',
    
    # Deployment
    'DeploymentManager',
]
