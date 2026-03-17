"""
Utils Package for Nanoprobe Sim Lab

Базовые утилиты проекта.
"""

__version__ = "2.0.0"

# Re-exports for backward compatibility
# Config module
from utils.config.config_manager import ConfigManager
from utils.config.config_optimizer import ConfigOptimizer
from utils.config.config_validator import ConfigValidator

# Security module
from utils.security.rate_limiter import RateLimiter
from utils.security.two_factor_auth import TwoFactorAuth

# Data module
from utils.data.data_exporter import DataExporter
from utils.data.data_integrity import DataIntegrityChecker
from utils.data.data_manager import DataManager
from utils.data.data_validator import DataValidator

# Core module
from utils.core.error_handler import ErrorInfo, ErrorSeverity

# AI module
from utils.ai.defect_analyzer import DefectDetector, DefectAnalysisPipeline, AdvancedDefectAnalyzer

# Reporting module
from utils.reporting.pdf_report_generator import ScientificPDFReport, generate_pdf_report

# Logging
from utils.logger import setup_project_logging

# Backup
from utils.backup_manager import BackupManager
