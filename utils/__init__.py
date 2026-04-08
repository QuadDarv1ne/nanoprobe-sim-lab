"""
Utils Package for Nanoprobe Sim Lab

Базовые утилиты проекта.
"""

__version__ = "2.0.0"

# Lazy imports for faster loading
# Import only when accessed to avoid slow startup
def __getattr__(name):
    import importlib
    
    _imports = {
        'ConfigManager': 'utils.config.config_manager',
        'ConfigOptimizer': 'utils.config.config_optimizer',
        'ConfigValidator': 'utils.config.config_validator',
        'RateLimiter': 'utils.security.rate_limiter',
        'TwoFactorAuth': 'utils.security.two_factor_auth',
        'DataExporter': 'utils.data.data_exporter',
        'DataIntegrityChecker': 'utils.data.data_integrity',
        'DataManager': 'utils.data.data_manager',
        'DataValidator': 'utils.data.data_validator',
        'ErrorInfo': 'utils.core.error_handler',
        'ErrorSeverity': 'utils.core.error_handler',
        'DefectDetector': 'utils.ai.defect_analyzer',
        'DefectAnalysisPipeline': 'utils.ai.defect_analyzer',
        'AdvancedDefectAnalyzer': 'utils.ai.defect_analyzer',
        'ScientificPDFReport': 'utils.reporting.pdf_report_generator',
        'generate_pdf_report': 'utils.reporting.pdf_report_generator',
        'setup_project_logging': 'utils.logger',
        'BackupManager': 'utils.backup_manager',
    }
    
    if name in _imports:
        module = importlib.import_module(_imports[name])
        if name == 'ErrorInfo':
            return module.ErrorInfo
        elif name == 'ErrorSeverity':
            return module.ErrorSeverity
        elif name == 'DefectDetector':
            return module.DefectDetector
        elif name == 'DefectAnalysisPipeline':
            return module.DefectAnalysisPipeline
        elif name == 'AdvancedDefectAnalyzer':
            return module.AdvancedDefectAnalyzer
        elif name == 'ScientificPDFReport':
            return module.ScientificPDFReport
        elif name == 'generate_pdf_report':
            return module.generate_pdf_report
        elif name == 'setup_project_logging':
            return module.setup_project_logging
        else:
            return getattr(module, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
