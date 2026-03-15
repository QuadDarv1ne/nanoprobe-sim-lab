"""Quick file mover for utils reorganization"""
import shutil
from pathlib import Path

UTILS = Path("utils")

# File mappings
MOVES = [
    # Core
    ("cli_utils.py", "core"),
    ("error_handler.py", "core"),
    
    # API
    ("space_image_downloader.py", "api"),
    
    # Caching
    ("cache_manager.py", "caching"),
    ("redis_cache.py", "caching"),
    ("circuit_breaker.py", "caching"),
    
    # Security
    ("rate_limiter.py", "security"),
    ("two_factor_auth.py", "security"),
    
    # Monitoring
    ("system_monitor.py", "monitoring"),
    ("enhanced_monitor.py", "monitoring"),
    ("system_health_monitor.py", "monitoring"),
    ("performance_monitor.py", "monitoring"),
    ("performance_monitoring_center.py", "monitoring"),
    ("realtime_dashboard.py", "monitoring"),
    
    # Performance
    ("performance_profiler.py", "performance"),
    ("performance_benchmark.py", "performance"),
    ("performance_analytics_dashboard.py", "performance"),
    ("memory_tracker.py", "performance"),
    ("resource_optimizer.py", "performance"),
    ("ai_resource_optimizer.py", "performance"),
    ("optimization_orchestrator.py", "performance"),
    ("optimization_logging_manager.py", "performance"),
    ("self_healing_system.py", "performance"),
    ("automated_optimization_scheduler.py", "performance"),
    
    # Data
    ("data_manager.py", "data"),
    ("data_validator.py", "data"),
    ("data_integrity.py", "data"),
    ("data_exporter.py", "data"),
    
    # Config
    ("config_manager.py", "config"),
    ("config_optimizer.py", "config"),
    ("config_validator.py", "config"),
    
    # Reporting
    ("report_generator.py", "reporting"),
    ("pdf_report_generator.py", "reporting"),
    ("documentation_generator.py", "reporting"),
    
    # AI
    ("machine_learning.py", "ai"),
    ("model_trainer.py", "ai"),
    ("defect_analyzer.py", "ai"),
    ("pretrained_defect_analyzer.py", "ai"),
    
    # Deployment
    ("deployment_manager.py", "deployment"),
    
    # Logging
    ("logger.py", "logging"),
    ("production_logger.py", "logging"),
    ("advanced_logger_analyzer.py", "logging"),
    
    # Visualization
    ("visualizer.py", "visualization"),
    ("analytics.py", "visualization"),
    ("spm_realtime_visualizer.py", "visualization"),
    ("surface_comparator.py", "visualization"),
    
    # Simulator
    ("simulator_orchestrator.py", "simulator"),
    
    # Testing
    ("test_framework.py", "testing"),
    
    # Dev
    ("code_analyzer.py", "dev"),
    
    # Batch
    ("batch_processor.py", "batch"),
]

moved = 0
errors = 0

for src_name, dst_dir in MOVES:
    src = UTILS / src_name
    dst = UTILS / dst_dir / src_name
    
    if src.exists():
        try:
            shutil.move(str(src), str(dst))
            print(f"✓ {src_name} → {dst_dir}/")
            moved += 1
        except Exception as e:
            print(f"✗ Error moving {src_name}: {e}")
            errors += 1
    else:
        print(f"⊘ Not found: {src_name}")

print(f"\nMoved: {moved}, Errors: {errors}")
