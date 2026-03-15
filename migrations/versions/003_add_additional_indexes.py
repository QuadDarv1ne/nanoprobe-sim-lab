"""Add additional indexes for query optimization

Revision ID: 003
Revises: 002
Create Date: 2026-03-15

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    """
    Создание дополнительных индексов для оптимизации запросов.
    
    Добавляются индексы для:
    - Частых WHERE условий
    - JOIN операций
    - ORDER BY сортировок
    - Foreign key полей
    """
    # ===== scan_results =====
    # Индекс для фильтрации по surface_type (частый WHERE)
    op.create_index('idx_scan_surface_type', 'scan_results', ['surface_type'], unique=False)
    
    # Индекс для сортировки по дате (частый ORDER BY)
    op.create_index('idx_scan_created_at_desc', 'scan_results', ['created_at', 'id'], unique=False)
    
    # Композитный индекс для частых запросов (surface_type + created_at)
    op.create_index('idx_scan_surface_date', 'scan_results', ['surface_type', 'created_at'], unique=False)
    
    # ===== simulations =====
    # Индекс для фильтрации по simulation_type
    op.create_index('idx_simulations_type', 'simulations', ['simulation_type'], unique=False)
    
    # Индекс для фильтрации по status (активные симуляции)
    op.create_index('idx_simulations_status', 'simulations', ['status'], unique=False)
    
    # Композитный индекс для запросов status + created_at
    op.create_index('idx_simulations_status_created', 'simulations', ['status', 'created_at'], unique=False)
    
    # ===== images =====
    # Индекс для фильтрации по source_type
    op.create_index('idx_image_source_type', 'images', ['source_type'], unique=False)
    
    # Индекс для foreign key scan_id
    op.create_index('idx_image_scan_id', 'images', ['scan_id'], unique=False)
    
    # Композитный индекс для processed + created_at
    op.create_index('idx_image_processed_date', 'images', ['processed', 'created_at'], unique=False)
    
    # ===== exports =====
    # Индекс для фильтрации по export_format
    op.create_index('idx_export_format', 'exports', ['export_format'], unique=False)
    
    # Индекс для foreign key source_id
    op.create_index('idx_export_source_id', 'exports', ['source_id'], unique=False)
    
    # ===== surface_comparisons =====
    # Индекс для фильтрации по surface1_id
    op.create_index('idx_comparison_surface1', 'surface_comparisons', ['surface1_id'], unique=False)
    
    # Индекс для фильтрации по surface2_id
    op.create_index('idx_comparison_surface2', 'surface_comparisons', ['surface2_id'], unique=False)
    
    # ===== defect_analysis =====
    # Индекс для foreign key image_id
    op.create_index('idx_defect_analysis_image_id', 'defect_analysis', ['image_id'], unique=False)
    
    # Индекс для фильтрации по defect_type
    op.create_index('idx_defect_type', 'defect_analysis', ['defect_type'], unique=False)
    
    # ===== pdf_reports =====
    # Индекс для foreign key simulation_id
    op.create_index('idx_pdf_report_simulation_id', 'pdf_reports', ['simulation_id'], unique=False)
    
    # Индекс для фильтрации по generated_at
    op.create_index('idx_pdf_report_generated_at', 'pdf_reports', ['generated_at'], unique=False)
    
    # ===== batch_jobs =====
    # Индекс для фильтрации по status
    op.create_index('idx_batch_jobs_status', 'batch_jobs', ['status'], unique=False)
    
    # Индекс для foreign key user_id
    op.create_index('idx_batch_jobs_user_id', 'batch_jobs', ['user_id'], unique=False)
    
    # Композитный индекс для status + created_at
    op.create_index('idx_batch_jobs_status_created', 'batch_jobs', ['status', 'created_at'], unique=False)


def downgrade():
    """
    Удаление дополнительных индексов (откат миграции).
    """
    # ===== scan_results =====
    op.drop_index('idx_scan_surface_type', 'scan_results')
    op.drop_index('idx_scan_created_at_desc', 'scan_results')
    op.drop_index('idx_scan_surface_date', 'scan_results')
    
    # ===== simulations =====
    op.drop_index('idx_simulations_type', 'simulations')
    op.drop_index('idx_simulations_status', 'simulations')
    op.drop_index('idx_simulations_status_created', 'simulations')
    
    # ===== images =====
    op.drop_index('idx_image_source_type', 'images')
    op.drop_index('idx_image_scan_id', 'images')
    op.drop_index('idx_image_processed_date', 'images')
    
    # ===== exports =====
    op.drop_index('idx_export_format', 'exports')
    op.drop_index('idx_export_source_id', 'exports')
    
    # ===== surface_comparisons =====
    op.drop_index('idx_comparison_surface1', 'surface_comparisons')
    op.drop_index('idx_comparison_surface2', 'surface_comparisons')
    
    # ===== defect_analysis =====
    op.drop_index('idx_defect_analysis_image_id', 'defect_analysis')
    op.drop_index('idx_defect_type', 'defect_analysis')
    
    # ===== pdf_reports =====
    op.drop_index('idx_pdf_report_simulation_id', 'pdf_reports')
    op.drop_index('idx_pdf_report_generated_at', 'pdf_reports')
    
    # ===== batch_jobs =====
    op.drop_index('idx_batch_jobs_status', 'batch_jobs')
    op.drop_index('idx_batch_jobs_user_id', 'batch_jobs')
    op.drop_index('idx_batch_jobs_status_created', 'batch_jobs')
