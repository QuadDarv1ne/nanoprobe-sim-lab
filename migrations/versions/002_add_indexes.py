"""Add indexes for performance

Revision ID: 002
Revises: initial
Create Date: 2026-03-14

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002'
down_revision = 'initial'
branch_labels = None
depends_on = None


def upgrade():
    # ===== scan_results =====
    # Индекс для частых запросов по created_at
    op.create_index('idx_scan_created_at', 'scan_results', ['created_at'], unique=False)
    
    # Индекс для подсчёта количества сканирований по типу
    op.create_index('idx_scan_type_count', 'scan_results', ['scan_type', 'id'], unique=False)
    
    # ===== simulations =====
    # Индекс для запросов по created_at
    op.create_index('idx_simulations_created_at', 'simulations', ['created_at'], unique=False)
    
    # Индекс для фильтрации по статусу и дате
    op.create_index('idx_simulations_status_date', 'simulations', ['status', 'start_time'], unique=False)
    
    # ===== images =====
    # Индекс для запросов по дате создания
    op.create_index('idx_image_created_at', 'images', ['created_at'], unique=False)
    
    # Индекс для processed flag
    op.create_index('idx_image_processed', 'images', ['processed'], unique=False)
    
    # Композитный индекс для фильтрации по типу и дате
    op.create_index('idx_image_type_created', 'images', ['image_type', 'created_at'], unique=False)
    
    # ===== exports =====
    # Индекс для запросов по дате
    op.create_index('idx_export_created_at', 'exports', ['created_at'], unique=False)
    
    # Индекс для фильтрации по источнику
    op.create_index('idx_export_source', 'exports', ['source_type', 'source_id'], unique=False)
    
    # ===== surface_comparisons =====
    # Индекс для запросов по дате
    op.create_index('idx_comparison_created_at_full', 'surface_comparisons', ['created_at'], unique=False)
    
    # ===== defect_analyses (если существует) =====
    # Индексы добавляются при создании таблицы


def downgrade():
    # ===== scan_results =====
    op.drop_index('idx_scan_created_at', 'scan_results')
    op.drop_index('idx_scan_type_count', 'scan_results')
    
    # ===== simulations =====
    op.drop_index('idx_simulations_created_at', 'simulations')
    op.drop_index('idx_simulations_status_date', 'simulations')
    
    # ===== images =====
    op.drop_index('idx_image_created_at', 'images')
    op.drop_index('idx_image_processed', 'images')
    op.drop_index('idx_image_type_created', 'images')
    
    # ===== exports =====
    op.drop_index('idx_export_created_at', 'exports')
    op.drop_index('idx_export_source', 'exports')
    
    # ===== surface_comparisons =====
    op.drop_index('idx_comparison_created_at_full', 'surface_comparisons')
