"""Initial schema

Revision ID: initial
Revises:
Create Date: 2026-03-12

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Таблица результатов сканирований
    """TODO: Add description"""
    op.create_table('scan_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.Text(), nullable=False),
        sa.Column('scan_type', sa.Text(), nullable=False),
        sa.Column('surface_type', sa.Text(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_scan_timestamp', 'scan_results', ['timestamp'], unique=False)
    op.create_index('idx_scan_type', 'scan_results', ['scan_type'], unique=False)
    op.create_index('idx_scan_type_timestamp', 'scan_results', ['scan_type', 'timestamp'], unique=False)
    op.create_index('idx_scan_file_path', 'scan_results', ['file_path'], unique=False)

    # Таблица симуляций
    op.create_table('simulations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('simulation_id', sa.Text(), nullable=False),
        sa.Column('simulation_type', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('start_time', sa.Text(), nullable=True),
        sa.Column('end_time', sa.Text(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('parameters', sa.Text(), nullable=True),
        sa.Column('results_summary', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('simulation_id')
    )
    op.create_index('idx_simulations_status_created', 'simulations', ['status', 'created_at'], unique=False)
    op.create_index('idx_simulation_status', 'simulations', ['status'], unique=False)

    # Таблица изображений
    op.create_table('images',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_path', sa.Text(), nullable=False),
        sa.Column('image_type', sa.Text(), nullable=True),
        sa.Column('source', sa.Text(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('channels', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('processed', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_path')
    )
    op.create_index('idx_image_type', 'images', ['image_type'], unique=False)

    # Таблица экспорта
    op.create_table('exports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('export_path', sa.Text(), nullable=False),
        sa.Column('export_format', sa.Text(), nullable=False),
        sa.Column('source_type', sa.Text(), nullable=True),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('export_path')
    )

    # Таблица сравнений поверхностей
    op.create_table('surface_comparisons',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('comparison_id', sa.Text(), nullable=False),
        sa.Column('image1_path', sa.Text(), nullable=False),
        sa.Column('image2_path', sa.Text(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=True),
        sa.Column('difference_map_path', sa.Text(), nullable=True),
        sa.Column('metrics', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('comparison_id')
    )
    op.create_index('idx_comparison_timestamp', 'surface_comparisons', ['created_at'], unique=False)

    # Таблица анализа дефектов
    op.create_table('defect_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('analysis_id', sa.Text(), nullable=False),
        sa.Column('image_path', sa.Text(), nullable=False),
        sa.Column('model_name', sa.Text(), nullable=True),
        sa.Column('defects_detected', sa.Integer(), nullable=True),
        sa.Column('defects_data', sa.Text(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('analysis_id')
    )
    op.create_index('idx_defect_image', 'defect_analysis', ['image_path'], unique=False)

    # Таблица PDF отчётов
    op.create_table('pdf_reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_path', sa.Text(), nullable=False),
        sa.Column('report_type', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('source_ids', sa.Text(), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('pages_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('report_path')
    )

    # Таблица пакетных заданий
    op.create_table('batch_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Text(), nullable=False),
        sa.Column('job_type', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('total_items', sa.Integer(), nullable=True),
        sa.Column('processed_items', sa.Integer(), nullable=True),
        sa.Column('failed_items', sa.Integer(), nullable=True),
        sa.Column('parameters', sa.Text(), nullable=True),
        sa.Column('results_summary', sa.Text(), nullable=True),
        sa.Column('started_at', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id')
    )
    op.create_index('idx_batch_status', 'batch_jobs', ['status'], unique=False)

    # Таблица метрик производительности
    op.create_table('performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.Text(), nullable=False),
        sa.Column('metric_type', sa.Text(), nullable=False),
        sa.Column('metric_name', sa.Text(), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.Text(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_metrics_timestamp', 'performance_metrics', ['timestamp'], unique=False)


def downgrade():
    """TODO: Add description"""
    op.drop_index('idx_metrics_timestamp', table_name='performance_metrics')
    op.drop_table('performance_metrics')
    op.drop_index('idx_batch_status', table_name='batch_jobs')
    op.drop_table('batch_jobs')
    op.drop_table('pdf_reports')
    op.drop_index('idx_defect_image', table_name='defect_analysis')
    op.drop_table('defect_analysis')
    op.drop_index('idx_comparison_timestamp', table_name='surface_comparisons')
    op.drop_table('surface_comparisons')
    op.drop_table('exports')
    op.drop_index('idx_image_type', table_name='images')
    op.drop_table('images')
    op.drop_index('idx_simulation_status', table_name='simulations')
    op.drop_index('idx_simulations_status_created', table_name='simulations')
    op.drop_table('simulations')
    op.drop_index('idx_scan_file_path', table_name='scan_results')
    op.drop_index('idx_scan_type_timestamp', table_name='scan_results')
    op.drop_index('idx_scan_type', table_name='scan_results')
    op.drop_index('idx_scan_timestamp', table_name='scan_results')
    op.drop_table('scan_results')
