# -*- coding: utf-8 -*-
"""
GraphQL API для Nanoprobe Sim Lab
Schema и резолверы для GraphQL endpoint
"""

import strawberry
from typing import List, Optional
from datetime import datetime
from utils.database import DatabaseManager


@strawberry.type
class Scan:
    """Тип сканирования"""
    id: int
    timestamp: str
    scan_type: str
    surface_type: Optional[str]
    width: Optional[int]
    height: Optional[int]
    file_path: Optional[str]
    created_at: str


@strawberry.type
class Simulation:
    """Тип симуляции"""
    id: int
    simulation_id: str
    simulation_type: str
    status: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: Optional[float]
    created_at: str


@strawberry.type
class Image:
    """Тип изображения"""
    id: int
    image_path: str
    image_type: Optional[str]
    source: Optional[str]
    width: Optional[int]
    height: Optional[int]
    channels: Optional[int]
    created_at: str


@strawberry.type
class DefectAnalysis:
    """Тип анализа дефектов"""
    id: int
    image_path: str
    defect_type: Optional[str]
    confidence: Optional[float]
    defect_count: Optional[int]
    created_at: str


@strawberry.type
class SurfaceComparison:
    """Тип сравнения поверхностей"""
    id: int
    comparison_id: str
    image1_path: str
    image2_path: str
    similarity_score: Optional[float]
    created_at: str


@strawberry.type
class DashboardStats:
    """Статистика дашборда"""
    total_scans: int
    total_simulations: int
    total_images: int
    total_analyses: int
    total_comparisons: int
    active_simulations: int


@strawberry.type
class Query:
    """GraphQL Query"""

    @strawberry.field
    def scans(self, limit: int = 50) -> List[Scan]:
        """Получить список сканирований"""
        db = DatabaseManager()
        results = db.get_scan_results(limit=limit)
        return [
            Scan(
                id=r['id'],
                timestamp=r['timestamp'],
                scan_type=r['scan_type'],
                surface_type=r.get('surface_type'),
                width=r.get('width'),
                height=r.get('height'),
                file_path=r.get('file_path'),
                created_at=r.get('created_at', '')
            )
            for r in results
        ]

    @strawberry.field
    def scan(self, scan_id: int) -> Optional[Scan]:
        """Получить сканирование по ID"""
        db = DatabaseManager()
        results = db.get_scan_results(limit=100)
        for r in results:
            if r['id'] == scan_id:
                return Scan(
                    id=r['id'],
                    timestamp=r['timestamp'],
                    scan_type=r['scan_type'],
                    surface_type=r.get('surface_type'),
                    width=r.get('width'),
                    height=r.get('height'),
                    file_path=r.get('file_path'),
                    created_at=r.get('created_at', '')
                )
        return None

    @strawberry.field
    def simulations(self, limit: int = 50) -> List[Simulation]:
        """Получить список симуляций"""
        db = DatabaseManager()
        results = db.get_simulations(limit=limit)
        return [
            Simulation(
                id=r['id'],
                simulation_id=r['simulation_id'],
                simulation_type=r['simulation_type'],
                status=r.get('status'),
                start_time=r.get('start_time'),
                end_time=r.get('end_time'),
                duration_seconds=r.get('duration_seconds'),
                created_at=r.get('created_at', '')
            )
            for r in results
        ]

    @strawberry.field
    def images(self, limit: int = 50) -> List[Image]:
        """Получить список изображений"""
        db = DatabaseManager()
        results = db.get_images(limit=limit)
        return [
            Image(
                id=r['id'],
                image_path=r['image_path'],
                image_type=r.get('image_type'),
                source=r.get('source'),
                width=r.get('width'),
                height=r.get('height'),
                channels=r.get('channels'),
                created_at=r.get('created_at', '')
            )
            for r in results
        ]

    @strawberry.field
    def stats(self) -> DashboardStats:
        """Получить статистику дашборда"""
        db = DatabaseManager()
        stats = db.get_statistics()
        return DashboardStats(
            total_scans=stats.get('total_scans', 0),
            total_simulations=stats.get('total_simulations', 0),
            total_images=stats.get('total_images', 0),
            total_analyses=stats.get('total_defect_analyses', 0),
            total_comparisons=stats.get('total_comparisons', 0),
            active_simulations=stats.get('active_simulations', 0)
        )


@strawberry.type
class Mutation:
    """GraphQL Mutation"""

    @strawberry.mutation
    def create_scan(
        self,
        scan_type: str,
        surface_type: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Scan:
        """Создать новое сканирование"""
        db = DatabaseManager()
        scan_id = db.add_scan_result(
            scan_type=scan_type,
            surface_type=surface_type,
            width=width,
            height=height
        )
        
        results = db.get_scan_results(limit=1)
        if results:
            r = results[0]
            return Scan(
                id=r['id'],
                timestamp=r['timestamp'],
                scan_type=r['scan_type'],
                surface_type=r.get('surface_type'),
                width=r.get('width'),
                height=r.get('height'),
                file_path=r.get('file_path'),
                created_at=r.get('created_at', '')
            )
        
        return Scan(
            id=scan_id,
            timestamp=datetime.now().isoformat(),
            scan_type=scan_type,
            surface_type=surface_type,
            width=width,
            height=height,
            file_path=None,
            created_at=datetime.now().isoformat()
        )


schema = strawberry.Schema(query=Query, mutation=Mutation)
