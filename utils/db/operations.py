"""Database CRUD operations."""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """Mixin-style class providing CRUD operations for the database."""

    def _row_to_dict(self, row) -> Dict:
        """Конвертирует строку результата в словарь."""
        result = dict(row)
        for key in [
            "metadata",
            "parameters",
            "results_summary",
            "metrics",
            "defects_data",
            "source_ids",
        ]:
            if key in result and result[key]:
                try:
                    result[key] = json.loads(result[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return result

    # ==================== Scan Operations ====================

    def add_scan_result(
        self,
        scan_type: str,
        surface_type: str = None,
        width: int = None,
        height: int = None,
        file_path: str = None,
        metadata: Dict = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO scan_results
                (timestamp, scan_type, surface_type, width, height, file_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    scan_type,
                    surface_type,
                    width,
                    height,
                    file_path,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            scan_id = cursor.lastrowid
        self.invalidate_cache("scans:")
        return scan_id

    def add_scan_result_batch(self, scan_results: List[Dict]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = []
            for scan in scan_results:
                data.append(
                    (
                        now,
                        scan.get("scan_type"),
                        scan.get("surface_type"),
                        scan.get("width"),
                        scan.get("height"),
                        scan.get("file_path"),
                        json.dumps(scan.get("metadata")) if scan.get("metadata") else None,
                        now,
                    )
                )
            cursor.executemany(
                """INSERT INTO scan_results
                (timestamp, scan_type, surface_type, width, height, file_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                data,
            )
            self.invalidate_cache("scans:")
            return len(data)

    def get_scan_results(
        self, scan_type: str = None, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        cache_key = self._get_cache_key(f"scans:{scan_type}:{limit}:{offset}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """SELECT id, timestamp, scan_type, surface_type, width, height,
                       file_path, metadata, created_at FROM scan_results"""
            params = []
            if scan_type:
                query += " WHERE scan_type = ?"
                params.append(scan_type)
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cursor.execute(query, params)
            result = [self._row_to_dict(row) for row in cursor.fetchall()]
            self._set_cache(cache_key, result)
            return result

    def get_scan_by_id(self, scan_id: int) -> Optional[Dict]:
        cache_key = self._get_cache_key(f"scan:id:{scan_id}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_results WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            result = self._row_to_dict(row) if row else None
            if result:
                self._set_cache(cache_key, result)
            return result

    def delete_scan(self, scan_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_results WHERE id = ?", (scan_id,))
            success = cursor.rowcount > 0
        if success:
            self.invalidate_cache("scans:")
            self.invalidate_cache(f"scan:id:{scan_id}")
        return success

    def search_scans(self, query: str, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            pattern = f"%{query}%"
            cursor.execute(
                """SELECT * FROM scan_results
                WHERE surface_type LIKE ? OR metadata LIKE ?
                ORDER BY timestamp DESC LIMIT ?""",
                (pattern, pattern, limit),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def count_scans(self, scan_type: str = None) -> int:
        cache_key = self._get_cache_key(f"scans:count:{scan_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if scan_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM scan_results WHERE scan_type = ?", (scan_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM scan_results")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    # ==================== Async Scan Operations ====================

    async def get_scan_results_async(
        self, scan_type: str = None, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        cache_key = self._get_cache_key(f"scans:{scan_type}:{limit}:{offset}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM scan_results"
            params = []
            if scan_type:
                query += " WHERE scan_type = ?"
                params.append(scan_type)
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cursor.execute(query, params)
            result = [self._row_to_dict(row) for row in cursor.fetchall()]
            self._set_cache(cache_key, result)
            return result

    async def get_scan_by_id_async(self, scan_id: int) -> Optional[Dict]:
        cache_key = self._get_cache_key(f"scan:id:{scan_id}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_results WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            result = self._row_to_dict(row) if row else None
            if result:
                self._set_cache(cache_key, result)
            return result

    async def count_scans_async(self, scan_type: str = None) -> int:
        cache_key = self._get_cache_key(f"scans:count:{scan_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            if scan_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM scan_results WHERE scan_type = ?", (scan_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM scan_results")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    # ==================== Simulation Operations ====================

    def add_simulation(
        self, simulation_id: str, simulation_type: str, parameters: Dict = None
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO simulations
                (simulation_id, simulation_type, start_time, status, parameters, created_at)
                VALUES (?, ?, ?, 'running', ?, ?)""",
                (
                    simulation_id,
                    simulation_type,
                    now,
                    json.dumps(parameters) if parameters else None,
                    now,
                ),
            )
            sim_id = cursor.lastrowid
        self.invalidate_cache("simulations:")
        return sim_id

    async def add_simulation_async(
        self, simulation_id: str, simulation_type: str, parameters: Dict = None
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO simulations
                (simulation_id, simulation_type, start_time, status, parameters, created_at)
                VALUES (?, ?, ?, 'running', ?, ?)""",
                (
                    simulation_id,
                    simulation_type,
                    now,
                    json.dumps(parameters) if parameters else None,
                    now,
                ),
            )
            sim_id = cursor.lastrowid
        self.invalidate_cache("simulations:")
        return sim_id

    def update_simulation(
        self, simulation_id: str, status: str = None, results_summary: Dict = None
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            updates, params = [], []
            if status:
                updates.append("status = ?")
                params.append(status)
            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))
            if status in ("completed", "failed", "stopped"):
                updates.append("end_time = ?")
                params.append(datetime.now(timezone.utc).isoformat())
                cursor.execute(
                    "SELECT start_time FROM simulations WHERE simulation_id = ?", (simulation_id,)
                )
                row = cursor.fetchone()
                if row and row["start_time"]:
                    start = datetime.fromisoformat(row["start_time"])
                    duration = (datetime.now(timezone.utc) - start).total_seconds()
                    updates.append("duration_seconds = ?")
                    params.append(duration)
            params.append(simulation_id)
            query = f"UPDATE simulations SET {', '.join(updates)} WHERE simulation_id = ?"
            cursor.execute(query, params)
        self.invalidate_cache("simulations:")

    async def update_simulation_async(
        self, simulation_id: str, status: str = None, results_summary: Dict = None
    ):
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            updates, params = [], []
            if status:
                updates.append("status = ?")
                params.append(status)
            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))
            if status in ("completed", "failed", "stopped"):
                updates.append("end_time = ?")
                params.append(datetime.now(timezone.utc).isoformat())
                cursor.execute(
                    "SELECT start_time FROM simulations WHERE simulation_id = ?", (simulation_id,)
                )
                row = cursor.fetchone()
                if row and row["start_time"]:
                    start = datetime.fromisoformat(row["start_time"])
                    duration = (datetime.now(timezone.utc) - start).total_seconds()
                    updates.append("duration_seconds = ?")
                    params.append(duration)
            params.append(simulation_id)
            query = f"UPDATE simulations SET {', '.join(updates)} WHERE simulation_id = ?"
            cursor.execute(query, params)
        self.invalidate_cache("simulations:")

    def get_simulations(self, status: str = None, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """SELECT id, simulation_id, simulation_type, status,
                       start_time, end_time, duration_seconds, parameters,
                       results_summary, created_at FROM simulations"""
            params = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def count_simulations(self, simulation_type: str = None) -> int:
        cache_key = self._get_cache_key(f"simulations:count:{simulation_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if simulation_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM simulations WHERE simulation_type = ?", (simulation_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM simulations")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    async def count_simulations_async(self, simulation_type: str = None) -> int:
        cache_key = self._get_cache_key(f"simulations:count:{simulation_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            if simulation_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM simulations WHERE simulation_type = ?", (simulation_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM simulations")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    # ==================== Image Operations ====================

    def add_image(
        self,
        image_path: str,
        image_type: str = None,
        source: str = None,
        width: int = None,
        height: int = None,
        channels: int = None,
        metadata: Dict = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO images
                (image_path, image_type, source, width, height, channels, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_path,
                    image_type,
                    source,
                    width,
                    height,
                    channels,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_images(
        self, image_type: str = None, source: str = None, limit: int = 100
    ) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """SELECT id, image_path, image_type, source, width, height,
                       channels, metadata, created_at FROM images"""
            params = []
            conditions = []
            if image_type:
                conditions.append("image_type = ?")
                params.append(image_type)
            if source:
                conditions.append("source = ?")
                params.append(source)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ==================== Export Operations ====================

    def add_export(
        self,
        export_path: str,
        export_format: str,
        source_type: str = None,
        source_id: int = None,
        file_size_bytes: int = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO exports
                (export_path, export_format, source_type, source_id, file_size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (export_path, export_format, source_type, source_id, file_size_bytes, now),
            )
            return cursor.lastrowid

    # ==================== Generic Query ====================

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    # ==================== Statistics ====================

    def get_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        cache_key = "statistics"
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            cursor.execute("SELECT COUNT(*) FROM scan_results")
            stats["total_scans"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM simulations")
            stats["total_simulations"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM simulations WHERE status = 'running'")
            stats["active_simulations"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM images")
            stats["total_images"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM exports")
            stats["total_exports"] = cursor.fetchone()[0]
            cursor.execute(
                "SELECT scan_type, COUNT(*) as count FROM scan_results GROUP BY scan_type"
            )
            stats["scans_by_type"] = {row["scan_type"]: row["count"] for row in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*) FROM surface_comparisons")
            stats["total_comparisons"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM defect_analysis")
            stats["total_defect_analyses"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM pdf_reports")
            stats["total_pdf_reports"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM batch_jobs")
            stats["total_batch_jobs"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM batch_jobs WHERE status = 'running'")
            stats["active_batch_jobs"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            stats["total_metrics"] = cursor.fetchone()[0]
            if use_cache:
                self._cache_result(cache_key, stats, ttl=10)
            return stats

    def count_analysis_results(self) -> int:
        cache_key = self._get_cache_key("analysis:count")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM defect_analysis")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    def count_comparisons(self) -> int:
        cache_key = self._get_cache_key("comparisons:count")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM surface_comparisons")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    def count_reports(self) -> int:
        cache_key = self._get_cache_key("reports:count")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM reports")
            except Exception as e:
                logger.debug(f"reports table fallback error: {e}")
                cursor.execute("SELECT COUNT(*) FROM exports WHERE export_format = 'PDF'")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    # ==================== Surface Comparisons ====================

    def add_surface_comparison(
        self,
        comparison_id: str,
        image1_path: str,
        image2_path: str,
        similarity_score: float,
        difference_map_path: str = None,
        metrics: Dict = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO surface_comparisons
                (comparison_id, image1_path, image2_path, similarity_score,
                 difference_map_path, metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    comparison_id,
                    image1_path,
                    image2_path,
                    similarity_score,
                    difference_map_path,
                    json.dumps(metrics) if metrics else None,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_surface_comparisons(self, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM surface_comparisons ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ==================== Defect Analysis ====================

    def add_defect_analysis(
        self,
        analysis_id: str,
        image_path: str,
        model_name: str,
        defects_detected: int,
        defects_data: Dict = None,
        confidence_score: float = None,
        processing_time_ms: float = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO defect_analysis
                (analysis_id, image_path, model_name, defects_detected,
                 defects_data, confidence_score, processing_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    image_path,
                    model_name,
                    defects_detected,
                    json.dumps(defects_data) if defects_data else None,
                    confidence_score,
                    processing_time_ms,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_defect_analyses(self, image_path: str = None, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if image_path:
                cursor.execute(
                    "SELECT * FROM defect_analysis WHERE image_path = ? ORDER BY created_at DESC LIMIT ?",
                    (image_path, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM defect_analysis ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ==================== PDF Reports ====================

    def add_pdf_report(
        self,
        report_path: str,
        report_type: str,
        title: str = None,
        source_ids: List[int] = None,
        file_size_bytes: int = None,
        pages_count: int = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO pdf_reports
                (report_path, report_type, title, source_ids, file_size_bytes, pages_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_path,
                    report_type,
                    title,
                    json.dumps(source_ids) if source_ids else None,
                    file_size_bytes,
                    pages_count,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_pdf_reports(self, report_type: str = None, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if report_type:
                cursor.execute(
                    "SELECT * FROM pdf_reports WHERE report_type = ? ORDER BY created_at DESC LIMIT ?",
                    (report_type, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM pdf_reports ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ==================== Batch Jobs ====================

    def add_batch_job(
        self, job_id: str, job_type: str, total_items: int = 0, parameters: Dict = None
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO batch_jobs
                (job_id, job_type, total_items, started_at, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    job_type,
                    total_items,
                    now,
                    json.dumps(parameters) if parameters else None,
                    now,
                ),
            )
            return cursor.lastrowid

    def update_batch_job(
        self,
        job_id: str,
        status: str = None,
        processed_items: int = None,
        failed_items: int = None,
        results_summary: Dict = None,
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            updates, params = [], []
            if status:
                updates.append("status = ?")
                params.append(status)
                if status in ("completed", "failed", "cancelled"):
                    updates.append("completed_at = ?")
                    params.append(datetime.now(timezone.utc).isoformat())
            if processed_items is not None:
                updates.append("processed_items = ?")
                params.append(processed_items)
            if failed_items is not None:
                updates.append("failed_items = ?")
                params.append(failed_items)
            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))
            params.append(job_id)
            query = f"UPDATE batch_jobs SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(query, params)

    def get_batch_jobs(self, status: str = None, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM batch_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ==================== Performance Metrics ====================

    def add_performance_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        unit: str = None,
        metadata: Dict = None,
    ) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO performance_metrics
                (timestamp, metric_type, metric_name, value, unit, metadata)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    metric_type,
                    metric_name,
                    value,
                    unit,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def get_performance_metrics(
        self,
        metric_type: str = None,
        metric_name: str = None,
        start_time: str = None,
        end_time: str = None,
        limit: int = 1000,
    ) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def cleanup_old_metrics(self, days: int = 7) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cursor.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff.isoformat(),)
            )
            return cursor.rowcount

    # ==================== Export ====================

    def export_to_json(self, output_path: str):
        from pathlib import Path

        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_results": [],
                "simulations": [],
                "images": [],
                "exports": [],
            }
            cursor.execute("SELECT * FROM scan_results")
            data["scan_results"] = [self._row_to_dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT * FROM simulations")
            data["simulations"] = [self._row_to_dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT * FROM images")
            data["images"] = [self._row_to_dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT * FROM exports")
            data["exports"] = [self._row_to_dict(row) for row in cursor.fetchall()]
            output = Path(output_path)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return output

    # ==================== User Management ====================

    def get_user(self, username: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def upsert_user(self, username: str, password_hash: str, role: str = "user") -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    role = excluded.role""",
                (username, password_hash, role, datetime.now(timezone.utc).isoformat()),
            )
            return cursor.lastrowid

    def update_last_login(self, username: str) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE username = ?",
                (datetime.now(timezone.utc).isoformat(), username),
            )

    def update_password_hash(self, username: str, new_hash: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username)
            )
            return cursor.rowcount > 0

    # ==================== Cache Methods ====================

    def _get_cached(self, key: str) -> Optional[Any]:
        if not self.enable_cache:
            return None
        if key in self._query_cache:
            value, timestamp = self._query_cache[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self._cache_ttl:
                return value
            del self._query_cache[key]
        return None

    def _cache_result(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self.enable_cache:
            return
        if len(self._query_cache) >= self._cache_max_size:
            oldest = min(self._query_cache.items(), key=lambda x: x[1][1])
            del self._query_cache[oldest[0]]
        self._query_cache[key] = (value, datetime.now(timezone.utc))

    def _get_cache_key(self, query: str, params: tuple = None) -> str:
        return f"{query}:{params}" if params else query

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if not self.enable_cache:
            return None
        if key in self._query_cache:
            value, timestamp = self._query_cache[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self._cache_ttl:
                return value
            else:
                del self._query_cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        if not self.enable_cache:
            return
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]
        self._query_cache[key] = (value, datetime.now(timezone.utc))

    def invalidate_cache(self, pattern: str = None):
        if pattern:
            keys_to_delete = [k for k in self._query_cache.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self._query_cache[key]
        else:
            self._query_cache.clear()

    def set_cache_ttl(self, ttl_seconds: int):
        self._cache_ttl = ttl_seconds

    def get_cache_stats(self) -> Dict:
        now = datetime.now(timezone.utc)
        valid_entries = sum(
            1
            for _, ts in self._query_cache.values()
            if (now - ts).total_seconds() < self._cache_ttl
        )
        return {
            "total_entries": len(self._query_cache),
            "valid_entries": valid_entries,
            "max_size": self._cache_max_size,
            "ttl_seconds": self._cache_ttl,
        }

    @staticmethod
    def cached_query(ttl: int = 300):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key_parts = [func.__name__]
                for arg in args[1:]:
                    cache_key_parts.append(str(arg))
                for k, v in sorted(kwargs.items()):
                    cache_key_parts.append(f"{k}={v}")
                cache_key = hashlib.md5("|".join(cache_key_parts).encode()).hexdigest()
                instance = args[0]
                cached = instance._get_cached(cache_key)
                if cached is not None:
                    return cached
                result = func(*args, **kwargs)
                instance._cache_result(cache_key, result, ttl=ttl)
                return result

            return wrapper

        return decorator
