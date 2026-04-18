#!/usr/bin/env python3
"""
Unit tests for DatabaseOperations from utils.db.operations
"""
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import the module under test
from utils.db.operations import DatabaseOperations


class TestDatabaseOperations:
    """Test DatabaseOperations CRUD methods."""

    @pytest.fixture(autouse=True)
    def setup_ops(self):
        """Setup DatabaseOperations with mocked connection."""
        # Create instance with cache enabled
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=True)

        # Mock the get_connection method
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.__enter__ = MagicMock(return_value=self.mock_cursor)
        self.mock_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_cursor.lastrowid = 1
        self.mock_cursor.rowcount = 1

        self.ops.get_connection = MagicMock(return_value=self.mock_conn)

        yield

    def test_add_scan_result(self):
        """Test adding a scan result."""
        scan_id = self.ops.add_scan_result(
            scan_type="afm",
            surface_type="silicon",
            width=512,
            height=512,
            file_path="/data/scan1.png",
            metadata={"author": "test"},
        )

        assert scan_id == 1
        self.mock_conn.cursor().execute.assert_called()

    def test_add_scan_result_batch(self):
        """Test adding multiple scan results."""
        scan_results = [
            {
                "scan_type": "afm",
                "surface_type": "silicon",
                "width": 512,
                "height": 512,
                "file_path": "/data/scan1.png",
                "metadata": {"author": "test1"},
            },
            {
                "scan_type": "stm",
                "surface_type": "gold",
                "width": 256,
                "height": 256,
                "file_path": "/data/scan2.png",
                "metadata": {"author": "test2"},
            },
        ]

        count = self.ops.add_scan_result_batch(scan_results)

        assert count == 2
        self.mock_cursor.executemany.assert_called()

    def test_get_scan_results_empty(self):
        """Test getting scan results when database is empty."""
        self.mock_cursor.fetchall.return_value = []

        results = self.ops.get_scan_results(scan_type="afm", limit=10)

        assert results == []

    def test_add_simulation_result(self):
        """Test adding a simulation result."""
        sim_id = self.ops.add_simulation_result(
            simulation_type="spm",
            parameters={"force": 1e-9, "speed": 1e-3},
            results_summary={"rms": 0.5},
            metrics={"duration": 10.5},
        )

        assert sim_id == 1

    def test_get_simulation_results(self):
        """Test getting simulation results."""
        self.mock_cursor.fetchall.return_value = []

        results = self.ops.get_simulation_results(simulation_type="spm", limit=10)

        assert isinstance(results, list)

    def test_add_image_metadata(self):
        """Test adding image metadata."""
        image_id = self.ops.add_image_metadata(
            image_path="/data/image1.png",
            image_type="afm",
            width=512,
            height=512,
            metadata={"acquisition": "test"},
        )

        assert image_id == 1

    def test_get_image_metadata(self):
        """Test getting image metadata."""
        self.mock_cursor.fetchall.return_value = []

        results = self.ops.get_image_metadata(image_type="afm", limit=10)

        assert isinstance(results, list)

    def test_cache_result(self):
        """Test caching a result."""
        key = "test:key"
        value = {"data": "test"}

        self.ops._cache_result(key, value)

        assert self.ops._query_cache[key][0] == value

    def test_get_cached_valid(self):
        """Test getting a valid cached result."""
        key = "test:cached"
        value = {"data": "cached"}

        self.ops._query_cache[key] = (value, datetime.now(timezone.utc))

        cached = self.ops._get_cached(key)

        assert cached == value

    def test_get_cached_expired(self):
        """Test getting an expired cached result."""
        key = "test:expired"
        value = {"data": "expired"}

        # Set old timestamp
        self.ops._query_cache[key] = (
            value,
            datetime.now(timezone.utc) - timedelta(seconds=301),
        )

        cached = self.ops._get_cached(key)

        assert cached is None

    def test_invalidate_cache_pattern(self):
        """Test invalidating cache with pattern."""
        from datetime import datetime, timezone

        self.ops._query_cache["scans:afm:10:0"] = (
            {"data": "test1"},
            datetime.now(timezone.utc),
        )
        self.ops._query_cache["scans:stm:10:0"] = (
            {"data": "test2"},
            datetime.now(timezone.utc),
        )
        self.ops._query_cache["simulations:spm:10:0"] = (
            {"data": "test3"},
            datetime.now(timezone.utc),
        )

        self.ops.invalidate_cache("scans:")

        assert "scans:afm:10:0" not in self.ops._query_cache
        assert "scans:stm:10:0" not in self.ops._query_cache
        assert "simulations:spm:10:0" in self.ops._query_cache

    def test_invalidate_cache_all(self):
        """Test invalidating all cache."""
        from datetime import datetime, timezone

        self.ops._query_cache["key1"] = (
            {"data": "test1"},
            datetime.now(timezone.utc),
        )
        self.ops._query_cache["key2"] = (
            {"data": "test2"},
            datetime.now(timezone.utc),
        )

        self.ops.invalidate_cache()

        assert len(self.ops._query_cache) == 0

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        from datetime import datetime, timezone

        self.ops._query_cache["key1"] = (
            {"data": "test1"},
            datetime.now(timezone.utc),
        )

        stats = self.ops.get_cache_stats()

        assert "total_entries" in stats
        assert "valid_entries" in stats
        assert stats["total_entries"] >= 1

    def test_set_cache_ttl(self):
        """Test setting cache TTL."""
        self.ops.set_cache_ttl(600)

        assert self.ops._cache_ttl == 600

    def test_row_to_dict_with_json_fields(self):
        """Test converting row to dict with JSON fields."""
        mock_row = MagicMock()
        mock_row.__iter__ = lambda self: iter(["id", "metadata", "parameters", "results_summary"])
        mock_row.__getitem__ = lambda self, key: {
            "id": 1,
            "metadata": '{"key": "value"}',
            "parameters": '{"force": 1e-9}',
            "results_summary": None,
        }.get(key)
        mock_row.keys = lambda: ["id", "metadata", "parameters", "results_summary"]

        result = self.ops._row_to_dict(mock_row)

        assert result["id"] == 1
        assert isinstance(result["metadata"], dict)
        assert result["metadata"]["key"] == "value"

    def test_add_performance_metric(self):
        """Test adding a performance metric."""
        self.ops.add_performance_metric(
            component="api",
            metric_name="response_time",
            value=0.15,
            unit="seconds",
        )

        self.mock_conn.cursor().execute.assert_called()

    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        self.mock_cursor.fetchall.return_value = []

        metrics = self.ops.get_performance_metrics(
            component="api", metric_name="response_time", limit=10
        )

        assert isinstance(metrics, list)

    def test_cleanup_old_metrics(self):
        """Test cleaning up old metrics."""
        self.mock_cursor.rowcount = 5

        count = self.ops.cleanup_old_metrics(days=7)

        assert count == 5


class TestDatabaseOperationsNoCache:
    """Test DatabaseOperations with caching disabled."""

    @pytest.fixture(autouse=True)
    def setup_ops_no_cache(self):
        """Setup DatabaseOperations with caching disabled."""
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=False)

        yield

    def test_cache_disabled(self):
        """Test that caching is disabled."""
        key = "test:key"
        value = {"data": "test"}

        self.ops._cache_result(key, value)

        assert key not in self.ops._query_cache

    def test_get_cached_returns_none_when_disabled(self):
        """Test that _get_cached returns None when caching is disabled."""
        result = self.ops._get_cached("any:key")

        assert result is None


class TestUserOperations:
    """Test user-related operations."""

    @pytest.fixture(autouse=True)
    def setup_ops(self):
        """Setup DatabaseOperations for user tests."""
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=False)

        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.__enter__ = MagicMock(return_value=self.mock_cursor)
        self.mock_cursor.__exit__ = MagicMock(return_value=False)
        self.mock_cursor.lastrowid = 1
        self.mock_cursor.rowcount = 1

        self.ops.get_connection = MagicMock(return_value=self.mock_conn)

        yield

    def test_upsert_user(self):
        """Test upserting a user."""
        user_id = self.ops.upsert_user(
            username="testuser",
            password_hash="hashed_password",
            role="admin",
        )

        assert user_id == 1

    def test_update_last_login(self):
        """Test updating last login timestamp."""
        self.ops.update_last_login(username="testuser")

        self.mock_conn.cursor().execute.assert_called()

    def test_update_password_hash(self):
        """Test updating password hash."""
        self.mock_cursor.rowcount = 1
        result = self.ops.update_password_hash(username="testuser", new_hash="new_hashed_password")
        assert result is True

    def test_get_user_not_found(self):
        """Test getting a non-existent user."""
        self.mock_cursor.fetchone.return_value = None

        user = self.ops.get_user(username="nonexistent")

        assert user is None


class TestExportOperations:
    """Test export-related operations."""

    @pytest.fixture(autouse=True)
    def setup_ops(self):
        """Setup DatabaseOperations for export tests."""
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=False)

        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.__enter__ = MagicMock(return_value=self.mock_cursor)
        self.mock_cursor.__exit__ = MagicMock(return_value=False)

        self.ops.get_connection = MagicMock(return_value=self.mock_conn)

        yield

    def test_export_to_json(self, tmp_path):
        """Test exporting data to JSON."""
        # Setup mock data
        mock_row = MagicMock()
        mock_row.__iter__ = lambda self: iter(["id", "timestamp"])
        mock_row.__getitem__ = lambda self, key: {"id": 1, "timestamp": "2026-01-01"}.get(key)
        mock_row.keys = lambda: ["id", "timestamp"]

        self.mock_cursor.fetchall.return_value = [mock_row]

        output_path = tmp_path / "export.json"
        result = self.ops.export_to_json(str(output_path))

        assert output_path.exists()
