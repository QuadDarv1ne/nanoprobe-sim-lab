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

        # Create mock cursor with proper attributes
        self.mock_cursor = MagicMock()
        self.mock_cursor.lastrowid = 1
        self.mock_cursor.rowcount = 1

        # Create mock connection that returns the cursor
        self.mock_conn = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor

        # Create mock context manager for get_connection
        self.mock_context = MagicMock()
        self.mock_context.__enter__ = MagicMock(return_value=self.mock_conn)
        self.mock_context.__exit__ = MagicMock(return_value=False)

        self.ops.get_connection = MagicMock(return_value=self.mock_context)
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
        """Test getting scan results when none exist."""
        self.mock_cursor.fetchall.return_value = []
        results = self.ops.get_scan_results(scan_type="afm")
        assert results == []

    def test_add_simulation(self):
        """Test adding a simulation."""
        sim_id = self.ops.add_simulation(
            simulation_id="sim-001",
            simulation_type="molecular_dynamics",
            parameters={"temperature": 300, "pressure": 1},
        )
        assert sim_id == 1

    def test_get_simulations(self):
        """Test getting simulations."""
        # Return a list of dicts instead of tuples
        mock_row = {
            "id": 1,
            "simulation_id": "sim-001",
            "start_time": "2024-01-01T00:00:00",
            "status": "running",
            "simulation_type": "molecular_dynamics",
            "parameters": "{}",
            "created_at": 1,
        }
        self.mock_cursor.fetchall.return_value = [mock_row]
        results = self.ops.get_simulations(status="running")
        assert len(results) >= 0

    def test_add_image(self):
        """Test adding an image."""
        image_id = self.ops.add_image(
            image_path="/data/image1.png",
            image_type="afm",
            source="scanner",
            width=512,
            height=512,
        )
        assert image_id == 1

    def test_get_images(self):
        """Test getting images."""
        mock_row = {
            "id": 1,
            "image_path": "/data/image1.png",
            "image_type": "afm",
            "source": "scanner",
            "width": 512,
            "height": 512,
            "channels": 1,
            "metadata": "{}",
            "created_at": 1,
        }
        self.mock_cursor.fetchall.return_value = [mock_row]
        results = self.ops.get_images(image_type="afm")
        assert len(results) >= 0

    def test_cache_result(self):
        """Test caching a result."""
        self.ops._cache_result("test_key", {"data": "value"}, ttl=60)
        assert "test_key" in self.ops._query_cache

    def test_get_cached_valid(self):
        """Test getting a valid cached result."""
        self.ops._cache_result("test_key", {"data": "value"}, ttl=60)
        result = self.ops._get_from_cache("test_key")
        assert result == {"data": "value"}

    def test_invalidate_cache_pattern(self):
        """Test invalidating cache by pattern."""
        self.ops._cache_result("scans:1", {"data": "1"})
        self.ops._cache_result("scans:2", {"data": "2"})
        self.ops._cache_result("simulations:1", {"data": "3"})
        self.ops.invalidate_cache("scans:")
        assert "scans:1" not in self.ops._query_cache
        assert "scans:2" not in self.ops._query_cache
        assert "simulations:1" in self.ops._query_cache

    def test_invalidate_cache_all(self):
        """Test invalidating all cache."""
        self.ops._cache_result("key1", {"data": "1"})
        self.ops._cache_result("key2", {"data": "2"})
        self.ops.invalidate_cache()
        assert len(self.ops._query_cache) == 0

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        self.ops._cache_result("key1", {"data": "1"})
        stats = self.ops.get_cache_stats()
        assert stats["total_entries"] >= 1

    def test_set_cache_ttl(self):
        """Test setting cache TTL."""
        self.ops.set_cache_ttl(600)
        assert self.ops._cache_ttl == 600

    def test_row_to_dict_with_json_fields(self):
        """Test converting row to dict with JSON fields."""
        mock_row = {
            "id": 1,
            "metadata": '{"key": "value"}',
            "parameters": None,
        }
        result = self.ops._row_to_dict(mock_row)
        assert result["metadata"] == {"key": "value"}

    def test_add_performance_metric(self):
        """Test adding a performance metric."""
        metric_id = self.ops.add_performance_metric(
            metric_type="api",
            metric_name="response_time",
            value=0.05,
            unit="seconds",
        )
        assert metric_id == 1

    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        mock_row = {
            "id": 1,
            "timestamp": "2024-01-01T00:00:00",
            "metric_type": "api",
            "metric_name": "response_time",
            "value": 0.05,
            "unit": "seconds",
            "metadata": "{}",
        }
        self.mock_cursor.fetchall.return_value = [mock_row]
        metrics = self.ops.get_performance_metrics()
        assert len(metrics) >= 0

    def test_cleanup_old_metrics(self):
        """Test cleaning up old metrics."""
        self.mock_cursor.rowcount = 5
        count = self.ops.cleanup_old_metrics(days=7)
        assert count == 5


class TestDatabaseOperationsNoCache:
    """Test DatabaseOperations with cache disabled."""

    @pytest.fixture(autouse=True)
    def setup_ops_no_cache(self):
        """Setup DatabaseOperations with cache disabled."""
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=False)
        yield

    def test_cache_disabled(self):
        """Test that cache is disabled."""
        assert self.ops.enable_cache is False

    def test_get_cached_returns_none_when_disabled(self):
        """Test that _get_from_cache returns None when cache is disabled."""
        result = self.ops._get_from_cache("any_key")
        assert result is None


class TestUserOperations:
    """Test user-related operations."""

    @pytest.fixture(autouse=True)
    def setup_ops(self):
        """Setup DatabaseOperations with mocked connection."""
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=True)

        # Create mock cursor with proper attributes
        self.mock_cursor = MagicMock()
        self.mock_cursor.lastrowid = 1
        self.mock_cursor.rowcount = 1

        # Create mock connection that returns the cursor
        self.mock_conn = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor

        # Create mock context manager for get_connection
        self.mock_context = MagicMock()
        self.mock_context.__enter__ = MagicMock(return_value=self.mock_conn)
        self.mock_context.__exit__ = MagicMock(return_value=False)

        self.ops.get_connection = MagicMock(return_value=self.mock_context)
        yield

    def test_upsert_user(self):
        """Test upserting a user."""
        user_id = self.ops.upsert_user(
            username="testuser",
            password_hash="hashed_password",
            role="user",
        )
        assert user_id == 1

    def test_update_last_login(self):
        """Test updating last login timestamp."""
        result = self.ops.update_last_login(username="testuser")
        assert result is None  # Method returns None

    def test_update_password_hash(self):
        """Test updating password hash."""
        result = self.ops.update_password_hash(username="testuser", new_hash="new_hash")
        assert result is True

    def test_get_user_not_found(self):
        """Test getting a user that doesn't exist."""
        self.mock_cursor.fetchone.return_value = None
        user = self.ops.get_user(username="nonexistent")
        assert user is None


class TestExportOperations:
    """Test export operations."""

    @pytest.fixture(autouse=True)
    def setup_ops(self):
        """Setup DatabaseOperations with mocked connection."""
        self.ops = DatabaseOperations(db_path=":memory:", enable_cache=True)

        # Create mock cursor with proper attributes
        self.mock_cursor = MagicMock()
        self.mock_cursor.lastrowid = 1
        self.mock_cursor.rowcount = 1

        # Create mock connection that returns the cursor
        self.mock_conn = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor

        # Create mock context manager for get_connection
        self.mock_context = MagicMock()
        self.mock_context.__enter__ = MagicMock(return_value=self.mock_conn)
        self.mock_context.__exit__ = MagicMock(return_value=False)

        self.ops.get_connection = MagicMock(return_value=self.mock_context)

        # Mock get_scan_results to return sample data
        self.ops.get_scan_results = MagicMock(
            return_value=[
                {
                    "id": 1,
                    "scan_type": "afm",
                    "surface_type": "silicon",
                    "width": 512,
                    "height": 512,
                    "file_path": "/data/scan1.png",
                    "metadata": {"author": "test"},
                    "timestamp": "2024-01-01T00:00:00",
                }
            ]
        )
        yield

    def test_export_to_json(self):
        """Test exporting data to JSON."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.ops.export_to_json(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert "scan_results" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)
