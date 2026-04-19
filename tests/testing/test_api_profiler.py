"""
Tests for API Performance Profiler
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from utils.testing.api_profiler import APIProfiler, EndpointStats, ProfilerReport


class TestEndpointStats:
    """Tests for EndpointStats dataclass"""

    def test_empty_stats(self):
        """Test empty endpoint statistics"""
        stats = EndpointStats(endpoint="/health", method="GET", iterations=0)

        assert stats.min_time == 0.0
        assert stats.max_time == 0.0
        assert stats.avg_time == 0.0
        assert stats.median_time == 0.0
        assert stats.p95_time == 0.0
        assert stats.success_rate == 0.0

    def test_add_result(self):
        """Test adding results"""
        stats = EndpointStats(endpoint="/health", method="GET", iterations=3)

        stats.add_result(0.1, 200)
        stats.add_result(0.2, 200)
        stats.add_result(0.15, 200)

        assert len(stats.response_times) == 3
        assert stats.status_codes[200] == 3

    def test_min_max_time(self):
        """Test min and max time calculations"""
        stats = EndpointStats(endpoint="/health", method="GET", iterations=3)

        stats.add_result(0.1, 200)
        stats.add_result(0.3, 200)
        stats.add_result(0.2, 200)

        assert stats.min_time == 0.1
        assert stats.max_time == 0.3

    def test_avg_time(self):
        """Test average time calculation"""
        stats = EndpointStats(endpoint="/health", method="GET", iterations=3)

        stats.add_result(0.1, 200)
        stats.add_result(0.2, 200)
        stats.add_result(0.3, 200)

        assert stats.avg_time == pytest.approx(0.2, rel=0.01)

    def test_median_time(self):
        """Test median time calculation"""
        stats = EndpointStats(endpoint="/health", method="GET", iterations=3)

        stats.add_result(0.1, 200)
        stats.add_result(0.3, 200)
        stats.add_result(0.2, 200)

        assert stats.median_time == pytest.approx(0.2, rel=0.01)

    def test_p95_time(self):
        """Test P95 time calculation"""
        stats = EndpointStats(endpoint="/health", method="GET", iterations=20)

        # Add 20 results with times 0.01 to 0.20
        for i in range(20):
            stats.add_result((i + 1) * 0.01, 200)

        # P95 should be around 0.19 (95th percentile)
        assert stats.p95_time == pytest.approx(0.19, rel=0.1)

    def test_success_rate(self):
        """Test success rate calculation"""
        stats = EndpointStats(endpoint="/api/test", method="GET", iterations=10)

        # 8 successful (2xx), 2 failed (5xx)
        for _ in range(8):
            stats.add_result(0.1, 200)
        for _ in range(2):
            stats.add_result(0.1, 500)

        # success_rate is based on response_times length, not iterations
        assert len(stats.response_times) == 10
        assert stats.success_rate == pytest.approx(80.0, rel=0.1)

    def test_error_tracking(self):
        """Test error tracking"""
        stats = EndpointStats(endpoint="/api/test", method="GET", iterations=3)

        stats.add_result(0.1, 200)
        stats.add_result(0, 0, "Timeout")
        stats.add_result(0.15, 200)

        assert len(stats.errors) == 1
        assert stats.errors[0] == "Timeout"


class TestProfilerReport:
    """Tests for ProfilerReport dataclass"""

    def test_empty_report(self):
        """Test empty report"""
        report = ProfilerReport(base_url="http://test.com")

        assert report.total_requests == 0
        assert len(report.endpoint_stats) == 0

    def test_add_stats(self):
        """Test adding endpoint statistics"""
        report = ProfilerReport(base_url="http://test.com")

        stats = EndpointStats(endpoint="/health", method="GET", iterations=5)
        stats.add_result(0.1, 200)
        stats.add_result(0.15, 200)

        report.add_stats(stats)

        assert report.total_requests == 5
        assert "GET /health" in report.endpoint_stats

    def test_generate_summary(self):
        """Test summary generation"""
        report = ProfilerReport(base_url="http://test.com")

        stats = EndpointStats(endpoint="/health", method="GET", iterations=3)
        stats.add_result(0.1, 200)
        stats.add_result(0.2, 200)
        stats.add_result(0.15, 200)

        report.add_stats(stats)

        summary = report.generate_summary()

        assert "API PERFORMANCE PROFILER REPORT" in summary
        assert "http://test.com" in summary
        assert "Total Requests: 3" in summary
        assert "GET /health" in summary

    def test_save_json(self):
        """Test saving report to JSON"""
        report = ProfilerReport(base_url="http://test.com")

        stats = EndpointStats(endpoint="/health", method="GET", iterations=2)
        stats.add_result(0.1, 200)
        stats.add_result(0.2, 200)

        report.add_stats(stats)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            report.save_json(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["base_url"] == "http://test.com"
            assert data["total_requests"] == 2
            assert "GET /health" in data["endpoints"]
            assert data["endpoints"]["GET /health"]["avg_time_ms"] == pytest.approx(150.0, rel=0.1)
        finally:
            import os

            os.unlink(filepath)


class TestAPIProfiler:
    """Tests for APIProfiler class"""

    def test_profiler_initialization(self):
        """Test profiler initialization"""
        profiler = APIProfiler(
            base_url="http://localhost:8000",
            default_iterations=20,
            timeout=15.0,
            verbose=True,
        )

        assert profiler.base_url == "http://localhost:8000"
        assert profiler.default_iterations == 20
        assert profiler.timeout == 15.0
        assert profiler.verbose is True

    def test_profiler_default_values(self):
        """Test profiler default values"""
        profiler = APIProfiler(base_url="http://test.com")

        assert profiler.default_iterations == 10
        assert profiler.timeout == 30.0
        assert profiler.verbose is False

    def test_base_url_normalization(self):
        """Test base URL trailing slash removal"""
        profiler = APIProfiler(base_url="http://test.com/")

        assert profiler.base_url == "http://test.com"

    def test_profile_endpoint(self):
        """Test endpoint profiling"""
        profiler = APIProfiler(base_url="http://test.com", default_iterations=3)

        # Create a mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        profiler.client = mock_client

        stats = profiler._profile_endpoint("/health", "GET", iterations=3)

        assert stats.iterations == 3
        assert len(stats.response_times) == 3
        assert stats.status_codes[200] == 3

    def test_profile_endpoint_timeout(self):
        """Test endpoint profiling with timeout"""
        import httpx

        profiler = APIProfiler(base_url="http://test.com", default_iterations=2)

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")
        profiler.client = mock_client

        stats = profiler._profile_endpoint("/slow", "GET", iterations=2)

        assert stats.iterations == 2
        assert len(stats.errors) == 2
        assert all(e == "Timeout" for e in stats.errors)

    def test_profile_endpoint_post(self):
        """Test POST endpoint profiling"""
        profiler = APIProfiler(base_url="http://test.com", default_iterations=2)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_client.post.return_value = mock_response
        profiler.client = mock_client

        stats = profiler._profile_endpoint(
            "/api/test", "POST", iterations=2, json_data={"key": "value"}
        )

        assert stats.iterations == 2
        assert stats.status_codes[201] == 2

    def test_start_method(self):
        """Test start method execution"""
        profiler = APIProfiler(base_url="http://test.com", default_iterations=1)

        # Mock the profiling methods to avoid actual HTTP calls
        with (
            patch.object(profiler, "_profile_health_endpoints"),
            patch.object(profiler, "_profile_public_endpoints"),
            patch.object(profiler, "_profile_auth_endpoints"),
        ):

            profiler.start()

        # Verify report was generated
        assert profiler.report.total_time >= 0


@pytest.mark.integration
class TestAPIProfilerIntegration:
    """Integration tests for APIProfiler (requires running API)"""

    @pytest.mark.skip(reason="Requires running API server")
    def test_profiler_against_real_api(self):
        """Test profiler against real API server"""
        profiler = APIProfiler(
            base_url="http://localhost:8000",
            default_iterations=5,
            verbose=True,
        )
        profiler.start()

        assert profiler.report.total_requests > 0
        assert len(profiler.report.endpoint_stats) > 0

    @pytest.mark.skip(reason="Requires running API server")
    def test_profiler_save_report(self):
        """Test profiler with report saving"""
        profiler = APIProfiler(
            base_url="http://localhost:8000",
            default_iterations=3,
        )
        profiler.start()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            profiler.report.save_json(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "endpoints" in data
            assert len(data["endpoints"]) > 0
        finally:
            import os

            os.unlink(filepath)
