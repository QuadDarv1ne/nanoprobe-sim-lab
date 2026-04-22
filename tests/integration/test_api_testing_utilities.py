"""
Integration tests for API Test Runner and Profiler utilities

These tests verify that the testing utilities work correctly with a real API server.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from unittest.mock import MagicMock

import httpx
import pytest

from utils.testing.api_profiler import APIProfiler, EndpointStats
from utils.testing.api_test_runner import APITestRunner, TestReport, TestResult

API_BASE_URL = "http://localhost:8000"


@pytest.mark.integration
class TestAPITestRunnerIntegration:
    """Integration tests for APITestRunner against real API"""

    def test_runner_can_connect(self):
        """Test that runner can connect to API"""
        try:
            with httpx.Client(base_url=API_BASE_URL, timeout=5.0) as client:
                response = client.get("/health")
                assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("API server not running")

    def test_runner_tests_health_endpoint(self):
        """Test runner can test health endpoint"""
        try:
            runner = APITestRunner(base_url=API_BASE_URL, timeout=10.0)

            # Mock the client to avoid actual HTTP calls in CI
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}
            mock_client.get.return_value = mock_response

            runner.client = mock_client

            result = runner._run_test("/health", "GET", 200)

            assert result.success is True
            assert result.status_code == 200
        except httpx.ConnectError:
            pytest.skip("API server not running")

    def test_runner_generates_report(self):
        """Test runner generates proper report"""
        runner = APITestRunner(base_url=API_BASE_URL, timeout=10.0)

        # Add some mock results
        runner.report.add_result(
            TestResult(
                endpoint="/health",
                method="GET",
                status_code=200,
                expected_status=200,
                response_time=0.1,
                success=True,
            )
        )

        runner.report.add_result(
            TestResult(
                endpoint="/api/test",
                method="GET",
                status_code=500,
                expected_status=200,
                response_time=0.2,
                success=False,
                error="Server error",
            )
        )

        assert runner.report.total_tests == 2
        assert runner.report.passed_tests == 1
        assert runner.report.failed_tests == 1
        assert runner.report.success_rate == 50.0


@pytest.mark.integration
class TestAPIProfilerIntegration:
    """Integration tests for APIProfiler against real API"""

    def test_profiler_can_connect(self):
        """Test that profiler can connect to API"""
        try:
            with httpx.Client(base_url=API_BASE_URL, timeout=5.0) as client:
                response = client.get("/health")
                assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("API server not running")

    def test_profiler_collects_stats(self):
        """Test profiler collects statistics"""
        try:
            profiler = APIProfiler(base_url=API_BASE_URL, default_iterations=3)

            # Mock the client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response

            profiler.client = mock_client

            stats = profiler._profile_endpoint("/health", "GET", iterations=3)

            assert stats.iterations == 3
            assert len(stats.response_times) == 3
            assert all(t > 0 for t in stats.response_times)
        except httpx.ConnectError:
            pytest.skip("API server not running")

    def test_profiler_calculates_percentiles(self):
        """Test profiler calculates percentiles correctly"""
        stats = EndpointStats(endpoint="/test", method="GET", iterations=100)

        # Add deterministic response times
        for i in range(100):
            stats.add_result((i + 1) * 0.01, 200)

        assert stats.min_time == 0.01
        assert stats.max_time == 1.0
        assert stats.avg_time == pytest.approx(0.505, rel=0.01)
        # P95 should be around 0.95-0.96 (95th percentile)
        assert 0.94 <= stats.p95_time <= 0.97

    def test_profiler_generates_report(self):
        """Test profiler generates proper report"""
        profiler = APIProfiler(base_url=API_BASE_URL, default_iterations=5)

        # Add mock stats
        stats = EndpointStats(endpoint="/health", method="GET", iterations=5)
        for i in range(5):
            stats.add_result(0.1 * (i + 1), 200)

        profiler.report.add_stats(stats)

        assert profiler.report.total_requests == 5
        assert "GET /health" in profiler.report.endpoint_stats

        # Test report generation
        summary = profiler.report.generate_summary()
        assert "API PERFORMANCE PROFILER REPORT" in summary
        assert "Total Requests: 5" in summary


@pytest.mark.integration
class TestUtilitiesWithRealAPI:
    """Test utilities with real API server (optional)"""

    @pytest.mark.skip(reason="Requires running API server")
    def test_full_test_runner(self):
        """Run full test suite against real API"""
        runner = APITestRunner(base_url=API_BASE_URL, timeout=30.0)
        runner.start()

        # Should have tested multiple endpoints
        assert runner.report.total_tests > 0
        # Success rate should be high for a working API
        assert runner.report.success_rate >= 80.0

    @pytest.mark.skip(reason="Requires running API server")
    def test_full_profiler(self):
        """Run full profiler against real API"""
        profiler = APIProfiler(
            base_url=API_BASE_URL,
            default_iterations=10,
            timeout=30.0,
        )
        profiler.start()

        # Should have profiled multiple endpoints
        assert profiler.report.total_requests > 0
        assert len(profiler.report.endpoint_stats) > 0
