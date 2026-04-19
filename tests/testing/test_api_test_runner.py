"""
Tests for API Test Runner
"""

import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from utils.testing.api_test_runner import APITestRunner, TestReport, TestResult


class TestTestResult:
    """Tests for TestResult dataclass"""

    def test_success_result(self):
        """Test successful result creation"""
        result = TestResult(
            endpoint="/health",
            method="GET",
            status_code=200,
            expected_status=200,
            response_time=0.123,
            success=True,
        )

        assert result.endpoint == "/health"
        assert result.method == "GET"
        assert result.status_code == 200
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed result creation"""
        result = TestResult(
            endpoint="/api/test",
            method="GET",
            status_code=500,
            expected_status=200,
            response_time=0.456,
            success=False,
            error="Expected 200, got 500",
        )

        assert result.success is False
        assert result.error == "Expected 200, got 500"

    def test_timestamp_format(self):
        """Test timestamp is ISO format"""
        result = TestResult(
            endpoint="/test",
            method="GET",
            status_code=200,
            expected_status=200,
            response_time=0.1,
            success=True,
        )

        # Should be valid ISO format
        datetime.fromisoformat(result.timestamp)


class TestTestReport:
    """Tests for TestReport dataclass"""

    def test_empty_report(self):
        """Test empty report statistics"""
        report = TestReport(base_url="http://test.com")

        assert report.total_tests == 0
        assert report.passed_tests == 0
        assert report.failed_tests == 0
        assert report.success_rate == 0.0

    def test_add_result_success(self):
        """Test adding successful result"""
        report = TestReport(base_url="http://test.com")

        result = TestResult(
            endpoint="/health",
            method="GET",
            status_code=200,
            expected_status=200,
            response_time=0.1,
            success=True,
        )

        report.add_result(result)

        assert report.total_tests == 1
        assert report.passed_tests == 1
        assert report.failed_tests == 0
        assert report.success_rate == 100.0

    def test_add_result_failure(self):
        """Test adding failed result"""
        report = TestReport(base_url="http://test.com")

        result = TestResult(
            endpoint="/api/test",
            method="GET",
            status_code=500,
            expected_status=200,
            response_time=0.2,
            success=False,
            error="Server error",
        )

        report.add_result(result)

        assert report.total_tests == 1
        assert report.passed_tests == 0
        assert report.failed_tests == 1
        assert report.success_rate == 0.0

    def test_mixed_results(self):
        """Test report with mixed results"""
        report = TestReport(base_url="http://test.com")

        report.add_result(
            TestResult(
                endpoint="/health",
                method="GET",
                status_code=200,
                expected_status=200,
                response_time=0.1,
                success=True,
            )
        )

        report.add_result(
            TestResult(
                endpoint="/api/test",
                method="GET",
                status_code=500,
                expected_status=200,
                response_time=0.2,
                success=False,
                error="Error",
            )
        )

        report.add_result(
            TestResult(
                endpoint="/api/another",
                method="POST",
                status_code=201,
                expected_status=201,
                response_time=0.15,
                success=True,
            )
        )

        assert report.total_tests == 3
        assert report.passed_tests == 2
        assert report.failed_tests == 1
        assert report.success_rate == pytest.approx(66.67, rel=0.1)

    def test_generate_summary(self):
        """Test summary generation"""
        report = TestReport(base_url="http://test.com")
        report.add_result(
            TestResult(
                endpoint="/health",
                method="GET",
                status_code=200,
                expected_status=200,
                response_time=0.1,
                success=True,
            )
        )

        summary = report.generate_summary()

        assert "API TEST REPORT" in summary
        assert "http://test.com" in summary
        assert "Total Tests: 1" in summary
        assert "Passed: 1" in summary
        assert "Failed: 0" in summary

    def test_generate_summary_with_failures(self):
        """Test summary generation with failures"""
        report = TestReport(base_url="http://test.com")
        report.add_result(
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

        summary = report.generate_summary()

        assert "❌ FAILED TESTS:" in summary
        assert "Server error" in summary

    def test_save_json(self):
        """Test saving report to JSON"""
        report = TestReport(base_url="http://test.com")
        report.add_result(
            TestResult(
                endpoint="/health",
                method="GET",
                status_code=200,
                expected_status=200,
                response_time=0.1,
                success=True,
            )
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            report.save_json(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["base_url"] == "http://test.com"
            assert data["total_tests"] == 1
            assert data["passed_tests"] == 1
            assert len(data["results"]) == 1
            assert data["results"][0]["endpoint"] == "/health"
        finally:
            import os

            os.unlink(filepath)


class TestAPITestRunner:
    """Tests for APITestRunner class"""

    def test_runner_initialization(self):
        """Test runner initialization"""
        runner = APITestRunner(base_url="http://localhost:8000", timeout=10.0, verbose=True)

        assert runner.base_url == "http://localhost:8000"
        assert runner.timeout == 10.0
        assert runner.verbose is True
        assert runner.report.base_url == "http://localhost:8000"

    def test_runner_default_values(self):
        """Test runner default values"""
        runner = APITestRunner(base_url="http://test.com")

        assert runner.timeout == 30.0
        assert runner.verbose is False

    def test_base_url_normalization(self):
        """Test base URL trailing slash removal"""
        runner = APITestRunner(base_url="http://test.com/")

        assert runner.base_url == "http://test.com"

    def test_run_test_success(self):
        """Test successful request"""
        runner = APITestRunner(base_url="http://test.com")

        # Create a mock client
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
        assert result.response_time > 0

    def test_run_test_failure(self):
        """Test failed request (wrong status code)"""
        runner = APITestRunner(base_url="http://test.com")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b""
        mock_client.get.return_value = mock_response

        runner.client = mock_client

        result = runner._run_test("/api/test", "GET", 200)

        assert result.success is False
        assert result.status_code == 500
        assert result.error is not None

    def test_run_test_timeout(self):
        """Test request timeout"""
        import httpx

        runner = APITestRunner(base_url="http://test.com")

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")
        runner.client = mock_client

        result = runner._run_test("/slow", "GET", 200)

        assert result.success is False
        assert result.error == "Request timeout"

    def test_run_test_post(self):
        """Test POST request"""
        runner = APITestRunner(base_url="http://test.com")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.content = b'{"id": 1}'
        mock_response.json.return_value = {"id": 1}
        mock_client.post.return_value = mock_response

        runner.client = mock_client

        result = runner._run_test("/api/test", "POST", 201, json_data={"key": "value"})

        assert result.success is True
        assert result.status_code == 201

    def test_start_method(self):
        """Test start method execution"""
        runner = APITestRunner(base_url="http://test.com")

        # Mock the test methods and _print_report to avoid actual HTTP calls and sys.exit
        with (
            patch.object(runner, "_test_health_endpoints"),
            patch.object(runner, "_test_auth_endpoints"),
            patch.object(runner, "_test_public_endpoints"),
            patch.object(runner, "_print_report"),
        ):

            runner.start()

        # Verify report was generated
        assert runner.report.total_time >= 0


@pytest.mark.integration
class TestAPITestRunnerIntegration:
    """Integration tests for APITestRunner (requires running API)"""

    @pytest.mark.skip(reason="Requires running API server")
    def test_runner_against_real_api(self):
        """Test runner against real API server"""
        runner = APITestRunner(base_url="http://localhost:8000", verbose=True)
        runner.start()

        assert runner.report.total_tests > 0
        # Note: This test may fail if API is not running
