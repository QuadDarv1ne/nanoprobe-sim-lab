#!/usr/bin/env python
"""
Unit тесты для Data Integrity Checker

Тестирование проверки целостности данных
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from utils.data.data_integrity import DataIntegrityChecker


class TestDataIntegrityInit:
    """Тесты инициализации DataIntegrityChecker"""

    def test_init(self):
        """Тест инициализации checker"""
        checker = DataIntegrityChecker()

        assert checker.check_results == {}
        print("  [PASS] Init")


class TestDataIntegrityChecksum:
    """Тесты вычисления контрольных сумм"""

    def test_calculate_checksum_consistent(self):
        """Тест консистентности контрольной суммы"""
        checker = DataIntegrityChecker()
        data = b"test data"

        checksum1 = checker.calculate_checksum(data)
        checksum2 = checker.calculate_checksum(data)

        assert checksum1 == checksum2
        assert isinstance(checksum1, str)
        assert len(checksum1) == 64  # SHA256 hex
        print("  [PASS] Calculate checksum consistent")

    def test_calculate_checksum_different_data(self):
        """Тест различных контрольных сумм для разных данных"""
        checker = DataIntegrityChecker()

        checksum1 = checker.calculate_checksum(b"data1")
        checksum2 = checker.calculate_checksum(b"data2")

        assert checksum1 != checksum2
        print("  [PASS] Calculate checksum different data")

    def test_calculate_file_checksum(self):
        """Тест контрольной суммы файла"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test file content")
            temp_path = f.name

        try:
            checksum = checker.calculate_file_checksum(temp_path)

            assert checksum is not None
            assert isinstance(checksum, str)
            assert len(checksum) == 64
        finally:
            os.unlink(temp_path)

        print("  [PASS] Calculate file checksum")

    def test_calculate_file_checksum_nonexistent(self):
        """Тест контрольной суммы несуществующего файла"""
        checker = DataIntegrityChecker()

        checksum = checker.calculate_file_checksum("/nonexistent/file.txt")

        assert checksum is None
        print("  [PASS] Calculate file checksum nonexistent")

    def test_verify_file_integrity_valid(self):
        """Тест проверки целостности файла (валидный)"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            expected = checker.calculate_file_checksum(temp_path)
            is_valid = checker.verify_file_integrity(temp_path, expected)

            assert is_valid is True
        finally:
            os.unlink(temp_path)

        print("  [PASS] Verify file integrity valid")

    def test_verify_file_integrity_invalid(self):
        """Тест проверки целостности файла (невалидный)"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            is_valid = checker.verify_file_integrity(temp_path, "wrong_checksum")

            assert is_valid is False
        finally:
            os.unlink(temp_path)

        print("  [PASS] Verify file integrity invalid")


class TestDataIntegrityNumpy:
    """Тесты проверки numpy массивов"""

    def test_check_numpy_array_integrity_valid(self):
        """Тест проверки валидного numpy массива"""
        checker = DataIntegrityChecker()
        array = np.array([1, 2, 3, 4, 5])

        results = checker.check_numpy_array_integrity(array)

        assert results["shape"] == (5,)
        assert results["dtype"] == "int32" or results["dtype"] == "int64"
        assert results["size"] == 5
        assert results["ndim"] == 1
        assert results["has_nan"] is False
        assert results["has_inf"] is False
        assert results["valid"] is True
        assert results["min_value"] == 1
        assert results["max_value"] == 5
        print("  [PASS] Check numpy array integrity valid")

    def test_check_numpy_array_integrity_with_nan(self):
        """Тест проверки numpy массива с NaN"""
        checker = DataIntegrityChecker()
        array = np.array([1, 2, np.nan, 4, 5])

        results = checker.check_numpy_array_integrity(array)

        assert results["has_nan"] is True
        assert results["valid"] is False
        print("  [PASS] Check numpy array integrity with NaN")

    def test_check_numpy_array_integrity_with_inf(self):
        """Тест проверки numpy массива с Inf"""
        checker = DataIntegrityChecker()
        array = np.array([1, 2, np.inf, 4, 5])

        results = checker.check_numpy_array_integrity(array)

        assert results["has_inf"] is True
        assert results["valid"] is False
        print("  [PASS] Check numpy array integrity with Inf")

    def test_check_numpy_array_integrity_empty(self):
        """Тест проверки пустого numpy массива"""
        checker = DataIntegrityChecker()
        array = np.array([])

        results = checker.check_numpy_array_integrity(array)

        assert results["size"] == 0
        assert results["valid"] is False  # Пустой массив невалиден
        print("  [PASS] Check numpy array integrity empty")

    def test_check_numpy_array_integrity_2d(self):
        """Тест проверки 2D numpy массива"""
        checker = DataIntegrityChecker()
        array = np.array([[1, 2], [3, 4], [5, 6]])

        results = checker.check_numpy_array_integrity(array)

        assert results["shape"] == (3, 2)
        assert results["ndim"] == 2
        assert results["size"] == 6
        print("  [PASS] Check numpy array integrity 2D")


class TestDataIntegrityCSV:
    """Тесты проверки CSV файлов"""

    def test_check_csv_integrity_valid(self):
        """Тест проверки валидного CSV файла"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            temp_path = f.name

        try:
            results = checker.check_csv_integrity(temp_path)

            assert results["valid"] is True
            assert results["rows"] == 2
            assert results["columns"] == 3
            assert "checksum" in results
        finally:
            os.unlink(temp_path)

        print("  [PASS] Check CSV integrity valid")

    def test_check_csv_integrity_nonexistent(self):
        """Тест проверки несуществующего CSV файла"""
        checker = DataIntegrityChecker()

        results = checker.check_csv_integrity("/nonexistent/file.csv")

        assert results["valid"] is False
        assert results["error"] is not None
        print("  [PASS] Check CSV integrity nonexistent")

    def test_check_csv_integrity_empty(self):
        """Тест проверки пустого CSV файла"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            results = checker.check_csv_integrity(temp_path)

            assert results["valid"] is False
        finally:
            os.unlink(temp_path)

        print("  [PASS] Check CSV integrity empty")


class TestDataIntegrityJSON:
    """Тесты проверки JSON файлов"""

    def test_check_json_integrity_valid(self):
        """Тест проверки валидного JSON файла"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value", "number": 42}, f)
            temp_path = f.name

        try:
            results = checker.check_json_integrity(temp_path)

            assert results["valid"] is True
            assert "checksum" in results
            assert results["keys"] is not None
        finally:
            os.unlink(temp_path)

        print("  [PASS] Check JSON integrity valid")

    def test_check_json_integrity_invalid(self):
        """Тест проверки невалидного JSON файла"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json}")
            temp_path = f.name

        try:
            results = checker.check_json_integrity(temp_path)

            assert results["valid"] is False
            assert results["error"] is not None
        finally:
            os.unlink(temp_path)

        print("  [PASS] Check JSON integrity invalid")


class TestDataIntegrityDatabase:
    """Тесты проверки базы данных"""

    def test_check_database_integrity_valid(self):
        """Тест проверки валидной SQLite БД"""
        checker = DataIntegrityChecker()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        # Создаём тестовую БД
        import sqlite3

        conn = sqlite3.connect(temp_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'test')")
        conn.commit()
        conn.close()

        try:
            results = checker.check_database_integrity(temp_path)

            assert results["valid"] is True
            assert "tables" in results
            assert "checksum" in results
        finally:
            os.unlink(temp_path)

        print("  [PASS] Check database integrity valid")

    def test_check_database_integrity_nonexistent(self):
        """Тест проверки несуществующей БД"""
        checker = DataIntegrityChecker()

        results = checker.check_database_integrity("/nonexistent/db.db")

        assert results["valid"] is False
        assert results["error"] is not None
        print("  [PASS] Check database integrity nonexistent")


class TestDataIntegrityReport:
    """Тесты отчётов о целостности"""

    def test_generate_integrity_report(self):
        """Тест генерации отчёта о целостности"""
        checker = DataIntegrityChecker()

        # Добавляем результаты проверок
        checker.check_results = {
            "file1.txt": {"valid": True, "checksum": "abc123"},
            "file2.txt": {"valid": False, "error": "Checksum mismatch"},
        }

        report = checker.generate_integrity_report()

        assert isinstance(report, dict)
        assert "timestamp" in report
        assert "total_checks" in report
        assert "valid_count" in report
        assert "invalid_count" in report
        assert report["total_checks"] == 2
        assert report["valid_count"] == 1
        assert report["invalid_count"] == 1
        print("  [PASS] Generate integrity report")

    def test_generate_integrity_report_empty(self):
        """Тест генерации пустого отчёта"""
        checker = DataIntegrityChecker()

        report = checker.generate_integrity_report()

        assert report["total_checks"] == 0
        assert report["valid_count"] == 0
        assert report["invalid_count"] == 0
        print("  [PASS] Generate integrity report empty")


# Import json для тестов
import json


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Data Integrity Checker Unit Tests")
    print("=" * 60)

    test_classes = [
        TestDataIntegrityInit,
        TestDataIntegrityChecksum,
        TestDataIntegrityNumpy,
        TestDataIntegrityCSV,
        TestDataIntegrityJSON,
        TestDataIntegrityDatabase,
        TestDataIntegrityReport,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: {e}")
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("  ✅ All tests passed!")
    else:
        print(f"  ❌ {total_tests - passed_tests} tests failed")
    print("=" * 60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
