"""Unit-тесты для модуля продвинутого анализа логов."""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import mock_open, patch

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logger_analyzer import AdvancedLoggerAnalyzer, LogAnalysisResult, LogEntry


class TestLogEntry(unittest.TestCase):
    """Тесты для класса LogEntry"""

    def test_init(self):
        """Тест инициализации LogEntry"""
        timestamp = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            component="test_component",
            message="Test message",
            module="test_module",
            function="test_function",
            thread_id="12345",
        )
        self.assertEqual(entry.timestamp, timestamp)
        self.assertEqual(entry.level, "INFO")
        self.assertEqual(entry.component, "test_component")
        self.assertEqual(entry.message, "Test message")
        self.assertEqual(entry.module, "test_module")
        self.assertEqual(entry.function, "test_function")
        self.assertEqual(entry.thread_id, "12345")


class TestLogAnalysisResult(unittest.TestCase):
    """Тесты для класса LogAnalysisResult"""

    def test_init(self):
        """Тест инициализации LogAnalysisResult"""
        now = datetime.now(timezone.utc)
        result = LogAnalysisResult(
            total_entries=100,
            error_count=10,
            warning_count=20,
            info_count=60,
            debug_count=10,
            unique_components=["comp1", "comp2"],
            time_range=(now - timedelta(hours=1), now),
            analysis_timestamp=now,
        )
        self.assertEqual(result.total_entries, 100)
        self.assertEqual(result.error_count, 10)
        self.assertEqual(result.warning_count, 20)
        self.assertEqual(result.info_count, 60)
        self.assertEqual(result.debug_count, 10)
        self.assertEqual(result.unique_components, ["comp1", "comp2"])
        self.assertEqual(result.time_range[0], now - timedelta(hours=1))
        self.assertEqual(result.time_range[1], now)
        self.assertEqual(result.analysis_timestamp, now)


class TestAdvancedLoggerAnalyzer(unittest.TestCase):
    """Тесты для класса AdvancedLoggerAnalyzer"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir()
        self.analyzer = AdvancedLoggerAnalyzer(log_directory=str(self.log_dir))

    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Тест инициализации анализатора логов"""
        self.assertEqual(self.analyzer.log_directory, self.log_dir)
        self.assertEqual(self.analyzer.log_entries, [])
        self.assertEqual(self.analyzer.analysis_results, {})
        self.assertIsInstance(self.analyzer.patterns, dict)
        self.assertIn("timestamp", self.analyzer.patterns)
        self.assertIsInstance(self.analyzer.level_colors, dict)
        self.assertFalse(self.analyzer.real_time_monitoring)
        self.assertIsNone(self.analyzer.monitoring_thread)

    def test_parse_log_line_standard_format(self):
        """Тест парсинга строки лога в стандартном формате"""
        line = "[2023-12-01 10:30:45] - INFO - test_component - Test message"
        entry = self.analyzer.parse_log_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.level, "INFO")
        self.assertEqual(entry.component, "test_component")
        self.assertEqual(entry.message, "Test message")
        self.assertEqual(entry.timestamp, datetime(2023, 12, 1, 10, 30, 45))

    def test_parse_log_line_with_milliseconds(self):
        """Тест парсинга строки лога с миллисекундами"""
        line = "[2023-12-01 10:30:45.123] - ERROR - test_component - Test error"
        entry = self.analyzer.parse_log_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.level, "ERROR")
        self.assertEqual(entry.component, "test_component")
        self.assertEqual(entry.message, "Test error")
        # Миллисекунды отбрасываются в текущей реализации
        self.assertEqual(entry.timestamp, datetime(2023, 12, 1, 10, 30, 45))

    def test_parse_log_line_invalid_format(self):
        """Тест парсинга невалидной строки лога"""
        line = "Это не строка лога"
        entry = self.analyzer.parse_log_line(line)
        self.assertIsNone(entry)

    def test_parse_log_line_empty_line(self):
        """Тест парсинга пустой строки"""
        line = ""
        entry = self.analyzer.parse_log_line(line)
        self.assertIsNone(entry)

    def test_parse_log_file(self):
        """Тест парсинга файла лога"""
        log_content = """[2023-12-01 10:30:45] - INFO - component1 - Message 1
[2023-12-01 10:30:46] - WARNING - component2 - Message 2
[2023-12-01 10:30:47] - ERROR - component1 - Message 3
Invalid line
[2023-12-01 10:30:48] - DEBUG - component3 - Message 4"""

        log_file = self.log_dir / "test.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)

        entries = self.analyzer.parse_log_file(str(log_file))
        self.assertEqual(len(entries), 4)  # Одна строка невалидная
        self.assertEqual(entries[0].level, "INFO")
        self.assertEqual(entries[0].component, "component1")
        self.assertEqual(entries[0].message, "Message 1")
        self.assertEqual(entries[1].level, "WARNING")
        self.assertEqual(entries[1].component, "component2")
        self.assertEqual(entries[1].message, "Message 2")
        self.assertEqual(entries[2].level, "ERROR")
        self.assertEqual(entries[2].component, "component1")
        self.assertEqual(entries[2].message, "Message 3")
        self.assertEqual(entries[3].level, "DEBUG")
        self.assertEqual(entries[3].component, "component3")
        self.assertEqual(entries[3].message, "Message 4")

    def test_scan_log_directory(self):
        """Тест сканирования директории логов"""
        # Создаем несколько файлов
        (self.log_dir / "app.log").touch()
        (self.log_dir / "error.log").touch()
        (self.log_dir / "debug.txt").touch()  # Не .log файл
        subdir = self.log_dir / "subdir"
        subdir.mkdir()
        (subdir / "sub.log").touch()

        log_files = self.analyzer.scan_log_directory("*.log")
        self.assertEqual(len(log_files), 3)  # app.log, error.log, subdir/sub.log
        # Проверяем, что файлы отсортированы по убыванию времени (новые первыми)
        # Поскольку все файлы созданы почти одновременно, порядок может быть любым,
        # но мы проверяем, что все ожидаемые файлы присутствуют
        log_file_names = {f.name for f in log_files}
        self.assertIn("app.log", log_file_names)
        self.assertIn("error.log", log_file_names)
        # sub.log находится в поддиректории, поэтому его имя просто sub.log
        # Но мы проверяем относительный путь?
        # На самом деле, scan_log_directory возвращает список Path objects
        # Мы можем проверить, что есть файл с именем sub.log в поддиректории
        subdir_logs = [f for f in log_files if f.parent.name == "subdir" and f.name == "sub.log"]
        self.assertEqual(len(subdir_logs), 1)

    def test_load_all_logs(self):
        """Тест загрузки всех логов из директории"""
        # Создаем тестовые лог-файлы
        log1_content = """[2023-12-01 10:30:45] - INFO - comp1 - Message 1
[2023-12-01 10:30:46] - WARNING - comp2 - Message 2"""
        log2_content = """[2023-12-01 10:30:47] - ERROR - comp1 - Message 3
[2023-12-01 10:30:48] - DEBUG - comp3 - Message 4"""

        (self.log_dir / "app1.log").write_text(log1_content, encoding="utf-8")
        (self.log_dir / "app2.log").write_text(log2_content, encoding="utf-8")

        entries = self.analyzer.load_all_logs()
        self.assertEqual(len(entries), 4)
        # Проверяем, что логи отсортированы по времени
        self.assertEqual(entries[0].message, "Message 1")
        self.assertEqual(entries[1].message, "Message 2")
        self.assertEqual(entries[2].message, "Message 3")
        self.assertEqual(entries[3].message, "Message 4")

    def test_filter_logs_by_level(self):
        """Тест фильтрации логов по уровню"""
        # Создаем тестовые логи
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Info message"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "WARNING", "comp2", "Warning message"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 47), "ERROR", "comp1", "Error message"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 48), "DEBUG", "comp3", "Debug message"),
        ]
        self.analyzer.log_entries = entries

        # Фильтруем по уровню ERROR
        filtered = self.analyzer.filter_logs(level="ERROR")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].level, "ERROR")
        self.assertEqual(filtered[0].message, "Error message")

        # Фильтруем по уровню INFO
        filtered = self.analyzer.filter_logs(level="INFO")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].level, "INFO")
        self.assertEqual(filtered[0].message, "Info message")

        # Фильтруем по уровню DEBUG
        filtered = self.analyzer.filter_logs(level="DEBUG")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].level, "DEBUG")
        self.assertEqual(filtered[0].message, "Debug message")

    def test_filter_logs_by_component(self):
        """Тест фильтрации логов по компоненту"""
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Message 1"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "WARNING", "comp2", "Message 2"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 47), "ERROR", "comp1", "Message 3"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 48), "DEBUG", "comp3", "Message 4"),
        ]
        self.analyzer.log_entries = entries

        # Фильтруем по компоненту comp1
        filtered = self.analyzer.filter_logs(component="comp1")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(e.component == "comp1" for e in filtered))
        messages = {e.message for e in filtered}
        self.assertEqual(messages, {"Message 1", "Message 3"})

        # Фильтруем по компоненту comp2
        filtered = self.analyzer.filter_logs(component="comp2")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].component, "comp2")
        self.assertEqual(filtered[0].message, "Message 2")

    def test_filter_logs_by_time_range(self):
        """Тест фильтрации логов по временному диапазону"""
        base_time = datetime(2023, 12, 1, 10, 30, 45)
        entries = [
            LogEntry(base_time, "INFO", "comp1", "Message 1"),
            LogEntry(base_time + timedelta(seconds=1), "WARNING", "comp2", "Message 2"),
            LogEntry(base_time + timedelta(seconds=2), "ERROR", "comp1", "Message 3"),
            LogEntry(base_time + timedelta(seconds=3), "DEBUG", "comp3", "Message 4"),
        ]
        self.analyzer.log_entries = entries

        start_time = base_time + timedelta(seconds=1)
        end_time = base_time + timedelta(seconds=2)
        filtered = self.analyzer.filter_logs(start_time=start_time, end_time=end_time)
        self.assertEqual(len(filtered), 2)
        # Должны быть записи за 1 и 2 секунды
        timestamps = [e.timestamp for e in filtered]
        self.assertIn(base_time + timedelta(seconds=1), timestamps)
        self.assertIn(base_time + timedelta(seconds=2), timestamps)

    def test_filter_logs_by_search_term(self):
        """Тест фильтрации логов по поисковому термину"""
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "User login successful"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "WARNING", "comp2", "Disk space low"),
            LogEntry(
                datetime(2023, 12, 1, 10, 30, 47), "ERROR", "comp1", "Failed to connect to database"
            ),
            LogEntry(
                datetime(2023, 12, 1, 10, 30, 48), "DEBUG", "comp3", "Debugging connection pool"
            ),
        ]
        self.analyzer.log_entries = entries

        # Ищем по термину "connection"
        filtered = self.analyzer.filter_logs(search_term="connection")
        self.assertEqual(len(filtered), 2)
        messages = {e.message for e in filtered}
        self.assertIn("Failed to connect to database", messages)
        self.assertIn("Debugging connection pool", messages)

        # Ищем по термину "disk"
        filtered = self.analyzer.filter_logs(search_term="disk")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].message, "Disk space low")

        # Ищем по термину, который отсутствует
        filtered = self.analyzer.filter_logs(search_term="nonexistent")
        self.assertEqual(len(filtered), 0)

    def test_analyze_logs_empty(self):
        """Тест анализа пустого списка логов"""
        result = self.analyzer.analyze_logs()
        self.assertEqual(result.total_entries, 0)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.warning_count, 0)
        self.assertEqual(result.info_count, 0)
        self.assertEqual(result.debug_count, 0)
        self.assertEqual(result.unique_components, [])
        # time_range должно быть кортежем из двух одинаковых времен (сейчас)
        self.assertEqual(len(result.time_range), 2)
        self.assertIsInstance(result.time_range[0], datetime)
        self.assertIsInstance(result.time_range[1], datetime)
        self.assertIsInstance(result.analysis_timestamp, datetime)

    def test_analyze_logs_with_data(self):
        """Тест анализа логов с данными"""
        base_time = datetime(2023, 12, 1, 10, 30, 45)
        entries = [
            LogEntry(base_time, "INFO", "comp1", "Info message 1"),
            LogEntry(base_time + timedelta(seconds=1), "INFO", "comp1", "Info message 2"),
            LogEntry(base_time + timedelta(seconds=2), "WARNING", "comp2", "Warning message"),
            LogEntry(base_time + timedelta(seconds=3), "ERROR", "comp1", "Error message"),
            LogEntry(base_time + timedelta(seconds=4), "CRITICAL", "comp3", "Critical message"),
            LogEntry(base_time + timedelta(seconds=5), "DEBUG", "comp2", "Debug message"),
        ]
        self.analyzer.log_entries = entries

        result = self.analyzer.analyze_logs()
        self.assertEqual(result.total_entries, 6)
        self.assertEqual(result.error_count, 2)  # ERROR + CRITICAL
        self.assertEqual(result.warning_count, 1)
        self.assertEqual(result.info_count, 2)
        self.assertEqual(result.debug_count, 1)
        self.assertEqual(set(result.unique_components), {"comp1", "comp2", "comp3"})
        # Проверяем временной диапазон
        self.assertEqual(result.time_range[0], base_time)
        self.assertEqual(result.time_range[1], base_time + timedelta(seconds=5))

    def test_generate_statistics_empty(self):
        """Тест генерации статистики для пустого списка логов"""
        stats = self.analyzer.generate_statistics()
        self.assertEqual(stats, {})

    def test_generate_statistics_with_data(self):
        """Тест генерации статистики для списка логов с данными"""
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Message 1"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "INFO", "comp1", "Message 2"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 47), "WARNING", "comp2", "Message 3"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 48), "ERROR", "comp1", "Message 4"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 49), "ERROR", "comp1", "Message 4"),  # Дубликат
            LogEntry(datetime(2023, 12, 1, 10, 30, 50), "DEBUG", "comp3", "Message 5"),
        ]
        self.analyzer.log_entries = entries

        stats = self.analyzer.generate_statistics()
        self.assertIn("level_distribution", stats)
        self.assertIn("component_distribution", stats)
        self.assertIn("top_components", stats)
        self.assertIn("top_messages", stats)
        self.assertIn("hourly_activity", stats)
        self.assertIn("avg_interval_between_logs", stats)
        self.assertIn("min_interval", stats)
        self.assertIn("max_interval", stats)
        self.assertIn("timestamp_stats", stats)

        # Проверяем распределение по уровням
        level_dist = stats["level_distribution"]
        self.assertEqual(level_dist["INFO"], 2)
        self.assertEqual(level_dist["WARNING"], 1)
        self.assertEqual(level_dist["ERROR"], 2)
        self.assertEqual(level_dist["DEBUG"], 1)

        # Проверяем распределение по компонентам
        comp_dist = stats["component_distribution"]
        self.assertEqual(comp_dist["comp1"], 4)
        self.assertEqual(comp_dist["comp2"], 1)
        self.assertEqual(comp_dist["comp3"], 1)

        # Проверяем топ компонентов
        top_comps = stats["top_components"]
        self.assertEqual(top_comps[0][0], "comp1")  # comp1 наиболее частый
        self.assertEqual(top_comps[0][1], 4)

        # Проверяем топ сообщений
        top_msgs = stats["top_messages"]
        self.assertEqual(top_msgs[0][0], "Message 4")  # Сообщение встречается 2 раза
        self.assertEqual(top_msgs[0][1], 2)

        # Проверяем статистику временных меток
        timestamp_stats = stats["timestamp_stats"]
        self.assertEqual(timestamp_stats["first_log"], "2023-12-01T10:30:45")
        self.assertEqual(timestamp_stats["last_log"], "2023-12-01T10:30:50")
        self.assertEqual(timestamp_stats["total_duration"], 5.0)  # 5 секунд

    def test_detect_anomalies_empty(self):
        """Тест обнаружения аномалий в пустом списке логов"""
        anomalies = self.analyzer.detect_anomalies()
        self.assertEqual(anomalies, [])

    def test_detect_anomalies_high_error_rate(self):
        """Тест обнаружения аномалии высокой частоты ошибок"""
        base_time = datetime(2023, 12, 1, 10, 30, 45)
        # Создаем логи, где 40% - ошибки (больше порога 10%)
        entries = [
            LogEntry(base_time + timedelta(seconds=i), "ERROR", "comp1", f"Error {i}")
            for i in range(4)
        ] + [
            LogEntry(base_time + timedelta(seconds=i + 4), "INFO", "comp1", f"Info {i}")
            for i in range(6)
        ]
        self.analyzer.log_entries = entries

        anomalies = self.analyzer.detect_anomalies()
        self.assertGreater(len(anomalies), 0)
        # Ищем аномалию высокой частоты ошибок
        high_error_anomaly = next((a for a in anomalies if a["type"] == "high_error_rate"), None)
        self.assertIsNotNone(high_error_anomaly)
        self.assertEqual(high_error_anomaly["severity"], "high")
        self.assertIn("Высокий процент ошибок", high_error_anomaly["description"])
        self.assertEqual(high_error_anomaly["description"].count("40.00%"), 1)  # 4 из 10 = 40%

    def test_detect_anomalies_repeated_messages(self):
        """Тест обнаружения аномалии повторяющихся сообщений"""
        base_time = datetime(2023, 12, 1, 10, 30, 45)
        # Создаем логи с повторяющимся сообщением (более 10 раз)
        entries = [
            LogEntry(base_time + timedelta(seconds=i), "INFO", "comp1", "Repeated message")
            for i in range(12)
        ] + [
            LogEntry(base_time + timedelta(seconds=i + 12), "INFO", "comp1", f"Unique message {i}")
            for i in range(3)
        ]
        self.analyzer.log_entries = entries

        anomalies = self.analyzer.detect_anomalies()
        self.assertGreater(len(anomalies), 0)
        # Ищем аномалию повторяющихся сообщений
        repeated_anomaly = next((a for a in anomalies if a["type"] == "repeated_messages"), None)
        self.assertIsNotNone(repeated_anomaly)
        self.assertEqual(repeated_anomaly["severity"], "medium")
        self.assertIn("Повторяющееся сообщение: Repeated message", repeated_anomaly["description"])
        self.assertEqual(repeated_anomaly["count"], 12)

    def test_detect_anomalies_rapid_logging(self):
        """Тест обнаружения аномалии быстрого логирования"""
        base_time = datetime(2023, 12, 1, 10, 30, 45)
        # Создаем логи с очень короткими интервалами (быстрое логирование)
        entries = []
        for i in range(20):
            # Интервал 0.01 секунды между записями (очень быстро)
            entries.append(
                LogEntry(base_time + timedelta(seconds=i * 0.01), "INFO", "comp1", f"Message {i}")
            )
        self.analyzer.log_entries = entries

        anomalies = self.analyzer.detect_anomalies()
        self.assertGreater(len(anomalies), 0)
        # Ищем аномалию быстрого логирования
        rapid_anomaly = next((a for a in anomalies if a["type"] == "rapid_logging"), None)
        self.assertIsNotNone(rapid_anomaly)
        self.assertEqual(rapid_anomaly["severity"], "medium")
        self.assertIn("Обнаружено", rapid_anomaly["description"])
        self.assertIn("случаев быстрого логирования", rapid_anomaly["description"])

    def test_export_filtered_logs_csv(self):
        """Тест экспорта отфильтрованных логов в CSV"""
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Message 1"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "ERROR", "comp2", "Message 2"),
        ]
        self.analyzer.log_entries = entries

        # Фильтруем только ошибки
        error_logs = self.analyzer.filter_logs(level="ERROR")
        self.assertEqual(len(error_logs), 1)

        output_file = self.log_dir / "export.csv"
        exported_path = self.analyzer.export_filtered_logs(
            error_logs, str(output_file), format_type="csv"
        )
        self.assertEqual(exported_path, str(output_file))
        self.assertTrue(output_file.exists())

        # Проверяем содержимое CSV
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["level"], "ERROR")
        self.assertEqual(rows[0]["component"], "comp2")
        self.assertEqual(rows[0]["message"], "Message 2")
        # Проверяем timestamp (должен быть в ISO формате)
        self.assertEqual(rows[0]["timestamp"], "2023-12-01T10:30:46")

    def test_export_filtered_logs_json(self):
        """Тест экспорта отфильтрованных логов в JSON"""
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Message 1"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "ERROR", "comp2", "Message 2"),
        ]
        self.analyzer.log_entries = entries

        # Фильтруем только ошибки
        error_logs = self.analyzer.filter_logs(level="ERROR")
        self.assertEqual(len(error_logs), 1)

        output_file = self.log_dir / "export.json"
        exported_path = self.analyzer.export_filtered_logs(
            error_logs, str(output_file), format_type="json"
        )
        self.assertEqual(exported_path, str(output_file))
        self.assertTrue(output_file.exists())

        # Проверяем содержимое JSON
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["level"], "ERROR")
        self.assertEqual(data[0]["component"], "comp2")
        self.assertEqual(data[0]["message"], "Message 2")
        self.assertEqual(data[0]["timestamp"], "2023-12-01T10:30:46")

    def test_export_filtered_logs_txt(self):
        """Тест экспорта отфильтрованных логов в TXT"""
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Message 1"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "ERROR", "comp2", "Message 2"),
        ]
        self.analyzer.log_entries = entries

        # Фильтруем только ошибки
        error_logs = self.analyzer.filter_logs(level="ERROR")
        self.assertEqual(len(error_logs), 1)

        output_file = self.log_dir / "export.txt"
        exported_path = self.analyzer.export_filtered_logs(
            error_logs, str(output_file), format_type="txt"
        )
        self.assertEqual(exported_path, str(output_file))
        self.assertTrue(output_file.exists())

        # Проверяем содержимое TXT
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        # Формат: [timestamp] - level - component - message
        line = lines[0].strip()
        self.assertTrue(line.startswith("[2023-12-01 10:30:46]"))
        self.assertIn(" - ERROR - ", line)
        self.assertIn("comp2", line)
        self.assertIn("Message 2", line)

    @patch("utils.logger_analyzer.AdvancedLoggerAnalyzer.visualize_logs")
    def test_visualize_logs(self, mock_visualize):
        """Тест визуализации логов (мокируем фактическое построение графиков)"""
        # Настраиваем мок для возврата тестового пути
        mock_visualize.return_value = "test_viz.png"

        # Добавляем некоторые логи для визуализации
        entries = [
            LogEntry(datetime(2023, 12, 1, 10, 30, 45), "INFO", "comp1", "Message 1"),
            LogEntry(datetime(2023, 12, 1, 10, 30, 46), "ERROR", "comp2", "Message 2"),
        ]
        self.analyzer.log_entries = entries

        output_path = self.analyzer.visualize_logs()
        self.assertEqual(output_path, "test_viz.png")
        mock_visualize.assert_called_once()

        # Тестируем визуализацию с пустыми логами
        self.analyzer.log_entries = []
        output_path = self.analyzer.visualize_logs()
        self.assertEqual(output_path, "")  # Должен вернуть пустую строку для пустых данных


if __name__ == "__main__":
    unittest.main()
