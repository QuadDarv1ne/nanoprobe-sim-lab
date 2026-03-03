#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-панель управления проектом "Лаборатория моделирования нанозонда"
Этот модуль предоставляет веб-интерфейс для управления всеми аспектами проекта,
включая симулятор СЗМ, анализатор изображений и наземную станцию SSTV.
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Добавляем путь к utils для импорта служебных модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Project root for component paths
project_root = Path(__file__).parent.parent.parent

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from utils.performance_monitor import PerformanceMonitor
from utils.system_monitor import SystemMonitor
from utils.config_manager import ConfigManager
from utils.cache_manager import CacheManager
from utils.data_manager import DataManager
from utils.data_exporter import DataExporter
from utils.database import DatabaseManager
from utils.cli_utils import Colors


class WebDashboard:
    """
    Класс веб-панели управления проектом
    Предоставляет веб-интерфейс для управления всеми компонентами проекта
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        """
        Инициализация веб-панели

        Args:
            host: Хост сервера
            port: Порт сервера
        """
        self.host = host
        self.port = port

        # Инициализация Flask приложения
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "nanoprobe_simulation_lab_secret_key"

        # Инициализация SocketIO
        self.socketio = SocketIO(
            self.app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25
        )

        # Инициализация служебных компонентов
        self.logger = Logger("WebDashboard")
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.data_manager = DataManager()
        self.data_exporter = DataExporter(output_dir="output")
        self.database = DatabaseManager(db_path="data/nanoprobe.db")

        # Время запуска
        self._start_time = datetime.now()

        # Регистрация маршрутов
        self._register_routes()

        # Регистрация обработчиков SocketIO
        self._register_socket_handlers()

        self.logger.log_system_event("Веб-панель инициализирована", "INFO")

    def _register_routes(self):
        """Регистрация HTTP маршрутов"""

        @self.app.route("/")
        def index():
            """Главная страница веб-панели"""
            return render_template("dashboard.html")

        @self.app.route("/api/system_info")
        def api_system_info():
            """API для получения информации о системе"""
            try:
                system_info = self.system_monitor.get_current_metrics()

                # Добавляем информацию о проекте
                system_info["project_info"] = {
                    "name": "Nanoprobe Simulation Lab",
                    "version": "1.0.0",
                    "status": "running",
                    "uptime": self._get_uptime(),
                }

                return jsonify(system_info)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения информации о системе: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/performance_data")
        def api_performance_data():
            """API для получения данных о производительности"""
            try:
                performance_data = self.performance_monitor.get_current_metrics()
                return jsonify(performance_data)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения данных о производительности: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/component_status")
        def api_component_status():
            """API для получения статуса компонентов"""
            try:
                component_status = {}

                # Проверяем SPM симулятор
                spm_python = (
                    project_root
                    / "components"
                    / "cpp-spm-hardware-sim"
                    / "src"
                    / "spm_simulator.py"
                )
                spm_cpp = (
                    project_root / "components" / "cpp-spm-hardware-sim" / "build" / "spm-simulator"
                )
                spm_exists = spm_python.exists() or spm_cpp.exists()
                component_status["spm_simulator"] = {
                    "status": "ready" if spm_exists else "not_installed",
                    "processes": 0,
                    "python_available": spm_python.exists(),
                    "cpp_available": spm_cpp.exists(),
                }

                # Проверяем анализатор изображений
                analyzer_path = (
                    project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py"
                )
                component_status["image_analyzer"] = {
                    "status": "ready" if analyzer_path.exists() else "not_installed",
                    "processes": 0,
                }

                # Проверяем SSTV станцию
                sstv_path = (
                    project_root / "components" / "py-sstv-groundstation" / "src" / "main.py"
                )
                component_status["sstv_station"] = {
                    "status": "ready" if sstv_path.exists() else "not_installed",
                    "processes": 0,
                }

                component_status["web_dashboard"] = {"status": "running", "processes": 1}

                return jsonify(component_status)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статуса компонентов: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/logs")
        def api_logs():
            """API для получения логов"""
            try:
                # Получаем последние N записей из логов
                log_file = "logs/web_dashboard.log"
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        recent_logs = lines[-50:]  # Последние 50 строк
                else:
                    recent_logs = ["Лог-файл не найден"]

                return jsonify({"logs": recent_logs})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения логов: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/config", methods=["GET", "POST"])
        def api_config():
            """API для управления конфигурацией"""
            try:
                if request.method == "GET":
                    # Возвращаем текущую конфигурацию
                    config = self.config_manager.get_config()
                    return jsonify(config)
                elif request.method == "POST":
                    # Обновляем конфигурацию
                    new_config = request.json
                    self.config_manager.update_config(new_config)
                    return jsonify({"status": "success", "message": "Конфигурация обновлена"})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка управления конфигурацией: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/cache_stats")
        def api_cache_stats():
            """API для получения статистики кэша"""
            try:
                cache_stats = self.cache_manager.get_cache_statistics()
                return jsonify(cache_stats)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статистики кэша: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/export/data", methods=["POST"])
        def api_export_data():
            """API для экспорта данных"""
            try:
                data = request.json.get('data')
                filename = request.json.get('filename', 'export')
                fmt = request.json.get('format', 'csv')

                if not data:
                    return jsonify({"error": "Данные не предоставлены"}), 400

                filepath = self.data_exporter.export(data, filename, fmt=fmt)
                return jsonify({
                    "status": "success",
                    "filepath": str(filepath),
                    "format": fmt
                })
            except Exception as e:
                self.error_handler.log_error(f"Ошибка экспорта данных: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/export/surface", methods=["POST"])
        def api_export_surface():
            """API для экспорта данных поверхности"""
            try:
                import numpy as np
                surface_data = np.array(request.json.get('surface_data', []))
                metadata = request.json.get('metadata', {})
                fmt = request.json.get('format', 'hdf5')

                if surface_data.size == 0:
                    return jsonify({"error": "Данные поверхности пусты"}), 400

                filepath = self.data_exporter.export_surface_data(surface_data, metadata, fmt=fmt)
                return jsonify({
                    "status": "success",
                    "filepath": str(filepath)
                })
            except Exception as e:
                self.error_handler.log_error(f"Ошибка экспорта поверхности: {e}")
                return jsonify({"error": str(e)}), 500

    def _register_socket_handlers(self):
        """Регистрация обработчиков SocketIO событий"""

        @self.socketio.on("connect")
        def handle_connect():
            """Обработка подключения клиента"""
            self.logger.log_system_event("Клиент подключен к веб-панели", "INFO")
            emit("connection_response", {"data": "Connected to Nanoprobe Simulation Lab Dashboard"})
            # Отправляем приветственное сообщение с uptime
            emit("system_status", {
                "status": "online",
                "uptime": self._get_uptime(),
                "timestamp": datetime.now().isoformat()
            })

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Обработка отключения клиента"""
            self.logger.log_system_event("Клиент отключен от веб-панели", "INFO")

        @self.socketio.on("request_metrics")
        def handle_request_metrics():
            """Запрос метрик системы в realtime"""
            try:
                import psutil
                metrics = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'timestamp': datetime.now().isoformat()
                }
                emit('metrics', metrics)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения метрик: {e}")

        @self.socketio.on("request_system_update")
        def handle_system_update_request():
            """Обработка запроса на обновление информации о системе"""
            try:
                system_info = self.system_monitor.get_current_metrics()
                system_info["project_info"] = {
                    "name": "Nanoprobe Simulation Lab",
                    "version": "1.0.0",
                    "status": "running",
                    "uptime": self._get_uptime(),
                }
                emit("system_update", system_info)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка отправки обновления системы: {e}")

        @self.socketio.on("request_performance_update")
        def handle_performance_update_request():
            """Обработка запроса на обновление информации о производительности"""
            try:
                performance_data = self.performance_monitor.get_current_metrics()
                emit("performance_update", performance_data)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка отправки обновления производительности: {e}")

        @self.socketio.on("export_data")
        def handle_export_data(data):
            """Обработка запроса на экспорт данных через WebSocket"""
            try:
                export_data = data.get('data')
                filename = data.get('filename', 'export')
                fmt = data.get('format', 'csv')

                filepath = self.data_exporter.export(export_data, filename, fmt=fmt)
                emit("export_result", {
                    "status": "success",
                    "filepath": str(filepath),
                    "format": fmt
                })
            except Exception as e:
                emit("export_result", {
                    "status": "error",
                    "error": str(e)
                })

        @self.socketio.on("execute_command")
        def handle_execute_command(data):
            """Обработка команды выполнения"""
            try:
                command = data.get("command", "")
                params = data.get("params", {})

                # Логика выполнения команд
                if command == "start_simulation":
                    result = self._execute_start_simulation(params)
                elif command == "stop_simulation":
                    result = self._execute_stop_simulation(params)
                elif command == "analyze_image":
                    result = self._execute_analyze_image(params)
                elif command == "cleanup_cache":
                    result = self._execute_cleanup_cache(params)
                elif command == "export_surface":
                    result = self._execute_export_surface(params)
                else:
                    result = {"status": "error", "message": f"Неизвестная команда: {command}"}

                emit("command_result", result)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка выполнения команды: {e}")
                emit("command_result", {"status": "error", "message": str(e)})

    def _get_uptime(self) -> str:
        """Получение времени работы системы"""
        # В реальной реализации это будет вычисляться с момента запуска
        start_time = getattr(self, "_start_time", datetime.now())
        uptime = datetime.now() - start_time
        return str(uptime).split(".")[0]  # Убираем микросекунды

    def _execute_start_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды запуска симуляции"""
        try:
            # Проверяем доступность SPM симулятора
            spm_python = (
                project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py"
            )
            spm_cpp = (
                project_root / "components" / "cpp-spm-hardware-sim" / "build" / "spm-simulator"
            )

            if not spm_python.exists() and not spm_cpp.exists():
                return {"status": "error", "message": "SPM симулятор не найден"}

            simulation_id = f"sim_{int(time.time())}"
            self.logger.log_system_event(f"Запуск симуляции: {simulation_id}", "INFO")

            # Запускаем симулятор в фоновом процессе
            simulator_path = str(spm_cpp if spm_cpp.exists() else spm_python)
            cmd = [sys.executable, simulator_path] if not spm_cpp.exists() else [str(spm_cpp)]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(project_root)
            )

            # Сохраняем информацию о процессе
            if not hasattr(self, "_active_processes"):
                self._active_processes = {}
            self._active_processes[simulation_id] = process

            return {
                "status": "success",
                "message": f"Симуляция запущена: {simulation_id}",
                "simulation_id": simulation_id,
                "process_id": process.pid,
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка запуска симуляции: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_stop_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды остановки симуляции"""
        try:
            simulation_id = params.get("simulation_id", "")

            if (
                not hasattr(self, "_active_processes")
                or simulation_id not in self._active_processes
            ):
                return {
                    "status": "warning",
                    "message": f"Симуляция {simulation_id} не найдена среди активных",
                }

            process = self._active_processes[simulation_id]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            del self._active_processes[simulation_id]
            self.logger.log_system_event(f"Остановка симуляции: {simulation_id}", "INFO")

            return {"status": "success", "message": f"Симуляция остановлена: {simulation_id}"}
        except Exception as e:
            self.error_handler.log_error(f"Ошибка остановки симуляции: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_analyze_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды анализа изображения"""
        try:
            image_path = params.get("image_path", "")
            self.logger.log_system_event(f"Анализ изображения: {image_path}", "INFO")

            # Здесь будет логика анализа изображения
            return {
                "status": "success",
                "message": f"Изображение проанализировано: {image_path}",
                "results": {"analysis_complete": True},
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка анализа изображения: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_cleanup_cache(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды очистки кэша"""
        try:
            force = params.get("force", False)
            self.logger.log_system_event("Запуск очистки кэша", "INFO")

            cleanup_result = self.cache_manager.cleanup_cache(force=force)

            return {"status": "success", "message": "Кэш очищен", "cleanup_result": cleanup_result}
        except Exception as e:
            self.error_handler.log_error(f"Ошибка очистки кэша: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_export_surface(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды экспорта данных поверхности"""
        try:
            import numpy as np
            surface_data = np.array(params.get('surface_data', []))
            filename = params.get('filename', 'surface_export')
            fmt = params.get('format', 'hdf5')
            metadata = params.get('metadata', {})

            if surface_data.size == 0:
                return {"status": "error", "message": "Данные поверхности пусты"}

            filepath = self.data_exporter.export_surface_data(surface_data, metadata, filename, fmt)
            
            self.logger.log_system_event(f"Экспорт поверхности: {filepath}", "INFO")
            
            return {
                "status": "success",
                "message": f"Поверхность экспортирована: {filepath}",
                "filepath": str(filepath)
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка экспорта поверхности: {e}")
            return {"status": "error", "message": str(e)}

    def start_server(self, open_browser: bool = True):
        """
        Запуск веб-сервера

        Args:
            open_browser: Открывать ли браузер автоматически
        """
        import socket

        self._start_time = datetime.now()

        # Проверяем доступность порта
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((self.host, self.port))
            sock.close()
        except OSError as e:
            self.logger.log_system_event(f"Порт {self.port} занят: {e}", "ERROR")
            print(f"Ошибка: Порт {self.port} уже используется")
            print("Попробуйте другой порт: python web_dashboard.py --port 5001")
            return

        self.logger.log_system_event(f"Запуск веб-панели на http://{self.host}:{self.port}", "INFO")

        print("=" * 60)
        print(f"Сервер запущен на http://{self.host}:{self.port}")
        print("Нажмите Ctrl+C для остановки сервера")
        print("=" * 60)

        # Открываем браузер в отдельном потоке
        if open_browser:

            def open_browser_func():
                """Open browser in a separate thread"""
                time.sleep(2)
                webbrowser.open(f"http://{self.host}:{self.port}")

            browser_thread = threading.Thread(target=open_browser_func)
            browser_thread.daemon = True
            browser_thread.start()

        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except KeyboardInterrupt:
            print("\nОстановка веб-панели...")
            self._cleanup_processes()
            self.logger.log_system_event("Веб-панель остановлена", "INFO")
        except Exception as e:
            self.error_handler.log_error(f"Ошибка запуска веб-панели: {e}")
            print(f"Ошибка: {e}")

    def _cleanup_processes(self):
        """Очищает все активные процессы при завершении"""
        if hasattr(self, "_active_processes"):
            for sim_id, process in self._active_processes.items():
                try:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=3)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            self._active_processes.clear()


def main():
    """Главная функция запуска веб-панели"""
    import argparse

    parser = argparse.ArgumentParser(description="Веб-панель управления проектом")
    parser.add_argument("--host", default="127.0.0.1", help="Хост сервера")
    parser.add_argument("--port", type=int, default=5000, help="Порт сервера")
    parser.add_argument(
        "--no-browser", action="store_true", help="Не открывать браузер автоматически"
    )

    args = parser.parse_args()

    # Создаем и запускаем веб-панель
    dashboard = WebDashboard(host=args.host, port=args.port)
    dashboard.start_server(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
