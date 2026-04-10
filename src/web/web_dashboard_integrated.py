#!/usr/bin/env python3
"""
Веб-панель управления проектом "Лаборатория моделирования нанозонда"
С интегрированным reverse proxy для FastAPI

Этот модуль предоставляет веб-интерфейс с полной интеграцией FastAPI REST API
через reverse proxy и общие сессии/токены.
"""

import os
import sys
import threading
import webbrowser
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import requests

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from functools import wraps

# Добавляем путь к utils для импорта служебных модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Project root for component paths
project_root = Path(__file__).parent.parent.parent

from utils.logger import NanoprobeLogger
from utils.core.error_handler import ErrorHandler
from utils.performance_monitor import PerformanceMonitor
from utils.system_monitor import SystemMonitor
from utils.config.config_manager import ConfigManager
from utils.caching.cache_manager import CacheManager
from utils.data.data_manager import DataManager
from utils.data.data_exporter import DataExporter
from utils.database.database import DatabaseManager, get_database
from utils.surface_comparator import compare_surfaces as compare_surfaces_util
from utils.ai.defect_analyzer import analyze_defects as analyze_defects_util

# Импорт reverse proxy
try:
    from api.reverse_proxy import register_proxy, FASTAPI_URL, JWT_SECRET
    PROXY_AVAILABLE = True
except ImportError:
    PROXY_AVAILABLE = False
    FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')


def login_required(f):
    """Декоратор для защиты маршрутов аутентификацией"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        """Проверяет наличие активной сессии пользователя."""
        if not session.get('logged_in'):
            return jsonify({'error': 'Требуется аутентификация'}), 401
        return f(*args, **kwargs)
    return decorated_function


class IntegratedWebDashboard:
    """
    Класс веб-панели управления с интеграцией FastAPI
    Предоставляет веб-интерфейс с reverse proxy к FastAPI API
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
        self.fastapi_url = FASTAPI_URL

        # Инициализация Flask приложения
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = os.getenv(
            'FLASK_SECRET_KEY',
            'nanoprobe_simulation_lab_secret_key_change_in_production'
        )
        self.app.config['SESSION_TYPE'] = 'filesystem'
        self.app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 час

        # Инициализация SocketIO
        self.socketio = SocketIO(
            self.app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25
        )

        # Инициализация служебных компонентов
        self.logger = NanoprobeLogger()
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.data_manager = DataManager()
        self.data_exporter = DataExporter(output_dir="output")
        self.database = DatabaseManager(db_path="data/nanoprobe.db")

        # Время запуска
        self._start_time = datetime.now(timezone.utc)

        # Регистрация reverse proxy (если доступен)
        if PROXY_AVAILABLE:
            register_proxy(self.app)
            self.logger.log_system_event("Reverse proxy зарегистрирован", "INFO")

        # Регистрация маршрутов
        self._register_routes()

        # Регистрация обработчиков SocketIO
        self._register_socket_handlers()

        self.logger.log_system_event("Веб-панель инициализирована с интеграцией FastAPI", "INFO")

    def _register_routes(self):
        """Регистрация HTTP маршрутов"""

        @self.app.route("/")
        def index():
            """Главная страница веб-панели"""
            return render_template("dashboard_integrated.html")

        @self.app.route("/login")
        def login_page():
            """Страница входа"""
            return render_template("login.html")

        # ==================== Аутентификация ====================

        @self.app.route("/api/auth/login", methods=["POST"])
        def api_login():
            """
            Аутентификация через FastAPI
            Сохраняет токены в сессии Flask
            """
            try:
                username = request.form.get('username')
                password = request.form.get('password')

                if not username or not password:
                    return jsonify({'error': 'Требуется имя пользователя и пароль'}), 400

                # Запрос к FastAPI
                response = requests.post(
                    f"{self.fastapi_url}/api/v1/auth/login",
                    data={'username': username, 'password': password},
                    timeout=10
                )

                if response.status_code == 200:
                    tokens = response.json()

                    # Сохранение токенов в сессии
                    session['access_token'] = tokens.get('access_token')
                    session['refresh_token'] = tokens.get('refresh_token')
                    session['logged_in'] = True
                    session.permanent = True

                    # Декодирование токена для информации о пользователе
                    try:
                        import jwt
                        payload = jwt.decode(
                            tokens.get('access_token'),
                            JWT_SECRET,
                            algorithms=['HS256']
                        )
                        session['user_id'] = payload.get('sub')
                        session['username'] = payload.get('username')
                    except Exception:
                        session['username'] = username

                    self.logger.log_system_event(f"Пользователь {username} вошёл в систему", "INFO")
                    return jsonify({
                        'status': 'success',
                        'username': session['username'],
                        'redirect': '/'
                    })
                else:
                    error_data = response.json()
                    return jsonify({
                        'status': 'error',
                        'message': error_data.get('detail', 'Ошибка аутентификации')
                    }), response.status_code

            except requests.RequestException as e:
                self.error_handler.log_error(f"Ошибка аутентификации: {e}")
                return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/auth/logout", methods=["POST"])
        def api_logout():
            """Выход из системы"""
            username = session.get('username', 'unknown')
            session.clear()
            self.logger.log_system_event(f"Пользователь {username} вышел из системы", "INFO")
            return jsonify({'status': 'success', 'redirect': '/login'})

        @self.app.route("/api/auth/status")
        def api_auth_status():
            """Проверка статуса аутентификации"""
            return jsonify({
                'logged_in': session.get('logged_in', False),
                'username': session.get('username'),
                'user_id': session.get('user_id')
            })

        # ==================== Проксирование FastAPI запросов ====================

        def proxy_to_fastapi(endpoint, method='GET', **kwargs):
            """Универсальная функция проксирования к FastAPI"""
            try:
                url = f"{self.fastapi_url}{endpoint}"
                headers = {}

                # Добавляем токен авторизации
                token = session.get('access_token')
                if token:
                    headers['Authorization'] = f'Bearer {token}'

                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=30,
                    **kwargs
                )

                return response
            except requests.RequestException as e:
                self.error_handler.log_error(f"Ошибка проксирования {endpoint}: {e}")
                return None

        @self.app.route("/api/proxy/scans", methods=["GET"])
        @login_required
        def proxy_get_scans():
            """Проксирование GET /scans"""
            response = proxy_to_fastapi('/api/v1/scans', params=request.args)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/scans/<int:scan_id>", methods=["GET"])
        @login_required
        def proxy_get_scan(scan_id):
            """Проксирование GET /scans/{id}"""
            response = proxy_to_fastapi(f'/api/v1/scans/{scan_id}')
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/scans", methods=["POST"])
        @login_required
        def proxy_create_scan():
            """Проксирование POST /scans"""
            response = proxy_to_fastapi('/api/v1/scans', method='POST', json=request.json)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/scans/<int:scan_id>", methods=["DELETE"])
        @login_required
        def proxy_delete_scan(scan_id):
            """Проксирование DELETE /scans/{id}"""
            response = proxy_to_fastapi(f'/api/v1/scans/{scan_id}', method='DELETE')
            if response:
                return jsonify({'status': 'success'}), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/simulations", methods=["GET"])
        @login_required
        def proxy_get_simulations():
            """Проксирование GET /simulations"""
            response = proxy_to_fastapi('/api/v1/simulations', params=request.args)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/simulations", methods=["POST"])
        @login_required
        def proxy_create_simulation():
            """Проксирование POST /simulations"""
            response = proxy_to_fastapi('/api/v1/simulations', method='POST', json=request.json)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/analysis/defects", methods=["POST"])
        @login_required
        def proxy_analyze_defects():
            """Проксирование POST /analysis/defects"""
            response = proxy_to_fastapi('/api/v1/analysis/defects', method='POST', json=request.json)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/comparison/surfaces", methods=["POST"])
        @login_required
        def proxy_compare_surfaces():
            """Проксирование POST /comparison/surfaces"""
            response = proxy_to_fastapi('/api/v1/comparison/surfaces', method='POST', json=request.json)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/reports", methods=["GET"])
        @login_required
        def proxy_get_reports():
            """Проксирование GET /reports"""
            response = proxy_to_fastapi('/api/v1/reports', params=request.args)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        @self.app.route("/api/proxy/reports/generate", methods=["POST"])
        @login_required
        def proxy_generate_report():
            """Проксирование POST /reports/generate"""
            response = proxy_to_fastapi('/api/v1/reports/generate', method='POST', json=request.json)
            if response:
                return jsonify(response.json()), response.status_code
            return jsonify({'error': 'FastAPI недоступен'}), 503

        # ==================== Локальные API (без проксирования) ====================

        @self.app.route("/api/system_info")
        def api_system_info():
            """API для получения информации о системе"""
            try:
                system_info = self.system_monitor.get_current_metrics()
                system_info["project_info"] = {
                    "name": "Nanoprobe Simulation Lab",
                    "version": "1.0.0",
                    "status": "running",
                    "uptime": self._get_uptime(),
                    "fastapi_integration": PROXY_AVAILABLE,
                    "fastapi_url": self.fastapi_url,
                }
                return jsonify(system_info)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения информации о системе: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/health")
        def api_health():
            """Health check endpoint"""
            try:
                return jsonify({
                    "status": "healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime": self._get_uptime(),
                    "components": {
                        "database": "ok",
                        "socketio": "ok" if hasattr(self, 'socketio') else "not_initialized",
                        "fastapi": "integrated" if PROXY_AVAILABLE else "not_integrated"
                    }
                })
            except Exception as e:
                return jsonify({"status": "unhealthy", "error": str(e)}), 500

        @self.app.route("/api/performance_data")
        def api_performance_data():
            """API для получения данных о производительности"""
            try:
                performance_data = self.performance_monitor.get_current_metrics()
                return jsonify(performance_data)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения данных о производительности: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/component_status")
        def api_component_status():
            """API для получения статуса компонентов"""
            try:
                component_status = {}

                # Проверяем SPM симулятор
                spm_python = (
                    project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py"
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
                component_status["fastapi"] = {
                    "status": "integrated" if PROXY_AVAILABLE else "not_integrated",
                    "url": self.fastapi_url
                }

                return jsonify(component_status)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статуса компонентов: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/components")
        def api_components():
            """API для получения списка компонентов в формате для frontend"""
            try:
                component_status = {}

                # Проверяем SPM симулятор
                spm_python = (
                    project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py"
                )
                spm_cpp = (
                    project_root / "components" / "cpp-spm-hardware-sim" / "build" / "spm-simulator"
                )
                spm_exists = spm_python.exists() or spm_cpp.exists()
                spm_running = False
                if hasattr(self, "_active_processes") and "spm_simulator" in self._active_processes:
                    proc = self._active_processes["spm_simulator"]
                    spm_running = proc.poll() is None

                component_status["spm_simulator"] = {
                    "name": "SPM Simulator",
                    "status": "running" if spm_running else ("ready" if spm_exists else "not_installed"),
                    "processes": 1 if spm_running else 0,
                }

                # Проверяем анализатор изображений
                analyzer_path = project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py"
                analyzer_running = False
                if hasattr(self, "_active_processes") and "image_analyzer" in self._active_processes:
                    proc = self._active_processes["image_analyzer"]
                    analyzer_running = proc.poll() is None

                component_status["image_analyzer"] = {
                    "name": "Image Analyzer",
                    "status": "running" if analyzer_running else ("ready" if analyzer_path.exists() else "not_installed"),
                    "processes": 1 if analyzer_running else 0,
                }

                # Проверяем SSTV станцию
                sstv_path = project_root / "components" / "py-sstv-groundstation" / "src" / "main.py"
                sstv_running = False
                if hasattr(self, "_active_processes") and "sstv_station" in self._active_processes:
                    proc = self._active_processes["sstv_station"]
                    sstv_running = proc.poll() is None

                component_status["sstv_station"] = {
                    "name": "SSTV Station",
                    "status": "running" if sstv_running else ("ready" if sstv_path.exists() else "not_installed"),
                    "processes": 1 if sstv_running else 0,
                }

                component_status["web_dashboard"] = {"name": "Web Dashboard", "status": "running", "processes": 1}
                component_status["fastapi"] = {"name": "FastAPI", "status": "integrated" if PROXY_AVAILABLE else "not_integrated", "processes": 1 if PROXY_AVAILABLE else 0}

                # Преобразуем в список для frontend
                components_list = []
                for comp_key, comp_data in component_status.items():
                    components_list.append({
                        "name": comp_key,
                        "status": comp_data.get("status", "unknown"),
                        "description": comp_data.get("name", comp_key),
                        "processes": comp_data.get("processes", 0),
                    })

                return jsonify(components_list)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения списка компонентов: {e}")
                return jsonify([]), 500

        @self.app.route("/api/processes", methods=["GET"])
        def api_processes():
            """API для получения статуса всех процессов компонентов"""
            try:
                processes = {}
                if hasattr(self, "_active_processes"):
                    for component, proc in self._active_processes.items():
                        poll_result = proc.poll()
                        processes[component] = {
                            "pid": proc.pid,
                            "status": "running" if poll_result is None else "stopped",
                            "exit_code": poll_result,
                            "returncode": proc.returncode
                        }

                return jsonify({
                    "active_count": len(getattr(self, "_active_processes", {})),
                    "processes": processes
                })
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статуса процессов: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/logs/component/<component_name>", methods=["GET"])
        def api_component_logs(component_name):
            """API для получения логов компонента"""
            try:
                log_dir = project_root / "logs" / "components"
                stdout_log = log_dir / f"{component_name}_stdout.log"
                stderr_log = log_dir / f"{component_name}_stderr.log"

                logs = {"stdout": "", "stderr": ""}

                if stdout_log.exists():
                    with open(stdout_log, "r", encoding="utf-8", errors="ignore") as f:
                        logs["stdout"] = f.read()[-5000:]

                if stderr_log.exists():
                    with open(stderr_log, "r", encoding="utf-8", errors="ignore") as f:
                        logs["stderr"] = f.read()[-5000:]

                return jsonify(logs)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка чтения логов: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/actions/quick", methods=["POST"])
        def api_quick_action():
            """API для быстрых действий"""
            try:
                data = request.json
                action = data.get("action", "")

                actions_map = {
                    "clean_cache": self.cache_manager.auto_cleanup,
                    "restart_all": self._restart_all_components,
                    "stop_all": self._stop_all_components,
                }

                if action not in actions_map:
                    return jsonify({
                        "success": False,
                        "error": f"Неизвестное действие: {action}"
                    }), 400

                result = actions_map[action]()
                return jsonify({
                    "success": True,
                    "message": f"Действие '{action}' выполнено",
                    "result": result
                })
            except Exception as e:
                self.error_handler.log_error(f"Ошибка быстрого действия: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        def _stop_all_components(self):
            """Остановить все компоненты"""
            stopped = []
            if hasattr(self, "_active_processes"):
                for component, process in list(self._active_processes.items()):
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1)
                    del self._active_processes[component]
                    stopped.append(component)
                    self.logger.log_system_event(f"Остановлен: {component}", "INFO")
            return {"stopped": stopped, "count": len(stopped)}

        def _restart_all_components(self):
            """Перезапустить все активные компоненты"""
            restarted = []
            failed = []

            self._stop_all_components()

            import time
            time.sleep(1)

            component_paths = {
                "spm_simulator": project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py",
                "image_analyzer": project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py",
                "sstv_station": project_root / "components" / "py-sstv-groundstation" / "src" / "main.py",
            }

            for component, path in component_paths.items():
                if path.exists():
                    try:
                        process = subprocess.Popen(
                            [sys.executable, str(path)],
                            cwd=str(project_root),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        if not hasattr(self, "_active_processes"):
                            self._active_processes = {}
                        self._active_processes[component] = process
                        restarted.append(component)
                    except Exception as e:
                        failed.append({"component": component, "error": str(e)})

            return {"restarted": restarted, "failed": failed}

        @self.app.route("/api/logs")
        def api_logs():
            """API для получения логов"""
            try:
                log_file = "logs/web_dashboard.log"
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        recent_logs = lines[-50:]
                else:
                    recent_logs = ["Лог-файл не найден"]
                return jsonify({"logs": recent_logs})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения логов: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/actions/start_component", methods=["POST"])
        def api_start_component_action():
            """API для запуска компонента"""
            try:
                data = request.json
                component = data.get("component", "unknown")

                component_paths = {
                    "spm_simulator": project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py",
                    "image_analyzer": project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py",
                    "sstv_station": project_root / "components" / "py-sstv-groundstation" / "src" / "main.py",
                }

                if component not in component_paths:
                    return jsonify({"success": False, "error": f"Компонент '{component}' не найден"}), 404

                component_path = component_paths[component]
                if not component_path.exists():
                    return jsonify({"success": False, "error": f"Файл не найден: {component_path}"}), 404

                if not hasattr(self, "_active_processes"):
                    self._active_processes = {}

                if component in self._active_processes:
                    proc = self._active_processes[component]
                    if proc.poll() is None:
                        return jsonify({"success": False, "error": f"Уже запущен (PID: {proc.pid})"}), 409

                log_dir = project_root / "logs" / "components"
                log_dir.mkdir(parents=True, exist_ok=True)

                # Аргументы для запуска компонента в фоновом режиме
                component_args = {
                    "spm_simulator": ["--size", "10", "--no-visualize"],
                    "image_analyzer": ["--stats"],
                    "sstv_station": ["--check"],
                }

                args = component_args.get(component, [])

                with open(log_dir / f"{component}_stdout.log", "ab") as out_f, \
                     open(log_dir / f"{component}_stderr.log", "ab") as err_f:
                    process = subprocess.Popen(
                        [sys.executable, str(component_path)] + args,
                        cwd=str(project_root),
                        stdout=out_f, stderr=err_f,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )

                self._active_processes[component] = process
                self.logger.log_system_event(f"Запуск: {component} (PID: {process.pid})", "INFO")

                import time
                time.sleep(0.5)
                if process.poll() is not None:
                    # WebSocket уведомление об ошибке
                    if hasattr(self, 'socketio'):
                        from flask_socketio import emit
                        emit('component_status', {
                            'component': component,
                            'status': 'error',
                            'message': f'Завершился с кодом {process.returncode}'
                        }, broadcast=True)

                    return jsonify({
                        "success": False,
                        "error": f"Завершился с кодом {process.returncode}",
                        "log": str(log_dir / f"{component}_stderr.log")
                    }), 500

                # WebSocket уведомление об успешном запуске
                if hasattr(self, 'socketio'):
                    from flask_socketio import emit
                    emit('component_status', {
                        'component': component,
                        'status': 'running',
                        'pid': process.pid
                    }, broadcast=True)

                return jsonify({"success": True, "message": f"{component} запущен", "pid": process.pid})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка запуска: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/actions/stop_component", methods=["POST"])
        def api_stop_component_action():
            """API для остановки компонента"""
            try:
                data = request.json
                component = data.get("component", "unknown")

                if not hasattr(self, "_active_processes"):
                    self._active_processes = {}

                if component not in self._active_processes:
                    return jsonify({"success": False, "error": "Не запущен"}), 404

                process = self._active_processes[component]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)

                del self._active_processes[component]
                self.logger.log_system_event(f"Остановка: {component}", "INFO")

                # WebSocket уведомление
                if hasattr(self, 'socketio'):
                    from flask_socketio import emit
                    emit('component_status', {
                        'component': component,
                        'status': 'stopped'
                    }, broadcast=True)

                return jsonify({"success": True, "message": f"{component} остановлен"})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка остановки: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/actions/restart_component", methods=["POST"])
        def api_restart_component_action():
            """API для перезапуска компонента"""
            try:
                data = request.json
                component = data.get("component", "unknown")

                if not hasattr(self, "_active_processes"):
                    self._active_processes = {}

                # Останавливаем если запущен
                if component in self._active_processes:
                    process = self._active_processes[component]
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1)
                    del self._active_processes[component]

                import time
                time.sleep(0.5)

                # Запускаем заново
                component_paths = {
                    "spm_simulator": project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py",
                    "image_analyzer": project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py",
                    "sstv_station": project_root / "components" / "py-sstv-groundstation" / "src" / "main.py",
                }

                if component not in component_paths:
                    return jsonify({"success": False, "error": "Компонент не найден"}), 404

                component_path = component_paths[component]
                if not component_path.exists():
                    return jsonify({"success": False, "error": "Файл не найден"}), 404

                log_dir = project_root / "logs" / "components"
                log_dir.mkdir(parents=True, exist_ok=True)

                with open(log_dir / f"{component}_stdout.log", "ab") as out_f, \
                     open(log_dir / f"{component}_stderr.log", "ab") as err_f:
                    process = subprocess.Popen(
                        [sys.executable, str(component_path)],
                        cwd=str(project_root),
                        stdout=out_f, stderr=err_f,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )

                self._active_processes[component] = process
                self.logger.log_system_event(f"Перезапуск: {component} (PID: {process.pid})", "INFO")

                time.sleep(0.5)
                if process.poll() is not None:
                    # WebSocket уведомление об ошибке
                    if hasattr(self, 'socketio'):
                        from flask_socketio import emit
                        emit('component_status', {
                            'component': component,
                            'status': 'error',
                            'message': f'Не удалось запустить (код {process.returncode})'
                        }, broadcast=True)

                    return jsonify({
                        "success": False,
                        "error": f"Не удалось запустить (код {process.returncode})"
                    }), 500

                # WebSocket уведомление об успешном перезапуске
                if hasattr(self, 'socketio'):
                    from flask_socketio import emit
                    emit('component_status', {
                        'component': component,
                        'status': 'running',
                        'pid': process.pid,
                        'restarted': True
                    }, broadcast=True)

                return jsonify({
                    "success": True,
                    "message": f"{component} перезапущен",
                    "pid": process.pid
                })
            except Exception as e:
                self.error_handler.log_error(f"Ошибка перезапуска: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/config", methods=["GET", "POST"])
        def api_config():
            """API для управления конфигурацией"""
            try:
                if request.method == "GET":
                    config = self.config_manager.get_config()
                    config['fastapi_integration'] = {
                        'enabled': PROXY_AVAILABLE,
                        'url': self.fastapi_url
                    }
                    return jsonify(config)
                elif request.method == "POST":
                    new_config = request.json
                    self.config_manager.update_config(new_config)
                    return jsonify({"status": "success", "message": "Конфигурация обновлена"})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка управления конфигурацией: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/cache_stats")
        def api_cache_stats():
            """API для получения статистики кэша"""
            try:
                cache_stats = self.cache_manager.get_cache_statistics()
                return jsonify(cache_stats)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статистики кэша: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/database/stats")
        def api_database_stats():
            """API для получения статистики базы данных"""
            try:
                db = get_database()
                stats = db.get_statistics()
                return jsonify(stats)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статистики БД: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/integration/health")
        def api_integration_health():
            """Проверка здоровья интеграции"""
            result = {
                "flask": {"status": "healthy", "uptime": self._get_uptime()},
                "fastapi": {"status": "unknown"},
                "proxy": {"status": "enabled" if PROXY_AVAILABLE else "disabled"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Проверка FastAPI
            try:
                response = requests.get(f"{self.fastapi_url}/health", timeout=5)
                if response.status_code == 200:
                    result["fastapi"] = {"status": "healthy"}
                else:
                    result["fastapi"] = {"status": "unhealthy"}
            except requests.RequestException:
                result["fastapi"] = {"status": "unreachable"}

            return jsonify(result)

        # ==================== Экспорт данных ====================

        @self.app.route("/api/export/data", methods=["POST"])
        @login_required
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
                return jsonify({'error': str(e)}), 500

        # ==================== Локальные операции с БД ====================

        @self.app.route("/api/database/scans", methods=["GET"])
        def api_database_scans():
            """API для получения списка сканирований (локально)"""
            try:
                db = get_database()
                scan_type = request.args.get('type')
                limit = int(request.args.get('limit', 50))
                offset = int(request.args.get('offset', 0))

                scans = db.get_scan_results(scan_type=scan_type, limit=limit, offset=offset)
                return jsonify({"scans": scans, "count": len(scans)})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения сканирований: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/database/simulations", methods=["GET"])
        def api_database_simulations():
            """API для получения списка симуляций (локально)"""
            try:
                db = get_database()
                status = request.args.get('status')
                limit = int(request.args.get('limit', 50))

                simulations = db.get_simulations(status=status, limit=limit)
                return jsonify({"simulations": simulations, "count": len(simulations)})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения симуляций: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/surface/compare", methods=["POST"])
        def api_surface_compare():
            """API для сравнения поверхностей (локально)"""
            try:
                data = request.json
                image1_path = data.get('image1_path')
                image2_path = data.get('image2_path')
                output_dir = data.get('output_dir', 'output/surface_comparisons')

                if not image1_path or not image2_path:
                    return jsonify({"error": "image1_path и image2_path обязательны"}), 400

                result = compare_surfaces_util(image1_path, image2_path, output_dir)
                return jsonify({"status": "success", "result": result})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка сравнения поверхностей: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/defect/analyze", methods=["POST"])
        def api_defect_analyze():
            """API для анализа дефектов (локально)"""
            try:
                data = request.json
                image_path = data.get('image_path')
                model_name = data.get('model_name', 'isolation_forest')
                output_dir = data.get('output_dir', 'output/defect_analysis')

                if not image_path:
                    return jsonify({"error": "image_path обязателен"}), 400

                result = analyze_defects_util(image_path, model_name, output_dir)
                return jsonify({"status": "success", "result": result})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка анализа дефектов: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/batch/jobs", methods=["GET"])
        def api_batch_jobs():
            """API для получения заданий пакетной обработки"""
            try:
                from utils.batch.batch_processor import BatchProcessor
                processor = BatchProcessor()
                status_filter = request.args.get('status', None)
                jobs = processor.get_all_jobs(status=status_filter)
                return jsonify({"jobs": jobs, "total": len(jobs)})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения заданий: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/batch/statistics", methods=["GET"])
        def api_batch_statistics():
            """API для получения статистики пакетной обработки"""
            try:
                from utils.batch.batch_processor import BatchProcessor
                processor = BatchProcessor()
                stats = processor.get_statistics()
                return jsonify(stats)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статистики: {e}")
                return jsonify({'error': str(e)}), 500

    def _register_socket_handlers(self):
        """Регистрация обработчиков SocketIO событий"""

        @self.socketio.on("connect")
        def handle_connect():
            """Обработка подключения клиента"""
            self.logger.log_system_event("Клиент подключен к веб-панели", "INFO")
            emit("connection_response", {"data": "Connected to Nanoprobe Simulation Lab Dashboard"})
            emit("system_status", {
                "status": "online",
                "uptime": self._get_uptime(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fastapi_integrated": PROXY_AVAILABLE
            })

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Обработка отключения клиента"""
            self.logger.log_system_event("Клиент отключен от веб-панели", "INFO")

        @self.socketio.on("request_metrics")
        def handle_request_metrics():
            """Запрос метрик системы в realtime"""
            try:
                import os, platform
                import psutil
                from utils.platform_utils import get_system_disk_usage
                disk_usage = get_system_disk_usage()
                metrics = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': disk_usage.percent if disk_usage else 0,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                emit('metrics', metrics)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения метрик: {e}")

        @self.socketio.on("check_integration_health")
        def handle_check_integration_health():
            """Проверка здоровья интеграции через WebSocket"""
            health = {
                "flask": "online",
                "fastapi": "unknown",
                "proxy": PROXY_AVAILABLE
            }

            try:
                response = requests.get(f"{self.fastapi_url}/health", timeout=5)
                if response.status_code == 200:
                    health["fastapi"] = "online"
                else:
                    health["fastapi"] = "error"
            except Exception:
                health["fastapi"] = "offline"

            emit('integration_health', health)

    def _get_uptime(self) -> str:
        """Получение времени работы системы"""
        start_time = getattr(self, "_start_time", datetime.now(timezone.utc))
        uptime = datetime.now(timezone.utc) - start_time
        return str(uptime).split(".")[0]

    def run(self, debug: bool = False, open_browser: bool = True):
        """
        Запуск веб-панели

        Args:
            debug: Режим отладки
            open_browser: Открыть браузер автоматически
        """
        print(f"\n{'='*60}")
        print("  NANOPROBE SIMULATION LAB - ВЕБ-ПАНЕЛЬ")
        print(f"  Интеграция с FastAPI: {'✓ Включена' if PROXY_AVAILABLE else '✗ Отключена'}")
        print(f"  FastAPI URL: {self.fastapi_url}")
        print(f"  Flask URL: http://{self.host}:{self.port}")
        print(f"{'='*60}\n")

        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(f"http://{self.host}:{self.port}")).start()

        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False
        )


def main():
    """Точка входа для запуска веб-панели"""
    import argparse

    parser = argparse.ArgumentParser(description='Веб-панель Nanoprobe Simulation Lab')
    parser.add_argument('--host', default='127.0.0.1', help='Хост сервера')
    parser.add_argument('--port', type=int, default=5000, help='Порт сервера')
    parser.add_argument('--debug', action='store_true', help='Режим отладки')
    parser.add_argument('--no-browser', action='store_true', help='Не открывать браузер')
    parser.add_argument('--fastapi-url', default='http://localhost:8000', help='URL FastAPI')

    args = parser.parse_args()

    # Установка переменной окружения для FastAPI URL
    os.environ['FASTAPI_URL'] = args.fastapi_url

    dashboard = IntegratedWebDashboard(host=args.host, port=args.port)
    dashboard.run(debug=args.debug, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
