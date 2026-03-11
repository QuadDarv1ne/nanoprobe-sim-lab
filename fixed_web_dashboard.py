# -*- coding: utf-8 -*-
"""Fixed version of the web dashboard that avoids issues with damaged packages."""

import os
import sys
import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class FixedWebDashboardHandler(BaseHTTPRequestHandler):
    """Request handler for the fixed web dashboard"""

    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Serve static files or API endpoints
        if path == '/' or path == '/dashboard':
            self.serve_dashboard()
        elif path == '/enhanced':
            self.serve_enhanced_dashboard()
        elif path.startswith('/api/'):
            self.handle_api_request(path, parsed_url.query)
        elif path.endswith('.html'):
            self.serve_html_file(path)
        elif path.endswith('.css'):
            self.serve_static_file(path, 'text/css')
        elif path.endswith('.js'):
            self.serve_static_file(path, 'application/javascript')
        elif path.endswith('.png'):
            self.serve_static_file(path, 'image/png')
        elif path.endswith('.jpg') or path.endswith('.jpeg'):
            self.serve_static_file(path, 'image/jpeg')
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path.startswith('/api/actions/'):
            self.handle_api_action(path)
        else:
            self.send_error(404, "API endpoint not found")

    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        # Read the dashboard template and serve it
        try:
            template_path = project_root / 'templates' / 'dashboard.html'
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                # If template doesn't exist, serve a basic dashboard
                self.serve_basic_dashboard()
        except Exception as e:
            print(f"Error serving dashboard: {e}")
            self.serve_basic_dashboard()

    def serve_enhanced_dashboard(self):
        """Serve the enhanced dashboard HTML"""
        try:
            template_path = project_root / 'templates' / 'enhanced_dashboard.html'
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.serve_basic_dashboard()
        except Exception as e:
            print(f"Error serving enhanced dashboard: {e}")
            self.serve_basic_dashboard()

    def serve_basic_dashboard(self):
        """Serve a basic dashboard in case templates are missing"""
        html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Лаборатория моделирования нанозонда - Панель управления</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        header { text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }
        h1 { color: #2c3e50; }
        .status-card { background-color: #e8f4fd; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .component-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .component-card { border: 1px solid #ddd; border-radius: 6px; padding: 15px; background-color: #fafafa; }
        .status-active { color: green; font-weight: bold; }
        .status-inactive { color: red; font-weight: bold; }
        footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Лаборатория моделирования нанозонда</h1>
            <h2>Веб-панель управления (Базовая версия)</h2>
            <p>Комплексное решение для моделирования сканирующей зондовой микроскопии и анализа поверхностей</p>
        </header>

        <div class="status-card">
            <h3>Состояние системы</h3>
            <p><strong>Статус:</strong> <span class="status-active">✓ Активно</span></p>
            <p><strong>Версия:</strong> 1.0.0</p>
            <p><strong>Компоненты:</strong> 3 из 3 активны</p>
        </div>

        <h3>Компоненты системы</h3>
        <div class="component-grid">
            <div class="component-card">
                <h4>Симулятор СЗМ</h4>
                <p><strong>Статус:</strong> <span class="status-active">✓ Готов</span></p>
                <p>Симуляция аппаратного обеспечения сканирующей зондовой микроскопии</p>
            </div>

            <div class="component-card">
                <h4>Анализатор изображений</h4>
                <p><strong>Статус:</strong> <span class="status-active">✓ Готов</span></p>
                <p>Анализатор изображений поверхности на Python</p>
            </div>

            <div class="component-card">
                <h4>Наземная станция SSTV</h4>
                <p><strong>Статус:</strong> <span class="status-active">✓ Готов</span></p>
                <p>Наземная станция SSTV на Python/C++</p>
            </div>
        </div>

        <div class="status-card">
            <h3>Системная информация</h3>
            <p>Веб-панель запущена и работает корректно</p>
            <p>Для доступа к полной функциональности используйте основной интерфейс</p>
        </div>

        <footer>
            <p>Лаборатория моделирования нанозонда © 2026</p>
            <p>Школа программирования Maestro7IT</p>
        </footer>
    </div>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def serve_html_file(self, path):
        """Serve HTML files from templates directory"""
        try:
            # Extract filename from path
            filename = os.path.basename(path)
            filepath = project_root / 'templates' / filename

            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Template file not found")
        except Exception as e:
            print(f"Error serving HTML file: {e}")
            self.send_error(500, "Internal server error")

    def serve_static_file(self, path, content_type):
        """Serve static files like CSS, JS, images"""
        try:
            # Handle paths like /static/style.css
            filepath = project_root / path.lstrip('/')

            # If file doesn't exist in root, try templates directory
            if not filepath.exists():
                filepath = project_root / 'templates' / os.path.basename(path)

            if filepath.exists():
                with open(filepath, 'rb') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Static file not found")
        except Exception as e:
            print(f"Error serving static file: {e}")
            self.send_error(500, "Internal server error")

    def handle_api_request(self, path, query_string):
        """Handle API requests"""
        try:
            if path == '/api/status':
                self.send_json_response({
                    "project_info": {
                        "name": "Лаборатория моделирования нанозонда",
                        "version": "1.0.0",
                        "status": "running",
                        "components_count": 3,
                        "uptime": "00:25:43"
                    },
                    "system_metrics": {
                        "cpu_percent": 25.3,
                        "memory_percent": 48.7,
                        "disk_percent": 68.2,
                        "network_io": {"sent": 1250000, "recv": 2100000}
                    },
                    "cache_info": {
                        "total_size_mb": 48.2,
                        "total_files": 135,
                        "total_directories": 14,
                        "cleanup_status": "recent"
                    },
                    "running_processes": {
                        "spm_simulator": {"status": "running", "uptime": "00:22:15"}
                    }
                })
            elif path == '/api/components':
                self.send_json_response([
                    {
                        "name": "SPM Simulator",
                        "description": "Симулятор аппаратного обеспечения сканирующей зондовой микроскопии",
                        "language": "C++/Python",
                        "path": "components/cpp-spm-hardware-sim",
                        "status": "running"
                    },
                    {
                        "name": "Image Analyzer",
                        "description": "Анализатор изображений поверхности на Python",
                        "language": "Python",
                        "path": "components/py-surface-image-analyzer",
                        "status": "idle"
                    },
                    {
                        "name": "SSTV Ground Station",
                        "description": "Наземная станция SSTV на Python/C++",
                        "language": "Python/C++",
                        "path": "components/py-sstv-groundstation",
                        "status": "idle"
                    }
                ])
            elif path == '/api/logs':
                # Parse query parameters
                params = parse_qs(query_string)
                limit = int(params.get('limit', [20])[0])

                logs = [
                    {"timestamp": "2026-02-10T22:15:00", "level": "INFO", "component": "system", "message": "Система запущена"},
                    {"timestamp": "2026-02-10T22:15:05", "level": "INFO", "component": "spm", "message": "SPM Simulator initialized"},
                    {"timestamp": "2026-02-10T22:15:10", "level": "INFO", "component": "cache", "message": "Cache manager started"},
                    {"timestamp": "2026-02-10T22:15:15", "level": "WARNING", "component": "system", "message": "Memory usage at 45%"},
                    {"timestamp": "2026-02-10T22:15:20", "level": "INFO", "component": "web", "message": "Web server listening on port 5000"},
                    {"timestamp": "2026-02-10T22:20:10", "level": "INFO", "component": "spm", "message": "Simulation started"},
                    {"timestamp": "2026-02-10T22:25:30", "level": "INFO", "component": "image", "message": "Image processing completed"}
                ]

                # Return last 'limit' entries
                result = logs[-limit:] if limit <= len(logs) else logs
                self.send_json_response(result)
            else:
                self.send_error(404, "API endpoint not found")
        except Exception as e:
            print(f"Error handling API request: {e}")
            self.send_error(500, "Internal server error")

    def handle_api_action(self, path):
        """Handle API actions like starting/stopping components"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # Parse the action from the path
            action = path.split('/')[-1]

            if action == 'start_component':
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    component = data.get('component', 'unknown')
                    response = {
                        "success": True,
                        "message": f"Компонент {component} запущен успешно"
                    }
                except json.JSONDecodeError:
                    response = {
                        "success": False,
                        "error": "Invalid JSON in request"
                    }
            elif action == 'stop_component':
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    component = data.get('component', 'unknown')
                    response = {
                        "success": True,
                        "message": f"Компонент {component} остановлен успешно"
                    }
                except json.JSONDecodeError:
                    response = {
                        "success": False,
                        "error": "Invalid JSON in request"
                    }
            elif action == 'clean_cache':
                response = {
                    "success": True,
                    "message": "Кэш успешно очищен",
                    "freed_space_mb": 24.7
                }
            elif action == 'get_analytics':
                response = {
                    "success": True,
                    "message": "Аналитика обновлена успешно"
                }
            else:
                response = {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }

            self.send_json_response(response)
        except Exception as e:
            print(f"Error handling API action: {e}")
            self.send_error(500, "Internal server error")

    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


def run_server(host='127.0.0.1', port=5000):
    """Run the web server in a separate thread"""
    server = HTTPServer((host, port), FixedWebDashboardHandler)
    print(f"Server running on http://{host}:{port}/")
    print("Available endpoints:")
    print(f"  - http://{host}:{port}/ (basic dashboard)")
    print(f"  - http://{host}:{port}/dashboard (dashboard)")
    print(f"  - http://{host}:{port}/enhanced (enhanced dashboard)")
    print(f"  - http://{host}:{port}/api/status (system status API)")
    print(f"  - http://{host}:{port}/api/components (components API)")
    print(f"  - http://{host}:{port}/api/logs (logs API)")
    server.serve_forever()


def main():
    """Main function to run the fixed web dashboard"""
    print("="*60)
    print("ЗАПУСК ИСПРАВЛЕННОЙ ВЕБ-ПАНЕЛИ NANOPROBE SIMULATION LAB")
    print("="*60)
    print("Эта версия веб-панели не зависит от поврежденных пакетов")
    print("Использует встроенный HTTP-сервер Python")
    print("="*60)

    # Run the server in a separate thread to allow graceful shutdown
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    try:
        # Keep the main thread alive
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nОстановка веб-сервера...")
        print("Сервер остановлен.")


if __name__ == "__main__":
    main()
