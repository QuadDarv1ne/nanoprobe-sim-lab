# Консолидация и Унификация Проекта

## Обзор проблем

### Текущие проблемы:
1. **Два dashboard файла**: `dashboard.py` (559 строк) и `enhanced_dashboard.py` (470 строк)
2. **Три entry points**: `start.py`, `start_all.py`, `start_universal.py`
3. **43+ модулей в utils/** без чёткой организации
4. **Дублирование кода** между Flask и Next.js версиями

---

# Часть 1: Консолидация Dashboard

## Анализ текущих файлов

### dashboard.py (559 строк)
- CLI dashboard
- Базовый функционал
- System monitoring
- Component management

### enhanced_dashboard.py (470 строк)
- Расширенный функционал
- Дополнительные виджеты
- Real-time updates
- Advanced visualizations

## Единая архитектура Dashboard

```
src/cli/
├── dashboard/
│   ├── __init__.py
│   ├── main.py              # Точка входа
│   ├── core.py              # Основной класс Dashboard
│   ├── widgets/             # Виджеты
│   │   ├── __init__.py
│   │   ├── system_monitor.py
│   │   ├── component_manager.py
│   │   ├── log_viewer.py
│   │   ├── metrics.py
│   │   ├── nasa_widget.py   # NASA API widget
│   │   └── sstv_widget.py   # SSTV widget
│   ├── layouts/             # Раскладки
│   │   ├── __init__.py
│   │   ├── standard.py
│   │   ├── enhanced.py
│   │   └── minimal.py
│   └── themes/              # Темы
│       ├── __init__.py
│       ├── dark.py
│       └── light.py
└── dashboard.py             # Точка входа (устаревший, ссылка на main.py)
```

## Унифицированный Dashboard

```python
# src/cli/dashboard/core.py
"""
Unified Dashboard Core

Объединяет функционал dashboard.py и enhanced_dashboard.py
в единую модульную архитектуру.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DashboardMode(Enum):
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MINIMAL = "minimal"

class WidgetPriority(Enum):
    CRITICAL = 0   # Always visible
    HIGH = 1       # Visible in standard+
    NORMAL = 2     # Visible in enhanced
    LOW = 3        # Optional

@dataclass
class Widget:
    """Базовый класс для виджетов dashboard"""
    name: str
    title: str
    priority: WidgetPriority = WidgetPriority.NORMAL
    refresh_interval: int = 5  # seconds
    enabled: bool = True
    position: tuple = (0, 0)  # (row, col)
    size: tuple = (1, 1)      # (height, width)

    # Callbacks
    on_refresh: Optional[Callable] = None
    on_click: Optional[Callable] = None

    # State
    last_update: Optional[datetime] = None
    data: Any = None
    error: Optional[str] = None

    async def refresh(self) -> Any:
        """Обновить данные виджета"""
        if self.on_refresh:
            try:
                self.data = await self.on_refresh()
                self.last_update = datetime.now()
                self.error = None
            except Exception as e:
                self.error = str(e)
                logger.error(f"Widget {self.name} refresh error: {e}")
        return self.data

    def is_visible(self, mode: DashboardMode) -> bool:
        """Проверить видимость в текущем режиме"""
        if not self.enabled:
            return False

        if mode == DashboardMode.MINIMAL:
            return self.priority == WidgetPriority.CRITICAL
        elif mode == DashboardMode.STANDARD:
            return self.priority.value <= WidgetPriority.HIGH.value
        else:  # ENHANCED
            return True


class UnifiedDashboard:
    """
    Единый Dashboard с поддержкой разных режимов отображения.

    Features:
    - Modular widget system
    - Multiple display modes
    - Real-time updates
    - Keyboard navigation
    - Extensible architecture
    """

    def __init__(
        self,
        mode: DashboardMode = DashboardMode.ENHANCED,
        theme: str = "dark",
        refresh_interval: int = 5
    ):
        self.mode = mode
        self.theme = theme
        self.refresh_interval = refresh_interval
        self.widgets: Dict[str, Widget] = {}
        self.running = False
        self._refresh_task: Optional[asyncio.Task] = None

        # Initialize core widgets
        self._init_core_widgets()

    def _init_core_widgets(self):
        """Инициализация основных виджетов"""

        # System Monitor (CRITICAL)
        self.register_widget(Widget(
            name="system_monitor",
            title="System Monitor",
            priority=WidgetPriority.CRITICAL,
            refresh_interval=5,
            position=(0, 0),
            size=(3, 4),
            on_refresh=self._get_system_metrics
        ))

        # Component Status (CRITICAL)
        self.register_widget(Widget(
            name="component_status",
            title="Components",
            priority=WidgetPriority.CRITICAL,
            refresh_interval=10,
            position=(0, 4),
            size=(3, 2),
            on_refresh=self._get_component_status
        ))

        # Log Viewer (HIGH)
        self.register_widget(Widget(
            name="log_viewer",
            title="Recent Logs",
            priority=WidgetPriority.HIGH,
            refresh_interval=2,
            position=(3, 0),
            size=(2, 4),
            on_refresh=self._get_recent_logs
        ))

        # Quick Actions (HIGH)
        self.register_widget(Widget(
            name="quick_actions",
            title="Quick Actions",
            priority=WidgetPriority.HIGH,
            position=(3, 4),
            size=(2, 2)
        ))

        # NASA APOD (NORMAL)
        self.register_widget(Widget(
            name="nasa_apod",
            title="NASA APOD",
            priority=WidgetPriority.NORMAL,
            refresh_interval=3600,  # 1 hour
            position=(5, 0),
            size=(2, 3),
            on_refresh=self._get_nasa_apod
        ))

        # ISS Position (NORMAL)
        self.register_widget(Widget(
            name="iss_position",
            title="ISS Position",
            priority=WidgetPriority.NORMAL,
            refresh_interval=30,
            position=(5, 3),
            size=(2, 3),
            on_refresh=self._get_iss_position
        ))

        # Performance Metrics (NORMAL)
        self.register_widget(Widget(
            name="performance",
            title="Performance",
            priority=WidgetPriority.NORMAL,
            refresh_interval=10,
            position=(7, 0),
            size=(2, 6),
            on_refresh=self._get_performance_metrics
        ))

    def register_widget(self, widget: Widget):
        """Зарегистрировать виджет"""
        self.widgets[widget.name] = widget

    def unregister_widget(self, name: str):
        """Удалить виджет"""
        if name in self.widgets:
            del self.widgets[name]

    def set_mode(self, mode: DashboardMode):
        """Изменить режим отображения"""
        self.mode = mode
        logger.info(f"Dashboard mode changed to: {mode.value}")

    def get_visible_widgets(self) -> List[Widget]:
        """Получить видимые виджеты для текущего режима"""
        visible = [w for w in self.widgets.values() if w.is_visible(self.mode)]
        return sorted(visible, key=lambda w: w.priority.value)

    async def refresh_all(self):
        """Обновить все видимые виджеты"""
        tasks = []
        for widget in self.get_visible_widgets():
            if widget.on_refresh:
                tasks.append(widget.refresh())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self):
        """Запустить dashboard с auto-refresh"""
        self.running = True

        async def refresh_loop():
            while self.running:
                await self.refresh_all()
                self.render()
                await asyncio.sleep(self.refresh_interval)

        self._refresh_task = asyncio.create_task(refresh_loop())

    async def stop(self):
        """Остановить dashboard"""
        self.running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

    def render(self):
        """Отрендерить dashboard"""
        # Clear screen
        print("\033[2J\033[H", end="")

        # Header
        print(self._render_header())

        # Widgets
        visible = self.get_visible_widgets()

        # Group by rows
        rows: Dict[int, List[Widget]] = {}
        for widget in visible:
            row = widget.position[0]
            if row not in rows:
                rows[row] = []
            rows[row].append(widget)

        # Render each row
        for row_num in sorted(rows.keys()):
            row_widgets = sorted(rows[row_num], key=lambda w: w.position[1])
            print(self._render_row(row_widgets))

        # Footer
        print(self._render_footer())

    def _render_header(self) -> str:
        """Рендер заголовка"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🔬 Nanoprobe Sim Lab Dashboard    │ Mode: {self.mode.value:10} │ {now}     ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""

    def _render_row(self, widgets: List[Widget]) -> str:
        """Рендер строки виджетов"""
        lines = []
        for widget in widgets:
            lines.append(f"┌─ {widget.title} {'─' * (30 - len(widget.title))}┐")
            lines.append(self._render_widget_content(widget))
            lines.append(f"└{'─' * 36}┘")
        return "\n".join(lines)

    def _render_widget_content(self, widget: Widget) -> str:
        """Рендер содержимого виджета"""
        if widget.error:
            return f"│ ⚠ Error: {widget.error[:28]}{'.' * (28 - len(widget.error))} │"
        elif widget.data:
            return f"│ {str(widget.data)[:34]}{' ' * (34 - len(str(widget.data)))} │"
        else:
            return f"│ {'Loading...':^34} │"

    def _render_footer(self) -> str:
        """Рендер футера"""
        return """
╠══════════════════════════════════════════════════════════════════════════════╣
║  [Q] Quit  [R] Refresh  [M] Mode  [H] Help  [1-9] Select Widget              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

    # Widget data providers
    async def _get_system_metrics(self) -> Dict:
        """Получить метрики системы"""
        import psutil
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }

    async def _get_component_status(self) -> Dict:
        """Получить статус компонентов"""
        return {
            "spm": "active",
            "analyzer": "idle",
            "sstv": "standby"
        }

    async def _get_recent_logs(self) -> List[str]:
        """Получить последние логи"""
        # Read from log file
        return ["[INFO] System started", "[INFO] API ready"]

    async def _get_nasa_apod(self) -> Dict:
        """Получить NASA APOD"""
        try:
            from utils.nasa_api_client import get_nasa_client
            client = await get_nasa_client()
            return await client.get_apod()
        except Exception as e:
            return {"error": str(e)}

    async def _get_iss_position(self) -> Dict:
        """Получить позицию МКС"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://api.open-notify.org/iss-now.json") as resp:
                    return await resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def _get_performance_metrics(self) -> Dict:
        """Получить метрики производительности"""
        return {
            "requests_per_min": 42,
            "avg_response_time": 0.125,
            "cache_hit_rate": 0.85
        }


# Entry point
def run_dashboard(mode: str = "enhanced"):
    """Запуск dashboard"""
    mode_enum = DashboardMode(mode.lower())
    dashboard = UnifiedDashboard(mode=mode_enum)

    try:
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "enhanced"
    run_dashboard(mode)
```

---

# Часть 2: Унификация Entry Points

## Текущие entry points

### start.py
```python
# Главная точка входа
# Поддержка CLI, web, manager
```

### start_all.py
```python
# Синхронизированный запуск Backend + Frontend
# Sync Manager
```

### start_universal.py
```python
# Универсальный лаунчер
# Flask/Next.js выбор
```

## Единый main.py

```python
# main.py
"""
Unified Entry Point for Nanoprobe Sim Lab

Комплексная точка входа, объединяющая все режимы запуска:
- CLI dashboard
- Web interfaces (Flask/Next.js)
- API server
- Background workers
- Development mode

Usage:
    python main.py                    # Interactive mode selection
    python main.py cli                # CLI dashboard
    python main.py web flask          # Flask web interface
    python main.py web nextjs         # Next.js web interface
    python main.py api                # API only
    python main.py all                # Full stack (API + Web)
    python main.py dev                # Development mode
    python main.py worker             # Background worker
"""

import sys
import os
import asyncio
import argparse
import subprocess
import signal
from typing import Optional, List
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerMode(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Service(Enum):
    API = "api"
    FLASK = "flask"
    NEXTJS = "nextjs"
    CLI = "cli"
    WORKER = "worker"
    ALL = "all"


class ProcessManager:
    """Управление процессами сервисов"""

    def __init__(self):
        self.processes: dict = {}
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.stop_all()

    def start_service(
        self,
        service: Service,
        mode: ServerMode = ServerMode.DEVELOPMENT,
        port: Optional[int] = None
    ) -> subprocess.Popen:
        """Запуск сервиса"""

        if service == Service.API:
            return self._start_api(mode, port or 8000)
        elif service == Service.FLASK:
            return self._start_flask(mode, port or 5000)
        elif service == Service.NEXTJS:
            return self._start_nextjs(mode, port or 3000)
        elif service == Service.CLI:
            return self._start_cli()
        elif service == Service.WORKER:
            return self._start_worker()
        else:
            raise ValueError(f"Unknown service: {service}")

    def _start_api(self, mode: ServerMode, port: int) -> subprocess.Popen:
        """Запуск FastAPI"""
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", str(port)
        ]

        if mode == ServerMode.DEVELOPMENT:
            cmd.extend(["--reload"])

        logger.info(f"Starting API server on port {port}")
        proc = subprocess.Popen(cmd)
        self.processes[Service.API] = proc
        return proc

    def _start_flask(self, mode: ServerMode, port: int) -> subprocess.Popen:
        """Запуск Flask web interface"""
        env = os.environ.copy()
        env["FLASK_APP"] = "src/web/web_dashboard.py"
        env["FLASK_ENV"] = "development" if mode == ServerMode.DEVELOPMENT else "production"

        cmd = [sys.executable, "-m", "flask", "run", "--port", str(port)]

        logger.info(f"Starting Flask server on port {port}")
        proc = subprocess.Popen(cmd, env=env)
        self.processes[Service.FLASK] = proc
        return proc

    def _start_nextjs(self, mode: ServerMode, port: int) -> subprocess.Popen:
        """Запуск Next.js frontend"""
        frontend_dir = Path(__file__).parent / "frontend"

        if mode == ServerMode.DEVELOPMENT:
            cmd = ["npm", "run", "dev"]
        else:
            cmd = ["npm", "run", "start"]

        logger.info(f"Starting Next.js server on port {port}")
        proc = subprocess.Popen(cmd, cwd=frontend_dir)
        self.processes[Service.NEXTJS] = proc
        return proc

    def _start_cli(self) -> subprocess.Popen:
        """Запуск CLI dashboard"""
        from src.cli.dashboard.core import run_dashboard
        logger.info("Starting CLI dashboard")
        # CLI runs in foreground
        run_dashboard("enhanced")
        return None

    def _start_worker(self) -> subprocess.Popen:
        """Запуск background worker"""
        cmd = [sys.executable, "-m", "api.worker"]
        logger.info("Starting background worker")
        proc = subprocess.Popen(cmd)
        self.processes[Service.WORKER] = proc
        return proc

    def stop_service(self, service: Service):
        """Остановить сервис"""
        if service in self.processes:
            proc = self.processes[service]
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                logger.info(f"Stopped {service.value}")

    def stop_all(self):
        """Остановить все сервисы"""
        for service in list(self.processes.keys()):
            self.stop_service(service)

    def wait(self):
        """Ожидание завершения всех процессов"""
        while self.running:
            # Check if any process died
            for service, proc in list(self.processes.items()):
                if proc and proc.poll() is not None:
                    logger.warning(f"{service.value} process died, restarting...")
                    self.start_service(service)

            import time
            time.sleep(1)


def interactive_mode():
    """Интерактивный выбор режима"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔬 Nanoprobe Sim Lab - Control Panel                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  [1] CLI Dashboard         - Interactive command-line dashboard              ║
║  [2] Flask Web Interface   - Legacy web dashboard (port 5000)                ║
║  [3] Next.js Web Interface - Modern web dashboard (port 3000)                ║
║  [4] API Server Only       - FastAPI backend (port 8000)                     ║
║  [5] Full Stack (Dev)      - API + Next.js + Hot Reload                      ║
║  [6] Full Stack (Prod)     - Production mode all services                    ║
║  [7] Background Worker     - Run async tasks                                 ║
║  [8] Development Mode      - All services with debugging                     ║
║                                                                              ║
║  [Q] Quit                                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    choice = input("Select option [1-8, Q]: ").strip().upper()

    manager = ProcessManager()

    if choice == "1":
        manager.start_service(Service.CLI)
    elif choice == "2":
        manager.start_service(Service.FLASK, ServerMode.DEVELOPMENT)
        manager.wait()
    elif choice == "3":
        manager.start_service(Service.NEXTJS, ServerMode.DEVELOPMENT)
        manager.wait()
    elif choice == "4":
        manager.start_service(Service.API, ServerMode.DEVELOPMENT)
        manager.wait()
    elif choice == "5":
        manager.start_service(Service.API, ServerMode.DEVELOPMENT, 8000)
        manager.start_service(Service.NEXTJS, ServerMode.DEVELOPMENT, 3000)
        manager.wait()
    elif choice == "6":
        manager.start_service(Service.API, ServerMode.PRODUCTION, 8000)
        manager.start_service(Service.NEXTJS, ServerMode.PRODUCTION, 3000)
        manager.wait()
    elif choice == "7":
        manager.start_service(Service.WORKER)
        manager.wait()
    elif choice == "8":
        manager.start_service(Service.API, ServerMode.DEVELOPMENT, 8000)
        manager.start_service(Service.NEXTJS, ServerMode.DEVELOPMENT, 3000)
        manager.start_service(Service.WORKER)
        manager.wait()
    elif choice == "Q":
        print("Goodbye!")
    else:
        print("Invalid option")


def main():
    """Главная точка входа"""
    parser = argparse.ArgumentParser(
        description="Nanoprobe Sim Lab - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Interactive mode
    python main.py cli                # CLI dashboard
    python main.py api --port 8080    # API on port 8080
    python main.py all --mode dev     # Full stack development
    python main.py web nextjs         # Next.js frontend only
        """
    )

    # Positional arguments
    parser.add_argument(
        "command",
        nargs="?",
        choices=["cli", "web", "api", "all", "worker", "dev"],
        help="Service to run"
    )

    # Subcommand for web
    parser.add_argument(
        "web_type",
        nargs="?",
        choices=["flask", "nextjs"],
        help="Web framework (for 'web' command)"
    )

    # Options
    parser.add_argument(
        "--mode", "-m",
        choices=["dev", "development", "prod", "production"],
        default="development",
        help="Server mode"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port number"
    )

    parser.add_argument(
        "--dashboard-mode",
        choices=["standard", "enhanced", "minimal"],
        default="enhanced",
        help="Dashboard display mode (for CLI)"
    )

    args = parser.parse_args()

    # Interactive mode if no command
    if not args.command:
        interactive_mode()
        return

    # Determine server mode
    mode = ServerMode.PRODUCTION if args.mode in ["prod", "production"] else ServerMode.DEVELOPMENT

    manager = ProcessManager()

    # Execute command
    if args.command == "cli":
        from src.cli.dashboard.core import run_dashboard
        run_dashboard(args.dashboard_mode)

    elif args.command == "web":
        if args.web_type == "flask":
            manager.start_service(Service.FLASK, mode, args.port)
        else:
            manager.start_service(Service.NEXTJS, mode, args.port)
        manager.wait()

    elif args.command == "api":
        manager.start_service(Service.API, mode, args.port)
        manager.wait()

    elif args.command == "all":
        manager.start_service(Service.API, mode, args.port or 8000)
        manager.start_service(Service.NEXTJS, mode, args.port or 3000)
        manager.wait()

    elif args.command == "worker":
        manager.start_service(Service.WORKER)
        manager.wait()

    elif args.command == "dev":
        manager.start_service(Service.API, ServerMode.DEVELOPMENT, 8000)
        manager.start_service(Service.NEXTJS, ServerMode.DEVELOPMENT, 3000)
        manager.start_service(Service.WORKER)
        manager.wait()


if __name__ == "__main__":
    main()
```

## Миграция с устаревших entry points

```python
# start.py (deprecated - redirects to main.py)
"""
DEPRECATED: Use main.py instead

This file is kept for backward compatibility.
It will be removed in a future version.
"""

import sys
import warnings

warnings.warn(
    "start.py is deprecated. Use 'python main.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect to main.py
if __name__ == "__main__":
    import main
    sys.argv[0] = "main.py"
    main.main()
```

---

# Часть 3: Реорганизация utils/

## Текущая структура (43+ модулей)

```
utils/
├── config_manager.py
├── logger.py
├── data_manager.py
├── cache_manager.py
├── system_monitor.py
├── error_handler.py
├── performance_profiler.py
├── resource_optimizer.py
├── advanced_logger_analyzer.py
├── memory_tracker.py
├── performance_benchmark.py
├── optimization_orchestrator.py
├── system_health_monitor.py
├── performance_analytics_dashboard.py
├── performance_verification_framework.py
├── optimization_config_manager.py
├── optimization_logging_manager.py
├── realtime_dashboard.py
├── performance_monitoring_center.py
├── predictive_analytics_engine.py
├── automated_optimization_scheduler.py
├── ai_resource_optimizer.py
├── self_healing_system.py
├── ... (20+ more)
```

## Новая структура

```
utils/
├── __init__.py                    # Публичный API
│
├── core/                          # Основные утилиты
│   ├── __init__.py
│   ├── config.py                  # Config management
│   ├── logging.py                 # Logging system
│   ├── errors.py                  # Error handling
│   └── cache.py                   # Caching
│
├── monitoring/                    # Мониторинг
│   ├── __init__.py
│   ├── system.py                  # System monitor
│   ├── health.py                  # Health checks
│   ├── metrics.py                 # Metrics collection
│   └── alerts.py                  # Alerting
│
├── performance/                   # Производительность
│   ├── __init__.py
│   ├── profiler.py                # Profiling
│   ├── benchmark.py               # Benchmarking
│   ├── optimizer.py               # Optimization
│   ├── memory.py                  # Memory tracking
│   └── analytics.py               # Performance analytics
│
├── optimization/                  # Автооптимизация
│   ├── __init__.py
│   ├── orchestrator.py            # Orchestration
│   ├── scheduler.py               # Scheduling
│   ├── ai_optimizer.py            # AI-based optimization
│   └── self_healing.py            # Self-healing system
│
├── security/                      # Безопасность
│   ├── __init__.py
│   ├── auth.py                    # Authentication
│   ├── jwt.py                     # JWT handling
│   ├── rate_limit.py              # Rate limiting
│   └── encryption.py              # Encryption
│
├── data/                          # Работа с данными
│   ├── __init__.py
│   ├── manager.py                 # Data management
│   ├── validation.py              # Validation
│   ├── migration.py               # Migrations
│   └── backup.py                  # Backup
│
├── api/                           # API утилиты
│   ├── __init__.py
│   ├── nasa_client.py             # NASA API client
│   ├── iss_tracker.py             # ISS tracking
│   └── external.py                # External APIs
│
└── dev/                           # Development tools
    ├── __init__.py
    ├── code_analyzer.py           # Code analysis
    ├── docs_generator.py          # Documentation
    └── testing.py                 # Testing utilities
```

## __init__.py с публичным API

```python
# utils/__init__.py
"""
Nanoprobe Sim Lab Utilities

Публичный API для доступа ко всем утилитам проекта.
"""

# Core
from utils.core.config import ConfigManager, get_config
from utils.core.logging import setup_logging, get_logger
from utils.core.errors import APIError, ValidationError, NotFoundError
from utils.core.cache import CacheManager, get_cache

# Monitoring
from utils.monitoring.system import SystemMonitor
from utils.monitoring.health import HealthChecker
from utils.monitoring.metrics import MetricsCollector

# Performance
from utils.performance.profiler import PerformanceProfiler
from utils.performance.benchmark import run_benchmark
from utils.performance.memory import MemoryTracker

# Security
from utils.security.auth import AuthManager
from utils.security.rate_limit import RateLimiter

# API
from utils.api.nasa_client import NASAAPIClient

# Convenience aliases
config = get_config
logger = get_logger
cache = get_cache

__all__ = [
    # Core
    "ConfigManager",
    "get_config",
    "setup_logging",
    "get_logger",
    "APIError",
    "ValidationError",
    "NotFoundError",
    "CacheManager",
    "get_cache",

    # Monitoring
    "SystemMonitor",
    "HealthChecker",
    "MetricsCollector",

    # Performance
    "PerformanceProfiler",
    "run_benchmark",
    "MemoryTracker",

    # Security
    "AuthManager",
    "RateLimiter",

    # API
    "NASAAPIClient",

    # Aliases
    "config",
    "logger",
    "cache",
]
```

## Пример использования

```python
# Старый способ (deprecated)
from utils.config_manager import ConfigManager
from utils.logger import Logger
from utils.system_monitor import SystemMonitor

# Новый способ
from utils import config, logger, SystemMonitor

# Или явно
from utils.core import ConfigManager
from utils.monitoring import SystemMonitor
```

---

# Часть 4: План миграции

## Этапы миграции

### Этап 1: Создание новой структуры (Week 1)
- [ ] Создать новые директории
- [ ] Создать __init__.py файлы
- [ ] Переместить файлы в новые директории
- [ ] Обновить импорты

### Этап 2: Обновление импортов (Week 2)
- [ ] Найти все импорты через grep
- [ ] Обновить импорты во всех файлах
- [ ] Запустить тесты
- [ ] Исправить ошибки

### Этап 3: Удаление дубликатов (Week 3)
- [ ] Сравнить dashboard.py и enhanced_dashboard.py
- [ ] Создать unified dashboard
- [ ] Удалить старые файлы
- [ ] Обновить документацию

### Этап 4: Тестирование (Week 4)
- [ ] Запустить все тесты
- [ ] Проверить все entry points
- [ ] Smoke testing всех функций
- [ ] Performance testing

## Команды для миграции

```bash
# Поиск всех импортов из utils
grep -r "from utils\." --include="*.py" > imports.txt

# Создание новых директорий
mkdir -p utils/{core,monitoring,performance,optimization,security,data,api,dev}

# Перемещение файлов (пример)
mv utils/config_manager.py utils/core/config.py
mv utils/logger.py utils/core/logging.py
mv utils/system_monitor.py utils/monitoring/system.py

# Обновление импортов (sed)
find . -name "*.py" -exec sed -i 's/from utils\.config_manager/from utils.core.config/g' {} \;
```
