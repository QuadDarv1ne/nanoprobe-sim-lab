#!/usr/bin/env python
"""
Nanoprobe Sim Lab - Universal Launcher
Launches Backend (FastAPI) + Frontend (Flask v1.0 or Next.js v2.0)

Usage:
    python start.py              # Interactive mode
    python start.py flask        # Flask frontend (port 5000)
    python start.py nextjs       # Next.js frontend (port 3000)
    python start.py api-only     # Backend API only (port 8000)

Requirements:
    Python 3.11, 3.12, 3.13, or 3.14
"""

# Проверка версии Python (требуется 3.11 - 3.14)
import sys
MIN_PYTHON_VERSION = (3, 11)
MAX_PYTHON_VERSION = (3, 14)
if sys.version_info < MIN_PYTHON_VERSION or sys.version_info >= (MAX_PYTHON_VERSION[0], MAX_PYTHON_VERSION[1] + 1):
    print(f"[ERROR] Требуется Python 3.11 - 3.14, текущая версия: {sys.version}")
    print(f"Путь к Python: {sys.executable}")
    print("Установите Python 3.11 - 3.14 с https://www.python.org/downloads/")
    sys.exit(1)

import os
import subprocess
import time
import webbrowser
import socket
from pathlib import Path

if sys.platform == "win32":
    os.system("chcp 65001 >nul")

PROJECT_ROOT = Path(__file__).parent
BACKEND_PORT = 8000
FLASK_PORT = 5000
NEXTJS_PORT = 3000


def print_banner():
    print("=" * 70)
    print("  Nanoprobe Sim Lab - Universal Launcher")
    print("=" * 70)
    print()


def print_versions():
    print("Available frontend versions:")
    print()
    print("  1. Flask Dashboard (v1.0)")
    print("     - Port: http://localhost:5000")
    print()
    print("  2. Next.js Dashboard (v2.0)")
    print("     - Port: http://localhost:3000")
    print()
    print("  3. Backend API only")
    print("     - Port: http://localhost:8000")
    print()
    print("=" * 70)
    print()


def check_port(port: int) -> bool:
    """Check if port is in use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except (socket.error, OSError):
        return False


def wait_for_port(port: int, timeout: int = 30) -> bool:
    """Wait for port to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not check_port(port):
            time.sleep(0.5)
        else:
            return True
    return False


def start_backend():
    """Start Backend (FastAPI)"""
    print("Starting Backend (FastAPI)...")
    print(f"   Port: http://localhost:{BACKEND_PORT}")
    print(f"   Swagger: http://localhost:{BACKEND_PORT}/docs")
    print()

    backend_script = PROJECT_ROOT / "run_api.py"
    if not backend_script.exists():
        print(f"[ERROR] {backend_script} not found!")
        return None

    # Check uvicorn
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("[ERROR] uvicorn not installed! Run: pip install uvicorn")
        return None

    process = subprocess.Popen(
        [sys.executable, str(backend_script), "--reload"],
        cwd=str(PROJECT_ROOT)
    )

    if wait_for_port(BACKEND_PORT, timeout=10):
        print("[OK] Backend started!")
        print()
        return process
    else:
        print("[ERROR] Failed to start Backend!")
        return None


def start_flask_frontend():
    """Start Flask frontend"""
    print("Starting Flask Frontend (Unified)...")
    print(f"   Port: http://localhost:{FLASK_PORT}")
    print()

    flask_script = PROJECT_ROOT / "src" / "web" / "web_dashboard_unified.py"
    if not flask_script.exists():
        print(f"[ERROR] {flask_script} not found!")
        return None

    # Check Flask
    try:
        import flask  # noqa: F401
    except ImportError:
        print("[ERROR] Flask not installed! Run: pip install flask")
        return None

    process = subprocess.Popen(
        [sys.executable, str(flask_script)],
        cwd=str(PROJECT_ROOT)
    )

    time.sleep(2)
    print("[OK] Flask Frontend started!")
    print()
    return process


def start_nextjs_frontend():
    """Start Next.js frontend"""
    print("Starting Next.js Frontend...")
    print(f"   Port: http://localhost:{NEXTJS_PORT}")
    print()

    frontend_dir = PROJECT_ROOT / "frontend"
    if not frontend_dir.exists():
        print(f"[ERROR] {frontend_dir} not found!")
        return None

    # Check node_modules
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("[WARN] Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir))

    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir)
    )

    time.sleep(5)
    print("[OK] Next.js Frontend started!")
    print()
    return process


def open_browser(version: str):
    """Open browser"""
    urls = {
        "flask": f"http://localhost:{FLASK_PORT}",
        "nextjs": f"http://localhost:{NEXTJS_PORT}",
        "api-only": f"http://localhost:{BACKEND_PORT}/docs"
    }
    url = urls.get(version, urls["api-only"])
    print(f"Opening browser: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    time.sleep(1)


def interactive_choice():
    """Interactive version selection"""
    print_versions()
    while True:
        choice = input("Select version (1/2/3 or flask/nextjs/api-only): ").strip().lower()
        if choice in ["1", "flask"]:
            return "flask"
        elif choice in ["2", "nextjs"]:
            return "nextjs"
        elif choice in ["3", "api-only", "api"]:
            return "api-only"
        else:
            print("[ERROR] Invalid choice. Try again.")
            print()


def main():
    """Main function"""
    print_banner()

    # Determine launch mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["flask", "nextjs", "api-only", "api"]:
            print(f"[ERROR] Invalid argument: {mode}")
            print("Use: flask, nextjs, or api-only")
            return
    else:
        mode = interactive_choice()

    print()
    print("=" * 70)
    print(f"  Launch mode: {mode.upper()}")
    print("=" * 70)
    print()

    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return

    processes = [backend_process]

    # Start frontend based on mode
    if mode == "flask":
        frontend_process = start_flask_frontend()
        if frontend_process:
            processes.append(frontend_process)
        open_browser("flask")

    elif mode == "nextjs":
        frontend_process = start_nextjs_frontend()
        if frontend_process:
            processes.append(frontend_process)
        open_browser("nextjs")

    elif mode == "api-only":
        print("Backend API only")
        print("   Swagger UI: http://localhost:8000/docs")
        print()
        open_browser("api-only")

    print("=" * 70)
    print("  [OK] All services started!")
    print("=" * 70)
    print()
    print("Useful links:")
    print(f"   - Backend API: http://localhost:{BACKEND_PORT}/docs")
    if mode == "flask":
        print(f"   - Flask Frontend: http://localhost:{FLASK_PORT}")
    elif mode == "nextjs":
        print(f"   - Next.js Frontend: http://localhost:{NEXTJS_PORT}")
    print()
    print("Press Ctrl+C to stop")
    print()

    # Wait for shutdown
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n\n[INFO] Stopping services...")
        for proc in processes:
            proc.terminate()
        print("[OK] All services stopped.")


if __name__ == "__main__":
    main()
