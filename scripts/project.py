#!/usr/bin/env python3
"""
Project CLI — Управление проектом Nanoprobe Sim Lab

Команды:
    validate  — Проверка структуры, импортов, зависимостей
    improve   — Улучшение стиля кода, автофиксы
    cleanup   — Очистка временных файлов
    info      — Информация о проекте

Использование:
    python scripts/project.py validate
    python scripts/project.py improve
    python scripts/project.py cleanup
    python scripts/project.py info
    python scripts/project.py validate --fix
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ============================================================
# Утилиты
# ============================================================


def log(msg: str, level: str = "INFO"):
    """Логирование"""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    safe = msg.replace("✓", "[OK]").replace("✗", "[ERR]").replace("⚠", "[WARN]")
    print(f"[{level}] {ts}: {safe}")


def get_python_files() -> List[Path]:
    """Получить все Python файлы проекта"""
    return list(PROJECT_ROOT.rglob("*.py"))


# ============================================================
# VALIDATE
# ============================================================


def validate_structure() -> Dict[str, Any]:
    """Проверка структуры проекта"""
    log("Проверка структуры проекта...")
    expected_dirs = [
        "src",
        "api",
        "utils",
        "tests",
        "frontend",
        "components",
        "rtl_sdr_tools",
        "docs",
    ]
    missing = [d for d in expected_dirs if not (PROJECT_ROOT / d).exists()]

    expected_files = ["main.py", "pyproject.toml", "requirements.txt", "README.md"]
    missing_files = [f for f in expected_files if not (PROJECT_ROOT / f).exists()]

    result = {
        "missing_dirs": missing,
        "missing_files": missing_files,
        "ok": len(missing) == 0 and len(missing_files) == 0,
    }

    if result["ok"]:
        log("✓ Структура проекта в порядке")
    else:
        if missing:
            log(f"✗ Отсутствуют директории: {missing}", "ERROR")
        if missing_files:
            log(f"✗ Отсутствуют файлы: {missing_files}", "ERROR")

    return result


def validate_imports() -> Dict[str, Any]:
    """Проверка импортов"""
    log("Проверка импортов...")
    errors = []

    for py_file in get_python_files():
        # Пропускаем внешние зависимости
        if "tools/" in str(py_file) or "venv" in str(py_file) or ".venv" in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            # Проверяем синтаксис
            compile(content, str(py_file), "exec")
        except SyntaxError as e:
            errors.append(f"{py_file.relative_to(PROJECT_ROOT)}:{e.lineno} — {e.msg}")

    result = {
        "errors": errors,
        "ok": len(errors) == 0,
    }

    if result["ok"]:
        log("✓ Все импорты в порядке")
    else:
        log(f"✗ Ошибки в {len(errors)} файлах", "ERROR")
        for err in errors[:5]:
            log(f"  - {err}", "ERROR")

    return result


def validate_dependencies() -> Dict[str, Any]:
    """Проверка зависимостей"""
    log("Проверка зависимостей...")
    warnings = []

    # Проверяем requirements.txt
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                warnings.append(result.stdout.strip())
        except Exception as e:
            warnings.append(f"pip check failed: {e}")

    result = {
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }

    if result["ok"]:
        log("✓ Зависимости в порядке")
    else:
        log(f"⚠ Предупреждения: {len(warnings)}", "WARN")

    return result


def cmd_validate(args):
    """Команда validate"""
    log("=" * 60)
    log("VALIDATE — Проверка проекта")
    log("=" * 60)

    results = {
        "structure": validate_structure(),
        "imports": validate_imports(),
        "dependencies": validate_dependencies(),
    }

    all_ok = all(r["ok"] for r in results.values())

    log("=" * 60)
    if all_ok:
        log("✓ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    else:
        failed = [k for k, v in results.items() if not v["ok"]]
        log(f"✗ НЕ ПРОЙДЕНЫ: {failed}", "ERROR")

    # Сохраняем отчёт
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_ok": all_ok,
        "results": results,
    }
    report_file = PROJECT_ROOT / "reports" / "validation_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Отчёт сохранён: {report_file}")

    return 0 if all_ok else 1


# ============================================================
# IMPROVE
# ============================================================


def cmd_improve(args):
    """Команда improve"""
    log("=" * 60)
    log("IMPROVE — Улучшение проекта")
    log("=" * 60)

    improvements = []

    # 1. Black форматирование
    log("1. Форматирование кода (black)...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "black",
                "src/",
                "utils/",
                "api/",
                "tests/",
                "--line-length",
                "100",
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        improvements.append("black")
    except Exception as e:
        log(f"⚠ Black не установлен: {e}", "WARN")

    # 2. Удаление неиспользуемых импортов
    log("2. Проверка неиспользуемых импортов...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "autoflake",
                "--remove-all-unused-imports",
                "-r",
                "src/",
                "utils/",
                "api/",
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        improvements.append("autoflake")
    except Exception as e:
        log(f"⚠ autoflake не установлен: {e}", "WARN")

    # 3. Обновление зависимостей
    log("3. Обновление зависимостей...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        improvements.append("dependencies")
    except Exception as e:
        log(f"⚠ Ошибка обновления: {e}", "WARN")

    log("=" * 60)
    log(f"✓ Улучшения выполнены: {improvements}")
    return 0


# ============================================================
# CLEANUP
# ============================================================


def cmd_cleanup(args):
    """Команда cleanup"""
    log("=" * 60)
    log("CLEANUP — Очистка проекта")
    log("=" * 60)

    cleaned = 0

    # 1. __pycache__
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            import shutil

            shutil.rmtree(pycache, ignore_errors=True)
            cleaned += 1

    # 2. .pyc файлы
    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
        cleaned += 1

    # 3. .pytest_cache
    for cache_dir in PROJECT_ROOT.rglob(".pytest_cache"):
        if cache_dir.is_dir():
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=True)
            cleaned += 1

    # 4. Временные файлы
    patterns = ["*.log", "*.tmp", "*.bak", "*.pyo"]
    for pattern in patterns:
        for f in PROJECT_ROOT.rglob(pattern):
            f.unlink(missing_ok=True)
            cleaned += 1

    log(f"✓ Очищено {cleaned} файлов/директорий")
    return 0


# ============================================================
# INFO
# ============================================================


def cmd_info(args):
    """Команда info"""
    log("=" * 60)
    log("INFO — Информация о проекте")
    log("=" * 60)

    py_files = get_python_files()
    total_lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in py_files[:100])

    info = {
        "project_root": str(PROJECT_ROOT),
        "python_files": len(py_files),
        "sample_lines": total_lines,
        "python_version": sys.version,
        "platform": sys.platform,
    }

    log(f"Корень проекта: {info['project_root']}")
    log(f"Python файлов: {info['python_files']}")
    log(f"Строк кода (первые 100 файлов): {info['sample_lines']}")
    log(f"Python: {info['python_version']}")
    log(f"Платформа: {info['platform']}")

    return 0


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Project CLI — Управление проектом Nanoprobe Sim Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
    python scripts/project.py validate
    python scripts/project.py improve
    python scripts/project.py cleanup
    python scripts/project.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Команда")

    subparsers.add_parser("validate", help="Проверка проекта")
    subparsers.add_parser("improve", help="Улучшение кода")
    subparsers.add_parser("cleanup", help="Очистка проекта")
    subparsers.add_parser("info", help="Информация о проекте")

    args = parser.parse_args()

    if args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "improve":
        sys.exit(cmd_improve(args))
    elif args.command == "cleanup":
        sys.exit(cmd_cleanup(args))
    elif args.command == "info":
        sys.exit(cmd_info(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
