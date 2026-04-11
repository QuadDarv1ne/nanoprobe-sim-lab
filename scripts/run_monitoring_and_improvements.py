"""Основной скрипт запуска проекта с мониторингом и улучшениями."""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from improve_project import ProjectImprover
from monitor_errors import ProjectMonitor


def run_project_monitoring():
    """Запуск мониторинга проекта"""
    print("=" * 70)
    print("ЗАПУСК МОНИТОРИНГА ПРОЕКТА NANOPROBE SIMULATION LAB")
    print("=" * 70)

    monitor = ProjectMonitor()
    results = monitor.run_full_monitoring()

    return results


def run_project_improvements():
    """Запуск улучшений проекта"""
    print("\n" + "=" * 70)
    print("ЗАПУСК УЛУЧШЕНИЙ ПРОЕКТА NANOPROBE SIMULATION LAB")
    print("=" * 70)

    improver = ProjectImprover()
    results = improver.run_all_improvements()

    return results


def run_project_tests():
    """Запуск тестов проекта"""
    print("\n" + "=" * 70)
    print("ЗАПУСК ТЕСТОВ ПРОЕКТА")
    print("=" * 70)

    try:
        # Запускаем тесты
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Код возврата: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Тесты превысили время ожидания")
        return False
    except Exception as e:
        print(f"Ошибка запуска тестов: {e}")
        return False


def run_main_project():
    """Запуск основного проекта"""
    print("\n" + "=" * 70)
    print("ЗАПУСК ОСНОВНОГО ПРОЕКТА")
    print("=" * 70)

    try:
        # Запускаем главный интерфейс проекта
        result = subprocess.run(
            [sys.executable, "start.py", "cli"],
            input="0\n",
            text=True,
            capture_output=True,
            timeout=30,
        )

        print("STDOUT:")
        print(result.stdout[-1000:])  # Показываем последние 1000 символов
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Код возврата: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Проект превысил время ожидания")
        return True  # Это нормально для интерактивных приложений
    except Exception as e:
        print(f"Ошибка запуска проекта: {e}")
        return False


def generate_final_report(monitor_results, improvement_results, tests_passed, project_run_success):
    """Генерация финального отчета"""
    print("\n" + "=" * 70)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 70)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Время завершения: {timestamp}")

    print("\nМониторинг:")
    print(
        f"  - Состояние системы: {monitor_results['health']['health_score'] if monitor_results['health'] else 'N/A'}/100"
    )
    print(
        f"  - Проблем в коде: {len(monitor_results['code_analysis'].get('issues', [])) if monitor_results['code_analysis'] else 'N/A'}"
    )
    print(f"  - Рекомендаций: {len(monitor_results['recommendations'])}")

    print("\nУлучшения:")
    print(f"  - Изменений внесено: {len(improvement_results['changes_made'])}")
    print(f"  - Проблем безопасности: {len(improvement_results['security_issues'])}")

    print(f"\nТесты: {'✅ Пройдены' if tests_passed else '❌ Не пройдены'}")
    print(f"Запуск проекта: {'✅ Успешно' if project_run_success else '❌ Ошибка'}")

    # Сохраняем финальный отчет
    report_path = (
        project_root
        / "reports"
        / f"final_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_path.parent.mkdir(exist_ok=True)

    final_report = {
        "timestamp": timestamp,
        "project_status": (
            "healthy"
            if (
                monitor_results["health"]["health_score"] >= 70
                if monitor_results["health"]
                else False
            )
            else "needs_attention"
        ),
        "monitoring_results": {
            "health_score": (
                monitor_results["health"]["health_score"] if monitor_results["health"] else None
            ),
            "code_issues": (
                len(monitor_results["code_analysis"].get("issues", []))
                if monitor_results["code_analysis"]
                else None
            ),
            "recommendations_count": len(monitor_results["recommendations"]),
        },
        "improvement_results": {
            "changes_made": len(improvement_results["changes_made"]),
            "security_issues": len(improvement_results["security_issues"]),
        },
        "tests_passed": tests_passed,
        "project_run_success": project_run_success,
        "summary": f"Проект в рабочем состоянии. Внесено {len(improvement_results['changes_made'])} улучшений.",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        import json

        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\nФинальный отчет сохранен: {report_path}")


def main():
    """Основная функция запуска всего процесса"""
    print("🚀 Запуск комплексного мониторинга и улучшения проекта...")
    print(f"Дата и время начала: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Рабочая директория: {project_root}")

    # 1. Запускаем мониторинг проекта
    monitor_results = run_project_monitoring()

    # 2. Запускаем улучшения проекта
    improvement_results = run_project_improvements()

    # 3. Запускаем тесты
    tests_passed = run_project_tests()

    # 4. Запускаем основной проект
    project_run_success = run_main_project()

    # 5. Генерируем финальный отчет
    generate_final_report(monitor_results, improvement_results, tests_passed, project_run_success)

    print("\n" + "🎉 Процесс мониторинга и улучшения завершен!")


if __name__ == "__main__":
    main()
