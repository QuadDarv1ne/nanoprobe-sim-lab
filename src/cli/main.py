"""
Главная консольная утилита проекта Лаборатория моделирования нанозонда.
"""

import atexit
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration paths
CONFIG_PATH = project_root / "config" / "config.json"

# Logger setup
logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# Активные процессы
_active_processes = {}


def show_header():
    """Отображает заголовок программы."""
    logger.info("=" * 80)
    logger.info(" ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
    logger.info(" Nanoprobe Simulation Lab - Main Console")
    logger.info("=" * 80)
    logger.info(f"Время запуска: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")


def show_project_overview():
    """Отображает обзор проекта."""
    logger.info("Проект включает три взаимосвязанных модуля:")
    logger.info(" 1. Симулятор аппаратного обеспечения СЗМ на C++")
    logger.info(" 2. Анализатор изображений поверхности на Python")
    logger.info(" 3. Наземная станция SSTV на Python/C++")
    logger.info("")


def show_menu():
    """Отображает главное меню."""
    logger.info("ДОСТУПНЫЕ ОПЕРАЦИИ:")
    logger.info(" 1. Запустить симулятор СЗМ (C++)")
    logger.info(" 2. Запустить анализатор изображений (Python)")
    logger.info(" 3. Запустить наземную станцию SSTV (Python/C++)")
    logger.info(" 4. Показать информацию о проекте")
    logger.info(" 5. Показать текущую лицензию")
    logger.info(" 6. Очистить кэш проекта")
    logger.info(" 0. Выход")
    logger.info("")


def run_spm_simulator() -> bool:
    """Запускает симулятор СЗМ."""
    logger.info("Запуск симулятора СЗМ...")
    try:
        cpp_build = project_root / "components" / "cpp-spm-hardware-sim" / "build"
        cpp_path = cpp_build / "spm-simulator"
        python_spm = (
            project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py"
        )

        if cpp_path.exists():
            logger.info(f"Запуск C++ версии: {cpp_path}")
            process = subprocess.Popen([str(cpp_path)], cwd=str(project_root))
            _active_processes["spm"] = process
            process.wait()
            return True
        elif python_spm.exists():
            logger.info(f"Запуск Python версии: {python_spm}")
            process = subprocess.Popen([sys.executable, str(python_spm)], cwd=str(project_root))
            _active_processes["spm"] = process
            process.wait()
            return True
        else:
            logger.error("Файлы симулятора СЗМ не найдены")
            return False
    except Exception as e:
        logger.error(f"Ошибка при запуске симулятора СЗМ: {e}")
        return False


def run_surface_analyzer() -> bool:
    """Запускает анализатор изображений."""
    logger.info("Запуск анализатора изображений поверхности...")
    try:
        analyzer_path = (
            project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py"
        )

        if analyzer_path.exists():
            logger.info(f"Запуск: {analyzer_path}")
            process = subprocess.Popen([sys.executable, str(analyzer_path)], cwd=str(project_root))
            _active_processes["analyzer"] = process
            process.wait()
            return True
        else:
            logger.error("Файл анализатора изображений не найден")
            return False
    except Exception as e:
        logger.error(f"Ошибка при запуске анализатора изображений: {e}")
        return False


def run_sstv_groundstation() -> bool:
    """Запускает наземную станцию SSTV."""
    logger.info("Запуск наземной станции SSTV...")
    try:
        station_path = project_root / "components" / "py-sstv-groundstation" / "src" / "main.py"

        if station_path.exists():
            logger.info(f"Запуск: {station_path}")
            process = subprocess.Popen([sys.executable, str(station_path)], cwd=str(project_root))
            _active_processes["sstv"] = process
            process.wait()
            return True
        else:
            logger.error("Файл наземной станции SSTV не найден")
            return False
    except Exception as e:
        logger.error(f"Ошибка при запуске наземной станции SSTV: {e}")
        return False


def show_project_info():
    """Показывает информацию о проекте."""
    logger.info("ИНФОРМАЦИЯ О ПРОЕКТЕ:")
    logger.info("-" * 40)
    logger.info("Название: Лаборатория моделирования нанозонда")
    logger.info("Версия: 1.0.0")
    logger.info("Описание: Комплексный проект для моделирования")
    logger.info(" сканирующей зондовой микроскопии")
    logger.info(" и обработки изображений поверхности")
    logger.info("Автор: Школа программирования Maestro7IT")
    logger.info("Лицензия: Проприетарная (ограниченные права)")
    logger.info("-" * 40)


def show_license():
    """Показывает информацию о лицензии."""
    logger.info("ИНФОРМАЦИЯ О ЛИЦЕНЗИИ:")
    logger.info("-" * 40)
    logger.info("Лицензия: Проприетарная лицензия")
    logger.info("Владелец: Школа программирования Maestro7IT")
    logger.info("Все права защищены")
    logger.info("")
    logger.info("ОГРАНИЧЕННЫЕ ПРАВА:")
    logger.info("• Использование в образовательных целях")
    logger.info("• Использование в научных исследованиях")
    logger.info("• Использование в некоммерческих проектах")
    logger.info("• Модификация исходного кода")
    logger.info("• Распространение в неизменном виде")
    logger.info("")
    logger.info("ЗАПРЕЩЕНО:")
    logger.info("• Коммерческое использование")
    logger.info("• Распространение с удалением авторских прав")
    logger.info("• Использование в военных целях")
    logger.info("• Использование в проектах с закрытым исходным кодом")
    logger.info("-" * 40)


def clean_project_cache() -> bool:
    """Очищает кэш проекта."""
    logger.info("Очистка кэша проекта...")
    try:
        from utils.caching.cache_manager import CacheManager

        cache_manager = CacheManager(str(project_root))
        stats = cache_manager.get_cache_statistics()
        logger.info(f"Текущий размер кэша: {stats['total_cache_size_mb']} MB")
        logger.info(f"Всего файлов в кэше: {stats['total_files']}")

        result = cache_manager.auto_cleanup()
        if "status" in result:
            logger.info(f"Статус: {result['status']}")
        else:
            logger.info(f"Удалено файлов: {result['deleted_files']}")
            logger.info(f"Освобождено места: {result['freed_space_mb']} MB")

        memory_result = cache_manager.optimize_memory_usage()
        logger.info(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")
        logger.info("Очистка кэша завершена успешно!")
        return True
    except ImportError:
        logger.error("Модуль управления кэшем не найден")
        return False
    except Exception as e:
        logger.error(f"Ошибка при очистке кэша: {e}")
        return False


def _cleanup_processes():
    """Очищает все активные процессы."""
    for name, process in _active_processes.items():
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
                logger.info(f"✓ Процесс {name} остановлен")
        except Exception:
            try:
                process.kill()
                logger.info(f"✓ Процесс {name} уничтожен")
            except Exception as e:
                logger.error(f"✗ Не удалось остановить процесс {name}: {e}")
    _active_processes.clear()


def auto_cleanup_on_exit():
    """Автоматическая очистка кэша при завершении программы."""
    logger.info("\n" + "=" * 50)
    logger.info("Завершение работы...")
    if _active_processes:
        logger.info("Остановка активных процессов...")
        _cleanup_processes()
    logger.info("Автоматическая очистка кэша...")
    try:
        cleanup_success = clean_project_cache()
        if cleanup_success:
            logger.info("✓ Завершение работы выполнено успешно")
        else:
            logger.warning("⚠ Завершение работы завершено с предупреждениями")
    except Exception as e:
        logger.error(f"❌ Ошибка при завершении работы: {e}")
    logger.info("=" * 50)


def main():
    """Главная функция программы."""
    atexit.register(auto_cleanup_on_exit)
    logger.info("Инициализация проекта...")
    try:
        from utils.caching.cache_manager import CacheManager

        cache_manager = CacheManager(str(project_root))
        cache_manager.auto_cleanup()
    except Exception as e:
        logger.warning(f"⚠ Ошибка инициализации кэша: {e}")

    show_header()
    show_project_overview()

    while True:
        show_menu()
        try:
            choice = input("Выберите действие (0-6): ").strip()

            if choice == "1":
                run_spm_simulator()
            elif choice == "2":
                run_surface_analyzer()
            elif choice == "3":
                run_sstv_groundstation()
            elif choice == "4":
                show_project_info()
            elif choice == "5":
                show_license()
            elif choice == "6":
                clean_project_cache()
            elif choice == "0":
                logger.info("\nСпасибо за использование Лаборатории моделирования нанозонда")
                logger.info("До новых встреч :)")
                break
            else:
                logger.warning("Неверный выбор. Пожалуйста, выберите от 0 до 6.")
        except KeyboardInterrupt:
            logger.warning("\n\nРабота программы прервана пользователем.")
            break
        except Exception as e:
            logger.error(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
