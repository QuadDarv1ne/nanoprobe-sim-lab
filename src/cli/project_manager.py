"""Менеджер проекта для Лаборатории моделирования нанозонда."""

import atexit
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration paths
CONFIG_PATH = project_root / "config" / "config.json"

# Logger setup
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


class ProjectManager:
    """
    Класс для управления всем проектом Лаборатории моделирования нанозонда.
    Обеспечивает унифицированный интерфейс для всех компонентов проекта.
    """

    def __init__(self, config_file: str = None):
        """
        Инициализирует менеджер проекта.

        Args:
            config_file: Путь к файлу конфигурации проекта.
        """
        if config_file is None:
            config_file = str(CONFIG_PATH)
        self.config_file = config_file
        self.config = self.load_config()
        self.components = {
            "spm_simulator": self.config["components"]["spm_simulator"],
            "image_analyzer": self.config["components"]["image_analyzer"],
            "sstv_station": self.config["components"]["sstv_station"],
        }
        atexit.register(self._auto_cleanup_on_exit)

    def load_config(self) -> dict:
        """
        Загружает конфигурацию проекта из JSON-файла.

        Returns:
            Словарь с конфигурацией проекта.
        """
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Файл конфигурации {self.config_file} не найден")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Ошибка при чтении файла конфигурации {self.config_file}")
            return {}

    def run_spm_simulator(self, use_python: bool = True):
        """
        Запускает симулятор СЗМ.

        Args:
            use_python: Использовать Python-реализацию вместо C++.
        """
        if use_python:
            spm_path = (
                project_root / self.components["spm_simulator"]["path"] / "src" / "spm_simulator.py"
            )
            if spm_path.exists():
                logger.info(f"Запуск симулятора СЗМ (Python): {spm_path}")
                subprocess.run([sys.executable, str(spm_path)])
            else:
                logger.warning(f"Файл симулятора СЗМ не найден: {spm_path}")
        else:
            build_dir = project_root / self.components["spm_simulator"]["path"] / "build"
            binary_path = build_dir / "spm-simulator"
            if binary_path.exists():
                logger.info(f"Запуск симулятора СЗМ (C++): {binary_path}")
                subprocess.run([str(binary_path)])
            else:
                logger.warning("C++ версия симулятора СЗМ не найдена.")
                logger.warning(
                    "Соберите: cd cpp-spm-hardware-sim && mkdir build && "
                    "cd build && cmake .. && make"
                )

    def run_image_analyzer(self):
        """Запускает анализатор изображений поверхности."""
        analyzer_path = project_root / self.components["image_analyzer"]["path"] / "src" / "main.py"
        if analyzer_path.exists():
            logger.info(f"Запуск анализатора изображений: {analyzer_path}")
            subprocess.run([sys.executable, str(analyzer_path)])
        else:
            logger.warning(f"Файл анализатора изображений не найден: {analyzer_path}")
            self.create_sample_image_analyzer()

    def create_sample_image_analyzer(self):
        """Создает пример скрипта анализатора изображений."""
        analyzer_path = project_root / self.components["image_analyzer"]["path"]
        main_py_path = analyzer_path / "src" / "main.py"
        sample_code = """#!/usr/bin/env python3
\"\"\"Пример скрипта для анализатора изображений поверхности.\"\"\"
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from image_processor import ImageProcessor, calculate_surface_roughness

def main():
    \"\"\"Основная функция анализатора изображений.\"\"\"
    print("=== АНАЛИЗАТОР ИЗОБРАЖЕНИЙ ПОВЕРХНОСТИ ===")
    print("Инициализация анализатора...")
    processor = ImageProcessor()
    print("Анализатор изображений готов к работе")

if __name__ == "__main__":
    main()
"""
        os.makedirs(main_py_path.parent, exist_ok=True)
        with open(main_py_path, "w", encoding="utf-8") as f:
            f.write(sample_code)
        logger.info(f"Создан пример скрипта: {main_py_path}")

    def run_sstv_station(self):
        """Запускает наземную станцию SSTV."""
        station_path = (
            project_root / self.components["sstv_groundstation"]["path"] / "src" / "main.py"
        )
        if station_path.exists():
            logger.info(f"Запуск наземной станции SSTV: {station_path}")
            subprocess.run([sys.executable, str(station_path)])
        else:
            logger.warning(f"Файл наземной станции SSTV не найден: {station_path}")
            self.create_sample_sstv_station()

    def create_sample_sstv_station(self):
        """Создает пример скрипта наземной станции SSTV."""
        station_path = project_root / self.components["sstv_groundstation"]["path"]
        main_py_path = station_path / "src" / "main.py"
        sample_code = """#!/usr/bin/env python3
\"\"\"Пример скрипта для наземной станции SSTV.\"\"\"
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sstv_decoder import SSTVDecoder, detect_sstv_signal

def main():
    \"\"\"Основная функция SSTV станции.\"\"\"
    print("=== НАЗЕМНАЯ СТАНЦИЯ SSTV ===")
    print("Инициализация декодера SSTV...")
    decoder = SSTVDecoder()
    print("Наземная станция SSTV готова к работе")

if __name__ == "__main__":
    main()
"""
        os.makedirs(main_py_path.parent, exist_ok=True)
        with open(main_py_path, "w", encoding="utf-8") as f:
            f.write(sample_code)
        logger.info(f"Создан пример скрипта: {main_py_path}")

    def build_cpp_components(self):
        """Собирает C++ компоненты проекта."""
        logger.info("Сборка C++ компонентов проекта...")
        spm_path = project_root / self.components["spm_simulator"]["path"]
        build_dir = spm_path / "build"
        os.makedirs(build_dir, exist_ok=True)
        try:
            logger.info("Запуск cmake...")
            subprocess.run(["cmake", "..", "-B", str(build_dir)], cwd=str(build_dir), check=True)
            logger.info("Запуск make...")
            subprocess.run(["make"], cwd=str(build_dir), check=True)
            logger.info("C++ компоненты успешно собраны!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при сборке C++ компонентов: {e}")
            logger.warning("Убедитесь, что установлены cmake и компилятор C++")

    def clean_cache(self):
        """Очищает кэш проекта."""
        try:
            from utils.caching.cache_manager import CacheManager

            cache_manager = CacheManager(str(project_root))
            logger.info("Очистка кэша проекта...")
            stats = cache_manager.get_cache_statistics()
            logger.info(f"Текущий размер кэша: {stats['total_cache_size_mb']} MB")
            result = cache_manager.auto_cleanup()
            if "status" in result:
                logger.info(f"Результат: {result['status']}")
            else:
                logger.info(f"Удалено файлов: {result['deleted_files']}")
                logger.info(f"Освобождено места: {result['freed_space_mb']} MB")
            memory_result = cache_manager.optimize_memory_usage()
            logger.info(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")
            return True
        except ImportError:
            logger.error("Модуль cache_manager не найден")
            return False
        except Exception as e:
            logger.error(f"Ошибка при очистке кэша: {e}")
            return False

    def _auto_cleanup_on_exit(self):
        """Внутренняя функция автоматической очистки при завершении."""
        logger.info("\n" + "=" * 50)
        logger.info("Автоматическая очистка кэша через ProjectManager...")
        try:
            cleanup_success = self.clean_cache()
            if cleanup_success:
                logger.info("✓ Автоматическая очистка кэша выполнена успешно")
            else:
                logger.warning("⚠ Автоматическая очистка кэша завершена с предупреждениями")
        except Exception as e:
            logger.error(f"❌ Ошибка при автоматической очистке кэша: {e}")
        logger.info("=" * 50)

    def show_project_info(self):
        """Показывает информацию о проекте."""
        project_info = self.config.get("project", {})
        logger.info("\n" + "=" * 60)
        logger.info(
            f"ИНФОРМАЦИЯ О ПРОЕКТЕ: " f"{project_info.get('name', 'Nanoprobe Simulation Lab')}"
        )
        logger.info("=" * 60)
        logger.info(f"Версия: {project_info.get('version', '1.0.0')}")
        logger.info(f"Описание: {project_info.get('description', 'Проект не описан')}")
        logger.info(f"Автор: {project_info.get('author', 'Не указан')}")
        logger.info(f"Авторские права: {project_info.get('copyright', 'Не указаны')}")
        logger.info("\nКОМПОНЕНТЫ ПРОЕКТА:")
        for name, info in self.components.items():
            logger.info(f" - {info['name']}: {info['description']}")
            logger.info(f" Путь: {info['path']}")
            logger.info(f" Язык: {info['language']}")
        logger.info("\nЛИЦЕНЗИЯ:")
        license_info = self.config.get("license", {})
        logger.info(f" Тип: {license_info.get('type', 'Не указана')}")
        logger.info(f" Файл: {license_info.get('file', 'Не указан')}")
        logger.info(f" Владелец: {license_info.get('owner', 'Не указан')}")
        reserved_rights = license_info.get("reserved_rights", [])
        if reserved_rights:
            logger.info(" Ограниченные права:")
            for right in reserved_rights:
                logger.info(f" - {right}")
        logger.info("=" * 60)

    def show_menu(self):
        """Отображает главное меню проекта."""
        logger.info("\n" + "=" * 50)
        logger.info(" ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
        logger.info(" Менеджер проекта")
        logger.info("=" * 50)
        logger.info("ДОСТУПНЫЕ ОПЕРАЦИИ:")
        logger.info(" 1. Запустить симулятор СЗМ (Python)")
        logger.info(" 2. Запустить симулятор СЗМ (C++)")
        logger.info(" 3. Запустить анализатор изображений")
        logger.info(" 4. Запустить наземную станцию SSTV")
        logger.info(" 5. Собрать C++ компоненты")
        logger.info(" 6. Очистить кэш проекта")
        logger.info(" 7. Показать информацию о проекте")
        logger.info(" 0. Выход")
        logger.info("=" * 50)

    def run_interactive(self):
        """Запускает интерактивный режим менеджера проекта."""
        while True:
            self.show_menu()
            try:
                choice = input("\nВыберите действие (0-7): ").strip()
                if choice == "1":
                    self.run_spm_simulator(use_python=True)
                elif choice == "2":
                    self.run_spm_simulator(use_python=False)
                elif choice == "3":
                    self.run_image_analyzer()
                elif choice == "4":
                    self.run_sstv_station()
                elif choice == "5":
                    self.build_cpp_components()
                elif choice == "6":
                    self.clean_cache()
                elif choice == "7":
                    self.show_project_info()
                elif choice == "0":
                    logger.info("\nСпасибо за использование Лаборатории моделирования нанозонда!")
                    break
                else:
                    logger.warning("\nНеверный выбор. Пожалуйста, выберите от 0 до 7.")
            except KeyboardInterrupt:
                logger.info("\n\nРабота программы прервана пользователем.")
                break
            except Exception as e:
                logger.error(f"\nОшибка: {str(e)}")


def main():
    """Главная функция запуска менеджера проекта."""
    manager = ProjectManager()
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "spm-python":
            manager.run_spm_simulator(use_python=True)
        elif command == "spm-cpp":
            manager.run_spm_simulator(use_python=False)
        elif command == "analyzer":
            manager.run_image_analyzer()
        elif command == "sstv":
            manager.run_sstv_station()
        elif command == "build":
            manager.build_cpp_components()
        elif command == "clean-cache":
            manager.clean_cache()
        elif command == "info":
            manager.show_project_info()
        else:
            logger.error(f"Неизвестная команда: {command}")
            logger.info(
                "Доступные команды: spm-python, spm-cpp, analyzer, sstv, "
                "build, clean-cache, info"
            )
    else:
        manager.run_interactive()


if __name__ == "__main__":
    main()
