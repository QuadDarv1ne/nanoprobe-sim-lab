# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Менеджер проекта для Лаборатории моделирования нанозонда
Этот скрипт предоставляет унифицированный интерфейс для управления
всеми компонентами проекта: симулятором СЗМ, анализатором изображений
и наземной станцией SSTV.
"""

import os
import sys
import atexit
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration paths
CONFIG_PATH = project_root / "config" / "config.json"

class ProjectManager:
    """
    Класс для управления всем проектом Лаборатории моделирования нанозонда
    Обеспечивает унифицированный интерфейс для всех компонентов проекта.
    """


    def __init__(self, config_file: str = None):
        """
        Инициализирует менеджер проекта

        Args:
            config_file: Путь к файлу конфигурации проекта
        """
        if config_file is None:
            config_file = str(CONFIG_PATH)

        self.config_file = config_file
        self.config = self.load_config()

        # Определяем пути к компонентам
        self.components = {
            'spm_simulator': self.config['components']['spm_simulator'],
            'surface_analyzer': self.config['components']['surface_analyzer'],
            'sstv_groundstation': self.config['components']['sstv_groundstation']
        }

        # Регистрируем автоматическую очистку кэша при завершении
        atexit.register(self._auto_cleanup_on_exit)


    def load_config(self) -> Dict:
        """
        Загружает конфигурацию проекта из JSON-файла

        Returns:
            Словарь с конфигурацией проекта
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Файл конфигурации {self.config_file} не найден")
            return {}
        except json.JSONDecodeError:
            print(f"Ошибка при чтении файла конфигурации {self.config_file}")
            return {}


    def run_spm_simulator(self, use_python: bool = True):
        """
        Запускает симулятор СЗМ

        Args:
            use_python: Использовать Python-реализацию вместо C++
        """
        if use_python:
            # Запускаем Python-реализацию
            spm_path = project_root / self.components['spm_simulator']['path'] / 'src' / 'spm_simulator.py'
            if spm_path.exists():
                print(f"Запуск симулятора СЗМ (Python-реализация): {spm_path}")
                subprocess.run([sys.executable, str(spm_path)])
            else:
                print(f"Файл симулятора СЗМ не найден: {spm_path}")
        else:
            # Запускаем C++ версию (если она доступна)
            build_dir = project_root / self.components['spm_simulator']['path'] / 'build'
            binary_path = build_dir / 'spm-simulator'

            if binary_path.exists():
                print(f"Запуск симулятора СЗМ (C++): {binary_path}")
                subprocess.run([str(binary_path)])
            else:
                print("C++ версия симулятора СЗМ не найдена. Пожалуйста, соберите проект сначала.")
                print("Выполнение: cd cpp-spm-hardware-sim && mkdir build && cd build && cmake .. && make")


    def run_surface_analyzer(self):
        """Запускает анализатор изображений поверхности"""
        analyzer_path = project_root / self.components['surface_analyzer']['path'] / 'src' / 'main.py'

        if analyzer_path.exists():
            print(f"Запуск анализатора изображений: {analyzer_path}")
            subprocess.run([sys.executable, str(analyzer_path)])
        else:
            # Если основной файл не найден, создаем простой тестовый скрипт
            print(f"Файл анализатора изображений не найден: {analyzer_path}")
            self.create_sample_analyzer()


    def create_sample_analyzer(self):
        """Создает пример скрипта анализатора изображений"""
        analyzer_path = project_root / self.components['surface_analyzer']['path']
        main_py_path = analyzer_path / 'src' / 'main.py'

        sample_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример скрипта для анализатора изображений поверхности
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_processor import ImageProcessor, calculate_surface_roughness

def main():

    print("=== АНАЛИЗАТОР ИЗОБРАЖЕНИЙ ПОВЕРХНОСТИ ===")
    print("Инициализация анализатора...")

    processor = ImageProcessor()

    # Здесь будет код для загрузки и анализа изображения
    print("Анализатор изображений готов к работе")
    print("Для использования загрузите изображение и примените фильтры")

    # Пример использования
    # processor.load_image("sample_image.jpg")
    # filtered = processor.apply_noise_reduction("gaussian")
    # edges = processor.detect_edges()

    print("Пример анализа изображения завершен")

if __name__ == "__main__":
    main()
'''

        os.makedirs(main_py_path.parent, exist_ok=True)
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(sample_code)

        print(f"Создан пример скрипта анализатора изображений: {main_py_path}")


    def run_sstv_station(self):
        """Запускает наземную станцию SSTV"""
        station_path = project_root / self.components['sstv_groundstation']['path'] / 'src' / 'main.py'

        if station_path.exists():
            print(f"Запуск наземной станции SSTV: {station_path}")
            subprocess.run([sys.executable, str(station_path)])
        else:
            # Если основной файл не найден, создаем простой тестовый скрипт
            print(f"Файл наземной станции SSTV не найден: {station_path}")
            self.create_sample_sstv_station()


    def create_sample_sstv_station(self):
        """Создает пример скрипта наземной станции SSTV"""
        station_path = project_root / self.components['sstv_groundstation']['path']
        main_py_path = station_path / 'src' / 'main.py'

        sample_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример скрипта для наземной станции SSTV
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sstv_decoder import SSTVDecoder, detect_sstv_signal


def main():
    print("=== НАЗЕМНАЯ СТАНЦИЯ SSTV ===")
    print("Инициализация декодера SSTV...")

    decoder = SSTVDecoder()

    # Здесь будет код для приема и декодирования SSTV-сигналов
    print("Наземная станция SSTV готова к работе")
    print("Для использования подключите SDR-приемник и начните поиск сигнала")

    # Пример использования
    # audio_file = "recorded_signal.wav"
    # decoded_image = decoder.decode_from_audio(audio_file)
    # if decoded_image:
    #     decoder.save_decoded_image("decoded_image.jpg")

    print("Пример декодирования SSTV завершен")

if __name__ == "__main__":
    main()
'''

        os.makedirs(main_py_path.parent, exist_ok=True)
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(sample_code)

        print(f"Создан пример скрипта наземной станции SSTV: {main_py_path}")


    def build_cpp_components(self):
        """Собирает C++ компоненты проекта"""
        print("Сборка C++ компонентов проекта...")

        spm_path = project_root / self.components['spm_simulator']['path']
        build_dir = spm_path / 'build'

        # Создаем директорию сборки
        os.makedirs(build_dir, exist_ok=True)

        # Запускаем cmake и make
        try:
            print("Запуск cmake...")
            subprocess.run(['cmake', '..', '-B', str(build_dir)], cwd=str(build_dir), check=True)
            print("Запуск make...")
            subprocess.run(['make'], cwd=str(build_dir), check=True)
            print("C++ компоненты успешно собраны!")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при сборке C++ компонентов: {e}")
            print("Убедитесь, что у вас установлены cmake и компилятор C++")


    def clean_cache(self):
        """Очищает кэш проекта"""
        try:
            from utils.cache_manager import CacheManager
            cache_manager = CacheManager(str(project_root))

            print("Очистка кэша проекта...")
            stats = cache_manager.get_cache_statistics()
            print(f"Текущий размер кэша: {stats['total_cache_size_mb']} MB")

            result = cache_manager.auto_cleanup()

            if "status" in result:
                print(f"Результат: {result['status']}")
            else:
                print(f"Удалено файлов: {result['deleted_files']}")
                print(f"Освобождено места: {result['freed_space_mb']} MB")

            # Оптимизация памяти
            memory_result = cache_manager.optimize_memory_usage()
            print(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")

            return True

        except ImportError:
            print("Модуль cache_manager не найден")
            return False
        except Exception as e:
            print(f"Ошибка при очистке кэша: {e}")
            return False


    def _auto_cleanup_on_exit(self):
        """Внутренняя функция автоматической очистки при завершении"""
        print("\n" + "="*50)
        print("Автоматическая очистка кэша через ProjectManager...")
        try:
            cleanup_success = self.clean_cache()
            if cleanup_success:
                print("✓ Автоматическая очистка кэша выполнена успешно")
            else:
                print("⚠ Автоматическая очистка кэша завершена с предупреждениями")
        except Exception as e:
            print(f"❌ Ошибка при автоматической очистке кэша: {e}")
        print("="*50)


    def show_project_info(self):
        """Показывает информацию о проекте"""
        project_info = self.config.get('project', {})

        print("\n" + "="*60)
        print(f"ИНФОРМАЦИЯ О ПРОЕКТЕ: {project_info.get('name', 'Nanoprobe Simulation Lab')}")
        print("="*60)
        print(f"Версия: {project_info.get('version', '1.0.0')}")
        print(f"Описание: {project_info.get('description', 'Проект не описан')}")
        print(f"Автор: {project_info.get('author', 'Не указан')}")
        print(f"Авторские права: {project_info.get('copyright', 'Не указаны')}")

        print("\nКОМПОНЕНТЫ ПРОЕКТА:")
        for name, info in self.components.items():
            print(f"  - {info['name']}: {info['description']}")
            print(f"    Путь: {info['path']}")
            print(f"    Язык: {info['language']}")

        print("\nЛИЦЕНЗИЯ:")
        license_info = self.config.get('license', {})
        print(f"  Тип: {license_info.get('type', 'Не указана')}")
        print(f"  Файл: {license_info.get('file', 'Не указан')}")
        print(f"  Владелец: {license_info.get('owner', 'Не указан')}")

        reserved_rights = license_info.get('reserved_rights', [])
        if reserved_rights:
            print("  Ограниченные права:")
            for right in reserved_rights:
                print(f"    - {right}")

        print("="*60)


    def show_menu(self):
        """Отображает главное меню проекта"""
        print("\n" + "="*50)
        print("    ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
        print("         Менеджер проекта")
        print("="*50)
        print("ДОСТУПНЫЕ ОПЕРАЦИИ:")
        print("  1. Запустить симулятор СЗМ (Python)")
        print("  2. Запустить симулятор СЗМ (C++)")
        print("  3. Запустить анализатор изображений")
        print("  4. Запустить наземную станцию SSTV")
        print("  5. Собрать C++ компоненты")
        print("  6. Очистить кэш проекта")
        print("  7. Показать информацию о проекте")
        print("  0. Выход")
        print("="*50)


    def run_interactive(self):
        """Запускает интерактивный режим менеджера проекта"""
        while True:
            self.show_menu()
            try:
                choice = input("\nВыберите действие (0-7): ").strip()

                if choice == '1':
                    self.run_spm_simulator(use_python=True)
                elif choice == '2':
                    self.run_spm_simulator(use_python=False)
                elif choice == '3':
                    self.run_surface_analyzer()
                elif choice == '4':
                    self.run_sstv_station()
                elif choice == '5':
                    self.build_cpp_components()
                elif choice == '6':
                    self.clean_cache()
                elif choice == '7':
                    self.show_project_info()
                elif choice == '0':
                    print("\nСпасибо за использование Лаборатории моделирования нанозонда!")
                    break
                else:
                    print("\nНеверный выбор. Пожалуйста, выберите от 0 до 7.")

            except KeyboardInterrupt:
                print("\n\nРабота программы прервана пользователем.")
                break
            except Exception as e:
                print(f"\nОшибка: {str(e)}")

def main():
    """Главная функция запуска менеджера проекта"""
    manager = ProjectManager()

    # Если переданы аргументы командной строки, выполняем соответствующую операцию
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'spm-python':
            manager.run_spm_simulator(use_python=True)
        elif command == 'spm-cpp':
            manager.run_spm_simulator(use_python=False)
        elif command == 'analyzer':
            manager.run_surface_analyzer()
        elif command == 'sstv':
            manager.run_sstv_station()
        elif command == 'build':
            manager.build_cpp_components()
        elif command == 'clean-cache':
            manager.clean_cache()
        elif command == 'info':
            manager.show_project_info()
        else:
            print(f"Неизвестная команда: {command}")
            print("Доступные команды: spm-python, spm-cpp, analyzer, sstv, build, clean-cache, info")
    else:
        # Запускаем интерактивный режим
        manager.run_interactive()

if __name__ == "__main__":
    main()

