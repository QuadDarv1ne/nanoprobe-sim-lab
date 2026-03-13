"""Скрипт сборки C++ компонентов для проекта Nanoprobe Simulation Lab."""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


class CPPBuilder:
    """Менеджер сборки C++ компонентов"""

    def __init__(self, project_root: str = None):
        """
        Инициализация сборщика

        Args:
            project_root: Корневая директория проекта
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        else:
            project_root = Path(project_root)

        self.project_root = project_root
        self.cpp_component_dir = self.project_root / "components" / "cpp-spm-hardware-sim"
        self.build_dir = self.cpp_component_dir / "build"

        # Проверка доступности инструментов
        self.cmake_available = self._check_command("cmake")
        self.make_available = self._check_command("make") or self._check_command("ninja")
        self.msbuild_available = self._check_command("msbuild") if os.name == "nt" else False

    def _check_command(self, cmd: str) -> bool:
        """Проверка доступности команды"""
        try:
            subprocess.run(
                [cmd, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def check_prerequisites(self) -> Dict[str, bool]:
        """
        Проверка необходимых инструментов

        Returns:
            Словарь с доступностью инструментов
        """
        return {
            'cmake': self.cmake_available,
            'make_or_ninja': self.make_available,
            'msbuild': self.msbuild_available,
            'cpp_component_exists': self.cpp_component_dir.exists(),
            'cmakelists_exists': (self.cpp_component_dir / "CMakeLists.txt").exists(),
        }

    def clean_build(self) -> bool:
        """
        Очистка директории сборки

        Returns:
            Успешность очистки
        """
        try:
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
                print(f"✓ Директория сборки очищена: {self.build_dir}")
            return True
        except Exception as e:
            print(f"❌ Ошибка очистки: {e}")
            return False

    def configure(self, build_type: str = "Release", extra_args: list = None) -> bool:
        """
        Конфигурация сборки (запуск cmake)

        Args:
            build_type: Тип сборки (Release, Debug, RelWithDebInfo)
            extra_args: Дополнительные аргументы cmake

        Returns:
            Успешность конфигурации
        """
        if not self.cmake_available:
            print("❌ CMake не найден. Установите CMake 3.15+")
            return False

        # Создание директории сборки
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Формирование команды
        cmd = ["cmake", "..", f"-DCMAKE_BUILD_TYPE={build_type}"]

        if extra_args:
            cmd.extend(extra_args)

        # Определение генератора для Windows
        if os.name == "nt" and self.msbuild_available:
            cmd.extend(["-G", "Visual Studio 17 2022"])

        print(f"Конфигурация сборки ({build_type})...")
        print(f"Команда: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=self.build_dir, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✓ Конфигурация завершена успешно")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка конфигурации:\n{e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ Не удалось запустить cmake")
            return False

    def build(self, target: str = None, jobs: int = None) -> bool:
        """
        Сборка проекта

        Args:
            target: Цель для сборки
            jobs: Количество потоков

        Returns:
            Успешность сборки
        """
        if not self.build_dir.exists():
            print("❌ Директория сборки не найдена. Запустите configure()")
            return False

        # Формирование команды
        if os.name == "nt" and self.msbuild_available:
            cmd = ["cmake", "--build", ".", "--config", "Release"]
        else:
            cmd = ["cmake", "--build", "."]

        if target:
            cmd.extend(["--target", target])

        if jobs:
            cmd.extend(["--parallel", str(jobs)])
        else:
            cmd.append("--parallel")  # Использовать все доступные ядра

        print("Сборка проекта...")
        print(f"Команда: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=self.build_dir, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✓ Сборка завершена успешно")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка сборки:\n{e.stderr}")
            return False

    def install(self, install_prefix: str = None) -> bool:
        """
        Установка собранного проекта

        Args:
            install_prefix: Префикс установки

        Returns:
            Успешность установки
        """
        if not (self.build_dir / "Makefile").exists() and not os.name == "nt":
            print("❌ Проект не собран. Запустите build()")
            return False

        cmd = ["cmake", "--install", "."]

        if install_prefix:
            cmd.extend(["--prefix", install_prefix])

        print(f"Установка в {install_prefix or 'систему'}...")

        try:
            result = subprocess.run(cmd, cwd=self.build_dir, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✓ Установка завершена")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки:\n{e.stderr}")
            return False

    def run(self, args: list = None) -> int:
        """
        Запуск собранного исполняемого файла

        Args:
            args: Аргументы командной строки

        Returns:
            Код возврата программы
        """
        # Поиск исполняемого файла
        if os.name == "nt":
            exe_path = self.build_dir / "bin" / "Release" / "spm-simulator.exe"
            if not exe_path.exists():
                exe_path = self.build_dir / "bin" / "spm-simulator.exe"
        else:
            exe_path = self.build_dir / "bin" / "spm-simulator"

        if not exe_path.exists():
            print(f"❌ Исполняемый файл не найден: {exe_path}")
            return -1

        cmd = [str(exe_path)]
        if args:
            cmd.extend(args)

        print(f"Запуск: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd)
            return result.returncode
        except FileNotFoundError:
            print("❌ Не удалось запустить исполняемый файл")
            return -1

    def full_build(self, build_type: str = "Release", clean: bool = True) -> Dict[str, Any]:
        """
        Полная сборка проекта

        Args:
            build_type: Тип сборки
            clean: Очистить перед сборкой

        Returns:
            Результаты сборки
        """
        results = {
            'success': False,
            'steps': {},
            'executable_path': None,
        }

        # Очистка
        if clean:
            results['steps']['clean'] = self.clean_build()

        # Конфигурация
        results['steps']['configure'] = self.configure(build_type)

        if not results['steps']['configure']:
            return results

        # Сборка
        results['steps']['build'] = self.build()

        if not results['steps']['build']:
            return results

        # Поиск исполняемого файла
        if os.name == "nt":
            exe_path = self.build_dir / "bin" / build_type / "spm-simulator.exe"
        else:
            exe_path = self.build_dir / "bin" / "spm-simulator"

        if exe_path.exists():
            results['executable_path'] = str(exe_path)
            results['success'] = True
            print(f"\n✓ Исполняемый файл: {exe_path}")
        else:
            print("\n⚠ Исполняемый файл не найден")

        return results


def main():
    """Главная функция"""
    print("=" * 60)
    print("       СБОРКА C++ КОМПОНЕНТОВ NANOPROBE SIMULATION LAB")
    print("=" * 60)

    builder = CPPBuilder()

    # Проверка инструментов
    print("\nПроверка инструментов...")
    prereqs = builder.check_prerequisites()

    for tool, available in prereqs.items():
        status = "✓" if available else "❌"
        print(f"  {status} {tool}: {'доступен' if available else 'не доступен'}")

    if not prereqs['cmake']:
        print("\n❌ CMake не найден. Установите CMake 3.15+")
        print("   https://cmake.org/download/")
        sys.exit(1)

    # Полная сборка
    print("\nЗапуск полной сборки...")
    results = builder.full_build(clean=True)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СБОРКИ")
    print("=" * 60)

    for step, success in results['steps'].items():
        status = "✓" if success else "❌"
        print(f"  {status} {step}: {'успешно' if success else 'ошибка'}")

    if results['success']:
        print(f"\n✓ Сборка завершена успешно!")
        print(f"  Исполняемый файл: {results['executable_path']}")
        print("\nДля запуска используйте:")
        print(f"  {results['executable_path']}")
        print("  или")
        print("  python start.py manager spm-cpp")
    else:
        print("\n❌ Сборка не завершена")
        print("   Проверьте логи ошибок выше")
        sys.exit(1)


if __name__ == "__main__":
    main()
