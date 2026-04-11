#!/usr/bin/env python3
"""
Скрипт проверки на циклические импорты в utils/

Использование:
    python scripts/check_cyclic_imports.py

Выявляет:
- Циклические зависимости между модулями
- Проблемные цепочки импортов
- Рекомендации по рефакторингу
"""

import ast
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ImportAnalyzer:
    """Анализатор импортов для выявления циклических зависимостей"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.module_files: Dict[str, Path] = {}

    def discover_modules(self) -> List[str]:
        """Обнаруживает все Python модули в директории"""
        modules = []

        for py_file in self.root_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                # Пакет
                module_path = py_file.relative_to(self.root_dir.parent)
                module_name = (
                    str(module_path).replace("/", ".").replace("\\", ".").replace(".__init__", "")
                )
            else:
                # Модуль
                module_path = py_file.relative_to(self.root_dir.parent)
                module_name = (
                    str(module_path).replace("/", ".").replace("\\", ".").replace(".py", "")
                )

            # Пропускаем тесты и venv
            if any(part in ["tests", "venv", ".venv", "__pycache__"] for part in py_file.parts):
                continue

            self.module_files[module_name] = py_file
            modules.append(module_name)

        return modules

    def parse_imports(self, module_name: str, file_path: Path) -> List[str]:
        """Парсит импорты из файла"""
        imports = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Относительные импорты
                        if node.level > 0:
                            # Преобразуем относительный импорт в абсолютный
                            parts = module_name.split(".")
                            base_parts = parts[: -node.level] if node.level <= len(parts) else []
                            full_module = ".".join(base_parts + [node.module])
                        else:
                            full_module = node.module

                        imports.append(full_module)

        except Exception as e:
            logger.warning(f"Ошибка парсинга {file_path}: {e}")

        return imports

    def build_import_graph(self):
        """Строит граф импортов"""
        logger.info("Обнаружение модулей...")
        modules = self.discover_modules()
        logger.info(f"Найдено {len(modules)} модулей")

        logger.info("Построение графа импортов...")
        for module_name, file_path in self.module_files.items():
            imports = self.parse_imports(module_name, file_path)

            for imp in imports:
                # Добавляем только если модуль существует в проекте
                if imp in self.module_files:
                    self.import_graph[module_name].add(imp)

    def find_cycles(self) -> List[List[str]]:
        """Находит все циклические зависимости"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.import_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Нашли цикл
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node in self.import_graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_import_depth(self, module: str) -> int:
        """Вычисляет глубину зависимостей модуля"""
        visited = set()

        def count_depth(mod: str, depth: int) -> int:
            if mod in visited or mod not in self.import_graph:
                return depth
            visited.add(mod)

            max_depth = depth
            for dep in self.import_graph.get(mod, set()):
                max_depth = max(max_depth, count_depth(dep, depth + 1))

            return max_depth

        return count_depth(module, 0)

    def generate_report(self) -> str:
        """Генерирует отчёт о циклических импортах"""
        self.build_import_graph()
        cycles = self.find_cycles()

        report = []
        report.append("=" * 80)
        report.append("Анализ циклических импортов")
        report.append("=" * 80)
        report.append(f"Модулей проанализировано: {len(self.module_files)}")
        report.append(f"Найдено циклических зависимостей: {len(cycles)}")
        report.append("")

        if cycles:
            report.append("⚠️  ОБНАРУЖЕНЫ ЦИКЛИЧЕСКИЕ ИМПОРТЫ:")
            report.append("")

            for i, cycle in enumerate(cycles, 1):
                report.append(f"Цикл #{i}:")
                report.append("  " + " → ".join(cycle))
                report.append("")

                # Рекомендации
                report.append("  Рекомендации:")
                report.append("  - Вынесите общие типы в отдельный модуль (types.py)")
                report.append("  - Используйте dependency injection")
                report.append("  - Примените lazy imports (импорты внутри функций)")
                report.append("")
        else:
            report.append("✅ Циклических импортов не обнаружено!")
            report.append("")

        # Топ модулей по глубине зависимостей
        report.append("📊 Топ-10 модулей по глубине зависимостей:")
        module_depths = []
        for module in self.module_files:
            depth = self.get_import_depth(module)
            if depth > 0:
                module_depths.append((module, depth))

        module_depths.sort(key=lambda x: x[1], reverse=True)

        for module, depth in module_depths[:10]:
            deps = len(self.import_graph.get(module, set()))
            report.append(f"  {module}: глубина={depth}, зависимостей={deps}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Основная функция"""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    if not utils_dir.exists():
        logger.error(f"Директория utils не найдена: {utils_dir}")
        sys.exit(1)

    logger.info(f"Анализ директории: {utils_dir}")
    logger.info("")

    analyzer = ImportAnalyzer(utils_dir)
    report = analyzer.generate_report()

    print(report)

    # Сохраняем отчёт
    report_path = project_root / "docs" / "cyclic_imports_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Отчёт сохранён: {report_path}")

    # Выход с кодом ошибки если найдены циклы
    if "ОБНАРУЖЕНЫ ЦИКЛИЧЕСКИЕ ИМПОРТЫ" in report:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
