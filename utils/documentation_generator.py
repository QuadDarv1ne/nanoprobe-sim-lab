# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль автоматической документации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для автоматической генерации
документации из исходного кода и комментариев.
"""

import os
import sys
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class DocItem:
    """Элемент документации"""
    name: str
    type: str  # module, class, function, method
    docstring: str
    signature: str
    file_path: str
    line_number: int
    parameters: List[Dict[str, str]]
    return_type: str
    decorators: List[str]

class DocumentationGenerator:
    """
    Класс генератора документации
    Обеспечивает автоматическую генерацию документации
    из исходного кода проекта.
    """


    def __init__(self, project_root: str = "."):
        """
        Инициализирует генератор документации

        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()
        self.doc_items = []
        self.project_info = {}


    def analyze_project_structure(self) -> Dict[str, Any]:
        """
        Анализирует структуру проекта

        Returns:
            Информация о структуре проекта
        """
        structure = {
            'modules': [],
            'packages': [],
            'files': [],
            'directories': []
        }

        # Анализируем директории
        for item in self.project_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                structure['directories'].append({
                    'name': item.name,
                    'path': str(item.relative_to(self.project_root)),
                    'type': 'package' if (item / '__init__.py').exists() else 'directory'
                })
                if (item / '__init__.py').exists():
                    structure['packages'].append(item.name)
            elif item.is_file() and item.suffix == '.py' and not item.name.startswith('.'):
                structure['files'].append({
                    'name': item.name,
                    'path': str(item.relative_to(self.project_root)),
                    'size': item.stat().st_size
                })
                if item.name != '__init__.py':
                    structure['modules'].append(item.stem)

        return structure


    def extract_docstrings(self, include_patterns: List[str] = None,

                          exclude_patterns: List[str] = None) -> List[DocItem]:
        """
        Извлекает docstring из проекта

        Args:
            include_patterns: Паттерны для включения файлов
            exclude_patterns: Паттерны для исключения файлов

        Returns:
            Список элементов документации
        """
        if include_patterns is None:
            include_patterns = ['*.py']

        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '*.pyc', 'venv', '.git', 'tests']

        # Находим все Python файлы
        python_files = []
        for pattern in include_patterns:
            for file_path in self.project_root.rglob(pattern):
                if not any(exclude in str(file_path) for exclude in exclude_patterns):
                    python_files.append(file_path)

        print(f"Найдено Python файлов для анализа: {len(python_files)}")

        # Извлекаем документацию из каждого файла
        for file_path in python_files:
            try:
                self._extract_from_file(file_path)
            except Exception as e:
                print(f"Ошибка анализа файла {file_path}: {e}")

        return self.doc_items


    def _extract_from_file(self, file_path: Path):
        """Извлекает документацию из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Парсим AST
            tree = ast.parse(content)

            # Извлекаем docstring модуля
            module_doc = ast.get_docstring(tree)
            if module_doc:
                self.doc_items.append(DocItem(
                    name=file_path.stem,
                    type='module',
                    docstring=module_doc,
                    signature='',
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=1,
                    parameters=[],
                    return_type='',
                    decorators=[]
                ))

            # Извлекаем классы и функции
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._extract_class_doc(node, file_path, content)
                elif isinstance(node, ast.FunctionDef):
                    # Проверяем, является ли функция методом класса
                    is_method = any(isinstance(parent, ast.ClassDef)
                                  for parent in ast.walk(tree)
                                  if node in ast.iter_child_nodes(parent))
                    self._extract_function_doc(node, file_path, content, is_method)

        except Exception as e:
            print(f"Ошибка обработки файла {file_path}: {e}")


    def _extract_class_doc(self, node: ast.ClassDef, file_path: Path, content: str):
        """Извлекает документацию класса"""
        docstring = ast.get_docstring(node)
        if not docstring:
            return

        # Получаем сигнатуру класса
        signature = f"class {node.name}"
        if node.bases:
            base_names = [ast.unparse(base) for base in node.bases]
            signature += f"({', '.join(base_names)})"

        # Получаем декораторы
        decorators = [ast.unparse(dec) for dec in node.decorator_list]

        self.doc_items.append(DocItem(
            name=node.name,
            type='class',
            docstring=docstring,
            signature=signature,
            file_path=str(file_path.relative_to(self.project_root)),
            line_number=node.lineno,
            parameters=[],
            return_type='',
            decorators=decorators
        ))



    def _extract_function_doc(self, node: ast.FunctionDef, file_path: Path,
                            content: str, is_method: bool = False):
        """Извлекает документацию функции/метода"""
        docstring = ast.get_docstring(node)
        if not docstring:
            return

        # Получаем сигнатуру
        params = []
        for arg in node.args.args:
            param_info = {'name': arg.arg, 'type': '', 'default': ''}
            if arg.annotation:
                param_info['type'] = ast.unparse(arg.annotation)
            params.append(param_info)

        # Добавляем *args и **kwargs
        if node.args.vararg:
            params.append({'name': f"*{node.args.vararg.arg}", 'type': '', 'default': ''})
        if node.args.kwarg:
            params.append({'name': f"**{node.args.kwarg.arg}", 'type': '', 'default': ''})

        # Получаем возвращаемый тип
        return_type = ''
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Получаем декораторы
        decorators = [ast.unparse(dec) for dec in node.decorator_list]

        # Определяем тип
        item_type = 'method' if is_method else 'function'

        self.doc_items.append(DocItem(
            name=node.name,
            type=item_type,
            docstring=docstring,
            signature=self._build_function_signature(node, is_method),
            file_path=str(file_path.relative_to(self.project_root)),
            line_number=node.lineno,
            parameters=params,
            return_type=return_type,
            decorators=decorators
        ))


    def _build_function_signature(self, node: ast.FunctionDef, is_method: bool) -> str:
        """Строит сигнатуру функции"""
        parts = []

        # Декораторы
        for dec in node.decorator_list:
            parts.append(f"@{ast.unparse(dec)}")

        # Определение функции
        if is_method:
            parts.append(f"def {node.name}(")
        else:
            parts.append(f"def {node.name}(")

        # Параметры
        param_parts = []
        args = node.args

        # Позиционные аргументы
        for i, arg in enumerate(args.args):
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {ast.unparse(arg.annotation)}"
            if i < len(args.defaults):
                default_val = ast.unparse(args.defaults[i])
                param_str += f" = {default_val}"
            param_parts.append(param_str)

        # *args
        if args.vararg:
            vararg_str = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                vararg_str += f": {ast.unparse(args.vararg.annotation)}"
            param_parts.append(vararg_str)

        # Keyword-only аргументы
        if args.kwonlyargs:
            if not args.vararg:
                param_parts.append("*")
            for i, arg in enumerate(args.kwonlyargs):
                param_str = arg.arg
                if arg.annotation:
                    param_str += f": {ast.unparse(arg.annotation)}"
                if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                    default_val = ast.unparse(args.kw_defaults[i])
                    param_str += f" = {default_val}"
                param_parts.append(param_str)

        # **kwargs
        if args.kwarg:
            kwargs_str = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                kwargs_str += f": {ast.unparse(args.kwarg.annotation)}"
            param_parts.append(kwargs_str)

        parts.append(", ".join(param_parts))
        parts.append(")")

        # Возвращаемый тип
        if node.returns:
            parts.append(f" -> {ast.unparse(node.returns)}")

        return "\n".join(parts)


    def generate_markdown_documentation(self, output_path: str = "docs/api_reference.md") -> str:
        """
        Генерирует документацию в формате Markdown

        Args:
            output_path: Путь для сохранения документации

        Returns:
            Путь к сгенерированному файлу
        """
        # Группируем элементы по файлам
        items_by_file = {}
        for item in self.doc_items:
            if item.file_path not in items_by_file:
                items_by_file[item.file_path] = []
            items_by_file[item.file_path].append(item)

        # Генерируем Markdown
        md_content = []
        md_content.append("# API Reference\n")
        md_content.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # Добавляем информацию о структуре проекта
        structure = self.analyze_project_structure()
        md_content.append("## Project Structure\n")
        for pkg in structure['packages']:
            md_content.append(f"- **{pkg}/** - Package\n")
        for mod in structure['modules']:
            md_content.append(f"- **{mod}.py** - Module\n")
        md_content.append("\n")

        # Добавляем документацию по каждому файлу
        for file_path, items in sorted(items_by_file.items()):
            md_content.append(f"## {file_path}\n")

            # Модуль
            module_items = [item for item in items if item.type == 'module']
            if module_items:
                module = module_items[0]
                md_content.append(f"### Module: {module.name}\n")
                md_content.append(f"{self._format_docstring(module.docstring)}\n")

            # Классы
            class_items = [item for item in items if item.type == 'class']
            if class_items:
                md_content.append("### Classes\n")
                for cls in class_items:
                    md_content.append(f"#### {cls.name}\n")
                    if cls.decorators:
                        for dec in cls.decorators:
                            md_content.append(f"`@{dec}`\n")
                    md_content.append(f"```python\n{cls.signature}\n```\n")
                    md_content.append(f"{self._format_docstring(cls.docstring)}\n")

            # Функции и методы
            func_items = [item for item in items if item.type in ['function', 'method']]
            if func_items:
                md_content.append("### Functions\n")
                for func in func_items:
                    func_type = "Method" if func.type == 'method' else "Function"
                    md_content.append(f"#### {func.name}\n")
                    if func.decorators:
                        for dec in func.decorators:
                            md_content.append(f"`@{dec}`\n")
                    md_content.append(f"```python\n{func.signature}\n```\n")
                    md_content.append(f"{self._format_docstring(func.docstring)}\n")

                    # Параметры
                    if func.parameters:
                        md_content.append("**Parameters:**\n")
                        for param in func.parameters:
                            param_desc = f"- `{param['name']}`"
                            if param['type']:
                                param_desc += f" (*{param['type']}*)"
                            md_content.append(param_desc)
                        md_content.append("\n")

                    # Возвращаемое значение
                    if func.return_type:
                        md_content.append(f"**Returns:** *{func.return_type}*\n\n")

            md_content.append("\n")

        # Сохраняем файл
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))

        return str(output_file)


    def generate_html_documentation(self, output_path: str = "docs/api_reference.html") -> str:
        """
        Генерирует документацию в формате HTML

        Args:
            output_path: Путь для сохранения документации

        Returns:
            Путь к сгенерированному файлу
        """
        # Это упрощенная реализация HTML-генерации
        # В реальном проекте лучше использовать шаблоны Jinja2

        md_content = self.generate_markdown_documentation()

        # Преобразуем Markdown в HTML (упрощенно)
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>API Reference</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1, h2, h3, h4 { color: #333; }",
            "code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }",
            "pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }",
            ".module { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            md_content.replace('\n', '<br>'),
            "</body>",
            "</html>"
        ]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))

        return str(output_file)


    def generate_json_documentation(self, output_path: str = "docs/api_reference.json") -> str:
        """
        Генерирует документацию в формате JSON

        Args:
            output_path: Путь для сохранения документации

        Returns:
            Путь к сгенерированному файлу
        """
        # Конвертируем doc_items в словари
        doc_dicts = [asdict(item) for item in self.doc_items]

        # Создаем структуру данных
        data = {
            'metadata': {
                'project': 'Nanoprobe Simulation Lab',
                'generated': datetime.now().isoformat(),
                'total_items': len(self.doc_items)
            },
            'project_structure': self.analyze_project_structure(),
            'documentation_items': doc_dicts
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return str(output_file)


    def _format_docstring(self, docstring: str) -> str:
        """Форматирует docstring для отображения"""
        if not docstring:
            return "No documentation available."

        # Обрабатываем многострочные docstring
        lines = docstring.strip().split('\n')
        formatted_lines = []

        for line in lines:
            # Убираем начальные отступы
            stripped = line.lstrip()
            formatted_lines.append(stripped)

        return '\n'.join(formatted_lines)


    def get_statistics(self) -> Dict[str, int]:
        """Получает статистику по документации"""
        stats = {
            'total_items': len(self.doc_items),
            'modules': len([item for item in self.doc_items if item.type == 'module']),
            'classes': len([item for item in self.doc_items if item.type == 'class']),
            'functions': len([item for item in self.doc_items if item.type == 'function']),
            'methods': len([item for item in self.doc_items if item.type == 'method']),
            'files': len(set(item.file_path for item in self.doc_items))
        }
        return stats

def main():
    """Главная функция для демонстрации генератора документации"""
    print("=== ГЕНЕРАТОР ДОКУМЕНТАЦИИ ===")

    # Создаем генератор документации
    doc_gen = DocumentationGenerator()

    print("✓ Генератор документации инициализирован")
    print(f"✓ Корневая директория: {doc_gen.project_root}")

    # Извлекаем docstring из проекта
    print("\nИзвлечение документации из кода...")
    doc_items = doc_gen.extract_docstrings()

    print(f"✓ Извлечено элементов документации: {len(doc_items)}")

    # Показываем статистику
    stats = doc_gen.get_statistics()
    print("\nСтатистика по документации:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    # Анализируем структуру проекта
    structure = doc_gen.analyze_project_structure()
    print(f"\nСтруктура проекта:")
    print(f"  - Директорий: {len(structure['directories'])}")
    print(f"  - Пакетов: {len(structure['packages'])}")
    print(f"  - Модулей: {len(structure['modules'])}")

    # Генерируем документацию в разных форматах
    print("\nГенерация документации...")

    # Markdown
    md_path = doc_gen.generate_markdown_documentation()
    print(f"✓ Markdown документация: {md_path}")

    # JSON
    json_path = doc_gen.generate_json_documentation()
    print(f"✓ JSON документация: {json_path}")

    # HTML
    html_path = doc_gen.generate_html_documentation()
    print(f"✓ HTML документация: {html_path}")

    print("\nГенератор документации успешно протестирован")
    print("\nДоступные функции:")
    print("- Извлечение docstring: extract_docstrings()")
    print("- Генерация Markdown: generate_markdown_documentation()")
    print("- Генерация HTML: generate_html_documentation()")
    print("- Генерация JSON: generate_json_documentation()")
    print("- Анализ структуры: analyze_project_structure()")
    print("- Статистика: get_statistics()")

if __name__ == "__main__":
    main()

