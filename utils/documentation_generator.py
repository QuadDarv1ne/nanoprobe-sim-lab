#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль генерации документации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для автоматической 
генерации документации и API справочника проекта.
"""

import ast
import inspect
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pydoc
import pypandoc
from jinja2 import Template
import markdown
from bs4 import BeautifulSoup


class DocumentationGenerator:
    """
    Класс генерации документации
    Обеспечивает автоматическую генерацию 
    документации и API справочника проекта.
    """
    
    def __init__(self, project_root: str = "."):
        """
        Инициализирует генератор документации
        
        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()
        self.modules = {}
        self.classes = {}
        self.functions = {}
        self.documentation_data = {}
    
    def analyze_code_structure(self, source_directory: str = ".") -> Dict[str, Any]:
        """
        Анализирует структуру кода проекта
        
        Args:
            source_directory: Директория с исходным кодом
            
        Returns:
            Словарь с анализом структуры кода
        """
        source_path = self.project_root / source_directory
        structure = {
            'directories': [],
            'files': [],
            'python_files': [],
            'modules': {},
            'classes': {},
            'functions': {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for root, dirs, files in os.walk(source_path):
            rel_root = Path(root).relative_to(source_path)
            
            # Добавляем директории
            for dir_name in dirs:
                dir_path = rel_root / dir_name
                structure['directories'].append(str(dir_path))
            
            # Обрабатываем файлы
            for file_name in files:
                file_path = Path(root) / file_name
                rel_file_path = file_path.relative_to(source_path)
                
                structure['files'].append(str(rel_file_path))
                
                if file_name.endswith('.py'):
                    structure['python_files'].append(str(rel_file_path))
                    
                    # Анализируем Python файл
                    module_info = self._analyze_python_file(file_path)
                    if module_info:
                        structure['modules'][str(rel_file_path)] = module_info
        
        return structure
    
    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Анализирует Python файл
        
        Args:
            file_path: Путь к Python файлу
            
        Returns:
            Словарь с информацией о файле
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            file_info = {
                'classes': [],
                'functions': [],
                'imports': [],
                'docstring': ast.get_docstring(tree),
                'file_path': str(file_path)
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        file_info['imports'].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [],
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno,
                        'parent_classes': [base.id for base in node.bases if isinstance(base, ast.Name)]
                    }
                    
                    # Находим методы класса
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'docstring': ast.get_docstring(item),
                                'line_number': item.lineno,
                                'parameters': [arg.arg for arg in item.args.args if arg.arg != 'self'],
                                'returns': self._extract_return_annotation(item)
                            }
                            class_info['methods'].append(method_info)
                    
                    file_info['classes'].append(class_info)
                elif isinstance(node, ast.FunctionDef) and node.name != '__init__':
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno,
                        'parameters': [arg.arg for arg in node.args.args],
                        'returns': self._extract_return_annotation(node)
                    }
                    file_info['functions'].append(func_info)
            
            return file_info
            
        except Exception as e:
            print(f"Ошибка анализа файла {file_path}: {e}")
            return {}
    
    def _extract_return_annotation(self, node: ast.AST) -> str:
        """
        Извлекает аннотацию возвращаемого значения
        
        Args:
            node: AST узел функции
            
        Returns:
            Строка с аннотацией возвращаемого значения
        """
        if hasattr(node, 'returns') and node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return str(node.returns.value)
            elif isinstance(node.returns, ast.Subscript):
                # Обработка сложных типов как List[str], Dict[str, int] и т.д.
                if hasattr(node.returns, 'value') and hasattr(node.returns.value, 'id'):
                    base_type = node.returns.value.id
                    if hasattr(node.returns, 'slice'):
                        if isinstance(node.returns.slice, ast.Index):
                            # Для старых версий Python
                            if hasattr(node.returns.slice.value, 'id'):
                                subtype = node.returns.slice.value.id
                            else:
                                subtype = str(node.returns.slice.value)
                            return f"{base_type}[{subtype}]"
                        elif hasattr(ast, 'Subscript') and isinstance(node.returns.slice, ast.Subscript):
                            # Для новых версий Python
                            return f"{base_type}[{ast.unparse(node.returns.slice) if hasattr(ast, 'unparse') else '...'}]"
                    return base_type
        return "Any"
    
    def generate_module_documentation(self, module_path: str) -> Dict[str, Any]:
        """
        Генерирует документацию для модуля
        
        Args:
            module_path: Путь к модулю
            
        Returns:
            Словарь с документацией модуля
        """
        module_file = self.project_root / module_path
        
        if not module_file.exists():
            return {}
        
        try:
            # Загружаем модуль
            spec = importlib.util.spec_from_file_location(module_path.stem, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            module_doc = {
                'name': module_path.stem,
                'docstring': module.__doc__ or '',
                'file_path': str(module_path),
                'classes': {},
                'functions': {},
                'attributes': {}
            }
            
            # Анализируем классы
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    class_doc = self._document_class(obj)
                    module_doc['classes'][name] = class_doc
            
            # Анализируем функции
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__:
                    func_doc = self._document_function(obj)
                    module_doc['functions'][name] = func_doc
            
            # Анализируем атрибуты
            for name, obj in inspect.getmembers(module):
                if not name.startswith('_') and not inspect.isclass(obj) and not inspect.isfunction(obj) and not inspect.ismodule(obj):
                    module_doc['attributes'][name] = {
                        'type': type(obj).__name__,
                        'value': str(obj)[:100] + '...' if len(str(obj)) > 100 else str(obj),  # Обрезаем длинные значения
                        'docstring': getattr(obj, '__doc__', '')
                    }
            
            return module_doc
            
        except Exception as e:
            print(f"Ошибка генерации документации для модуля {module_path}: {e}")
            return {}
    
    def _document_class(self, cls) -> Dict[str, Any]:
        """
        Документирует класс
        
        Args:
            cls: Класс для документирования
            
        Returns:
            Словарь с документацией класса
        """
        class_doc = {
            'name': cls.__name__,
            'docstring': cls.__doc__ or '',
            'methods': {},
            'properties': {},
            'inheritance': [base.__name__ for base in cls.__bases__ if base != object],
            'module': cls.__module__
        }
        
        # Анализируем методы
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name != '__init__':
                class_doc['methods'][name] = self._document_function(method)
        
        # Анализируем свойства
        for name, obj in inspect.getmembers(cls):
            if isinstance(obj, property):
                class_doc['properties'][name] = {
                    'docstring': obj.__doc__ or '',
                    'fget': obj.fget.__name__ if obj.fget else None,
                    'fset': obj.fset.__name__ if obj.fset else None
                }
        
        return class_doc
    
    def _document_function(self, func) -> Dict[str, Any]:
        """
        Документирует функцию
        
        Args:
            func: Функция для документирования
            
        Returns:
            Словарь с документацией функции
        """
        sig = inspect.signature(func)
        
        func_doc = {
            'name': func.__name__,
            'docstring': func.__doc__ or '',
            'signature': str(sig),
            'parameters': {},
            'return_annotation': str(func.__annotations__.get('return', 'Any')) if hasattr(func, '__annotations__') else 'Any',
            'module': func.__module__
        }
        
        # Документируем параметры
        for param_name, param in sig.parameters.items():
            param_doc = {
                'kind': param.kind.name,
                'default': param.default if param.default != inspect.Parameter.empty else 'No default',
                'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
            }
            func_doc['parameters'][param_name] = param_doc
        
        return func_doc
    
    def generate_api_reference(self) -> Dict[str, Any]:
        """
        Генерирует API справочник проекта
        
        Returns:
            Словарь с API справочником
        """
        api_reference = {
            'project_name': 'Nanoprobe Simulation Lab',
            'generated_at': datetime.now().isoformat(),
            'endpoints': [],
            'models': [],
            'utilities': []
        }
        
        # Сканируем API модули
        api_dir = self.project_root / 'api'
        if api_dir.exists():
            for file_path in api_dir.rglob('*.py'):
                if file_path.name != '__init__.py':
                    module_info = self._analyze_python_file(file_path)
                    
                    # Находим endpoint'ы (функции с декораторами Flask)
                    for func_info in module_info.get('functions', []):
                        if any(word in func_info['name'].lower() for word in ['endpoint', 'route', 'api']):
                            api_reference['endpoints'].append({
                                'name': func_info['name'],
                                'file': str(file_path.relative_to(self.project_root)),
                                'docstring': func_info['docstring'],
                                'parameters': func_info.get('parameters', [])
                            })
        
        # Сканируем utility модули
        utils_dir = self.project_root / 'utils'
        if utils_dir.exists():
            for file_path in utils_dir.rglob('*.py'):
                if file_path.name != '__init__.py':
                    module_info = self._analyze_python_file(file_path)
                    
                    # Добавляем классы и функции как утилиты
                    for class_info in module_info.get('classes', []):
                        api_reference['utilities'].append({
                            'type': 'class',
                            'name': class_info['name'],
                            'file': str(file_path.relative_to(self.project_root)),
                            'docstring': class_info['docstring'],
                            'methods': len(class_info.get('methods', []))
                        })
                    
                    for func_info in module_info.get('functions', []):
                        api_reference['utilities'].append({
                            'type': 'function',
                            'name': func_info['name'],
                            'file': str(file_path.relative_to(self.project_root)),
                            'docstring': func_info['docstring'],
                            'parameters': func_info.get('parameters', [])
                        })
        
        return api_reference
    
    def generate_markdown_docs(self, output_dir: str = "docs/generated") -> bool:
        """
        Генерирует документацию в формате Markdown
        
        Args:
            output_dir: Директория для вывода документации
            
        Returns:
            True если генерация успешна, иначе False
        """
        try:
            output_path = self.project_root / output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Генерируем общую документацию
            structure = self.analyze_code_structure()
            
            # Создаем главный файл документации
            main_doc_content = f"""# Документация проекта Nanoprobe Simulation Lab

**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Обзор структуры проекта

Количество директорий: {len(structure['directories'])}
Количество файлов: {len(structure['files'])}
Количество Python файлов: {len(structure['python_files'])}

### Директории:
{chr(10).join(['- ' + d for d in structure['directories'][:10]])}{'' if len(structure['directories']) <= 10 else f'\n- ... и еще {len(structure['directories']) - 10} директорий'}

### Python файлы:
{chr(10).join(['- ' + f for f in structure['python_files'][:10]])}{'' if len(structure['python_files']) <= 10 else f'\n- ... и еще {len(structure['python_files']) - 10} файлов'}

## API Справочник

### Endpoint'ы:
"""
            
            # Добавляем информацию об endpoint'ах
            api_ref = self.generate_api_reference()
            for endpoint in api_ref['endpoints']:
                main_doc_content += f"""
#### {endpoint['name']}
- **Файл:** `{endpoint['file']}`
- **Описание:** {endpoint['docstring'] or 'Нет описания'}
- **Параметры:** {', '.join(endpoint['parameters']) or 'Нет параметров'}
"""
            
            main_doc_content += "\n### Утилиты:\n"
            for utility in api_ref['utilities']:
                main_doc_content += f"""
#### {utility['name']}
- **Тип:** {utility['type']}
- **Файл:** `{utility['file']}`
- **Описание:** {utility['docstring'] or 'Нет описания'}
"""
            
            # Записываем главный файл
            with open(output_path / "main.md", 'w', encoding='utf-8') as f:
                f.write(main_doc_content)
            
            # Генерируем документацию для каждого модуля
            for py_file in structure['python_files']:
                module_path = self.project_root / py_file
                module_doc = self.generate_module_documentation(module_path)
                
                if module_doc:
                    module_doc_content = f"""# Документация модуля: {module_doc['name']}

**Файл:** `{module_doc['file_path']}`

**Описание:**
{module_doc['docstring'] or 'Нет описания'}

"""
                    
                    # Добавляем классы
                    if module_doc['classes']:
                        module_doc_content += "## Классы\n\n"
                        for class_name, class_info in module_doc['classes'].items():
                            module_doc_content += f"""
### {class_name}
- **Описание:** {class_info['docstring'] or 'Нет описания'}
- **Наследование:** {', '.join(class_info['inheritance']) or 'Нет'}
- **Модуль:** `{class_info['module']}`

#### Методы:
"""
                            for method_name, method_info in class_info['methods'].items():
                                module_doc_content += f"""
##### {method_name}
- **Сигнатура:** `{method_info['signature']}`
- **Описание:** {method_info['docstring'] or 'Нет описания'}
- **Возвращает:** {method_info['return_annotation']}
"""
                    
                    # Добавляем функции
                    if module_doc['functions']:
                        module_doc_content += "\n## Функции\n\n"
                        for func_name, func_info in module_doc['functions'].items():
                            module_doc_content += f"""
### {func_name}
- **Сигнатура:** `{func_info['signature']}`
- **Описание:** {func_info['docstring'] or 'Нет описания'}
- **Возвращает:** {func_info['return_annotation']}
- **Модуль:** `{func_info['module']}`
"""
                    
                    # Записываем документацию модуля
                    module_file_name = py_file.replace('/', '_').replace('\\', '_').replace('.py', '.md')
                    with open(output_path / f"module_{module_file_name}", 'w', encoding='utf-8') as f:
                        f.write(module_doc_content)
            
            print(f"✓ Документация в формате Markdown создана: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка генерации Markdown документации: {e}")
            return False
    
    def generate_html_docs(self, output_dir: str = "docs/generated/html") -> bool:
        """
        Генерирует документацию в формате HTML
        
        Args:
            output_dir: Директория для вывода HTML документации
            
        Returns:
            True если генерация успешна, иначе False
        """
        try:
            output_path = self.project_root / output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Шаблон для HTML страницы
            html_template = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        code { background-color: #f1f2f6; padding: 2px 4px; border-radius: 3px; }
        pre { background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px; overflow-x: auto; }
        .section { margin: 20px 0; padding: 15px; background-color: #ecf0f1; border-left: 4px solid #3498db; }
        .nav { background-color: #34495e; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        .nav a { color: white; text-decoration: none; margin-right: 15px; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="index.html">Главная</a>
            <a href="api_reference.html">API Справочник</a>
            <a href="structure.html">Структура проекта</a>
        </div>
        
        <h1>{{ title }}</h1>
        
        {{ content }}
        
        <hr>
        <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
            Сгенерировано автоматически | {{ timestamp }}
        </p>
    </div>
</body>
</html>"""
            
            template = Template(html_template)
            
            # Генерируем главную страницу
            structure = self.analyze_code_structure()
            main_content = f"""
<p>Добро пожаловать в автоматически сгенерированную документацию проекта Nanoprobe Simulation Lab.</p>

<div class="section">
    <h2>Статистика проекта</h2>
    <ul>
        <li><strong>Директорий:</strong> {len(structure['directories'])}</li>
        <li><strong>Файлов:</strong> {len(structure['files'])}</li>
        <li><strong>Python файлов:</strong> {len(structure['python_files'])}</li>
        <li><strong>Проанализировано:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
    </ul>
</div>
"""
            
            main_html = template.render(
                title="Документация проекта Nanoprobe Simulation Lab",
                content=main_content,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            with open(output_path / "index.html", 'w', encoding='utf-8') as f:
                f.write(main_html)
            
            # Генерируем страницу со структурой
            structure_content = f"""
<h2>Директории проекта</h2>
<ul>
{''.join([f'<li><code>{d}</code></li>' for d in structure['directories']])}
</ul>

<h2>Python файлы</h2>
<ul>
{''.join([f'<li><code>{f}</code></li>' for f in structure['python_files']])}
</ul>
"""
            
            structure_html = template.render(
                title="Структура проекта - Документация",
                content=structure_content,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            with open(output_path / "structure.html", 'w', encoding='utf-8') as f:
                f.write(structure_html)
            
            # Генерируем API справочник
            api_ref = self.generate_api_reference()
            api_content = "<h2>Endpoint'ы API</h2>\n"
            
            if api_ref['endpoints']:
                for endpoint in api_ref['endpoints']:
                    api_content += f"""
<div class="section">
    <h3>{endpoint['name']}</h3>
    <p><strong>Файл:</strong> <code>{endpoint['file']}</code></p>
    <p><strong>Описание:</strong> {endpoint['docstring'] or 'Нет описания'}</p>
    <p><strong>Параметры:</strong> {', '.join(endpoint['parameters']) or 'Нет параметров'}</p>
</div>
"""
            else:
                api_content += "<p>Endpoint'ы не найдены.</p>"
            
            api_content += "<h2>Утилиты</h2>\n"
            
            if api_ref['utilities']:
                for utility in api_ref['utilities']:
                    api_content += f"""
<div class="section">
    <h3>{utility['name']}</h3>
    <p><strong>Тип:</strong> {utility['type']}</p>
    <p><strong>Файл:</strong> <code>{utility['file']}</code></p>
    <p><strong>Описание:</strong> {utility['docstring'] or 'Нет описания'}</p>
</div>
"""
            else:
                api_content += "<p>Утилиты не найдены.</p>"
            
            api_html = template.render(
                title="API Справочник - Документация",
                content=api_content,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            with open(output_path / "api_reference.html", 'w', encoding='utf-8') as f:
                f.write(api_html)
            
            print(f"✓ HTML документация создана: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка генерации HTML документации: {e}")
            return False
    
    def generate_api_documentation(self, output_format: str = "markdown") -> bool:
        """
        Генерирует документацию API в заданном формате
        
        Args:
            output_format: Формат вывода ('markdown', 'html', 'json')
            
        Returns:
            True если генерация успешна, иначе False
        """
        api_ref = self.generate_api_reference()
        
        if output_format.lower() == "json":
            output_file = self.project_root / "docs" / "api_reference.json"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(api_ref, f, indent=2, ensure_ascii=False, default=str)
                
        elif output_format.lower() == "markdown":
            output_file = self.project_root / "docs" / "api_reference.md"
            output_file.parent.mkdir(exist_ok=True)
            
            md_content = f"""# API Справочник проекта Nanoprobe Simulation Lab

**Сгенерировано:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Endpoint'ы

"""
            
            for endpoint in api_ref['endpoints']:
                md_content += f"""
### {endpoint['name']}
- **Файл:** `{endpoint['file']}`
- **Описание:** {endpoint['docstring'] or 'Нет описания'}
- **Параметры:** {', '.join(endpoint['parameters']) or 'Нет параметров'}

"""
            
            md_content += "## Утилиты\n\n"
            
            for utility in api_ref['utilities']:
                md_content += f"""
### {utility['name']}
- **Тип:** {utility['type']}
- **Файл:** `{utility['file']}`
- **Описание:** {utility['docstring'] or 'Нет описания'}

"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        elif output_format.lower() == "html":
            return self.generate_html_docs()
        
        print(f"✓ API документация создана в формате {output_format}: {output_file if output_format in ['json', 'markdown'] else 'docs/generated/html'}")
        return True


def main():
    """Главная функция для демонстрации возможностей генератора документации"""
    print("=== ГЕНЕРАТОР ДОКУМЕНТАЦИИ ПРОЕКТА ===")
    
    # Создаем генератор документации
    doc_generator = DocumentationGenerator()
    
    print("✓ Генератор документации инициализирован")
    print(f"✓ Корневая директория: {doc_generator.project_root}")
    
    # Анализируем структуру проекта
    print("\nАнализ структуры проекта...")
    structure = doc_generator.analyze_code_structure()
    print(f"  - Найдено директорий: {len(structure['directories'])}")
    print(f"  - Найдено файлов: {len(structure['files'])}")
    print(f"  - Найдено Python файлов: {len(structure['python_files'])}")
    
    # Генерируем API справочник
    print("\nГенерация API справочника...")
    api_ref = doc_generator.generate_api_reference()
    print(f"  - Найдено endpoint'ов: {len(api_ref['endpoints'])}")
    print(f"  - Найдено утилит: {len(api_ref['utilities'])}")
    
    # Генерируем Markdown документацию
    print("\nГенерация Markdown документации...")
    md_success = doc_generator.generate_markdown_docs()
    print(f"  - Результат: {'Успешно' if md_success else 'Ошибка'}")
    
    # Генерируем HTML документацию
    print("\nГенерация HTML документации...")
    html_success = doc_generator.generate_html_docs()
    print(f"  - Результат: {'Успешно' if html_success else 'Ошибка'}")
    
    # Генерируем JSON API документацию
    print("\nГенерация JSON API документации...")
    json_success = doc_generator.generate_api_documentation("json")
    print(f"  - Результат: {'Успешно' if json_success else 'Ошибка'}")
    
    print("\nГенератор документации успешно протестирован")
    print("\nДоступные функции:")
    print("- Анализ структуры кода: analyze_code_structure()")
    print("- Генерация документации модуля: generate_module_documentation()")
    print("- Генерация API справочника: generate_api_reference()")
    print("- Генерация Markdown документации: generate_markdown_docs()")
    print("- Генерация HTML документации: generate_html_docs()")
    print("- Генерация API документации: generate_api_documentation()")


# Добавляем отсутствующий импорт в начало файла
import importlib.util


if __name__ == "__main__":
    main()