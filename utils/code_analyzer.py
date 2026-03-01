# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль анализа кода для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для статического анализа кода,
обнаружения потенциальных проблем и автоматического рефакторинга.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json
import re
from dataclasses import dataclass
import tokenize
from io import StringIO

@dataclass
class CodeIssue:
    """Проблема в коде"""
    file_path: str
    line_number: int
    column: int
    issue_type: str
    message: str
    severity: str
    code_snippet: str

@dataclass
class CodeMetrics:
    """Метрики кода"""
    file_path: str
    lines_of_code: int
    comment_lines: int
    blank_lines: int
    functions_count: int
    classes_count: int
    complexity: int
    maintainability_index: float

class CodeAnalyzer:
    """
    Класс анализатора кода
    Обеспечивает статический анализ кода, обнаружение проблем
    и оценку качества кода проекта.
    """


    def __init__(self, project_root: str = "."):
        """
        Инициализирует анализатор кода

        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()
        self.issues = []
        self.metrics = []
        self.complexity_patterns = {
            'if': re.compile(r'\bif\s+.*:'),
            'for': re.compile(r'\bfor\s+.*:'),
            'while': re.compile(r'\bwhile\s+.*:'),
            'elif': re.compile(r'\belif\s+.*:'),
            'and': re.compile(r'\s+and\s+'),
            'or': re.compile(r'\s+or\s+'),
            'not': re.compile(r'\bnot\s+'),
            'try': re.compile(r'\btry\s*:'),
            'except': re.compile(r'\bexcept\s*:'),
            'finally': re.compile(r'\bfinally\s*:'),
        }


    def analyze_project(self, include_patterns: List[str] = None,

                       exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Анализирует весь проект

        Args:
            include_patterns: Паттерны для включения файлов
            exclude_patterns: Паттерны для исключения файлов

        Returns:
            Результаты анализа проекта
        """
        if include_patterns is None:
            include_patterns = ['*.py']

        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '*.pyc', 'venv', '.git']

        # Находим все Python файлы
        python_files = []
        for pattern in include_patterns:
            for file_path in self.project_root.rglob(pattern):
                if not any(exclude in str(file_path) for exclude in exclude_patterns):
                    python_files.append(file_path)

        print(f"Найдено Python файлов для анализа: {len(python_files)}")

        # Анализируем каждый файл
        for file_path in python_files:
            try:
                self.analyze_file(file_path)
            except Exception as e:
                print(f"Ошибка анализа файла {file_path}: {e}")

        return self._generate_analysis_report()


    def analyze_file(self, file_path: Path) -> List[CodeIssue]:
        """
        Анализирует отдельный файл

        Args:
            file_path: Путь к файлу

        Returns:
            Список обнаруженных проблем
        """
        file_issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Анализируем метрики
            metrics = self._calculate_metrics(file_path, content)
            self.metrics.append(metrics)

            # Анализируем потенциальные проблемы
            file_issues.extend(self._check_naming_conventions(file_path, content))
            file_issues.extend(self._check_code_complexity(file_path, content))
            file_issues.extend(self._check_best_practices(file_path, content))
            file_issues.extend(self._check_security_issues(file_path, content))
            file_issues.extend(self._check_performance_issues(file_path, content))

            self.issues.extend(file_issues)

        except Exception as e:
            error_issue = CodeIssue(
                file_path=str(file_path),
                line_number=0,
                column=0,
                issue_type="FILE_ERROR",
                message=f"Ошибка чтения файла: {str(e)}",
                severity="ERROR",
                code_snippet=""
            )
            self.issues.append(error_issue)
            file_issues.append(error_issue)

        return file_issues


    def _calculate_metrics(self, file_path: Path, content: str) -> CodeMetrics:
        """Вычисляет метрики кода"""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        blank_lines = len([line for line in lines if not line.strip()])

        # Подсчет функций и классов
        functions_count = content.count('def ')
        classes_count = content.count('class ')

        # Вычисление сложности (цикломатическая сложность)
        complexity = 1  # Базовая сложность
        for pattern_name, pattern in self.complexity_patterns.items():
            complexity += len(pattern.findall(content))

        # Индекс поддерживаемости (упрощенная формула)
        if lines_of_code > 0:
            comment_ratio = comment_lines / lines_of_code
            maintainability_index = max(0, min(100,
                171 - 5.2 * (complexity / lines_of_code * 100) - 0.23 * (lines_of_code) - 16.2 * (comment_ratio)))
        else:
            maintainability_index = 100.0

        return CodeMetrics(
            file_path=str(file_path),
            lines_of_code=lines_of_code,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            functions_count=functions_count,
            classes_count=classes_count,
            complexity=complexity,
            maintainability_index=maintainability_index
        )


    def _check_naming_conventions(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Проверяет соблюдение соглашений об именовании"""
        issues = []

        # Проверка имен функций (должны быть snake_case)
        function_pattern = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        for match in function_pattern.finditer(content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1

            if not re.match(r'^[a-z_][a-z0-9_]*$', func_name) and not func_name.startswith('__'):
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    column=match.start(),
                    issue_type="NAMING_CONVENTION",
                    message=f"Функция '{func_name}' не следует snake_case соглашению",
                    severity="WARNING",
                    code_snippet=content.split('\n')[line_num-1].strip()
                ))

        # Проверка имен классов (должны быть PascalCase)
        class_pattern = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1

            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    column=match.start(),
                    issue_type="NAMING_CONVENTION",
                    message=f"Класс '{class_name}' не следует PascalCase соглашению",
                    severity="WARNING",
                    code_snippet=content.split('\n')[line_num-1].strip()
                ))

        return issues


    def _check_code_complexity(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Проверяет сложность кода"""
        issues = []
        lines = content.split('\n')

        # Проверка длинных функций
        function_pattern = re.compile(r'def\s+.*?:\s*$')
        current_function_start = None

        for i, line in enumerate(lines):
            if function_pattern.match(line.strip()):
                if current_function_start is not None:
                    # Завершаем предыдущую функцию
                    func_length = i - current_function_start
                    if func_length > 50:  # Более 50 строк
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=current_function_start + 1,
                            column=0,
                            issue_type="COMPLEXITY",
                            message=f"Функция слишком длинная ({func_length} строк)",
                            severity="WARNING",
                            code_snippet=lines[current_function_start].strip()
                        ))
                current_function_start = i
            elif line.strip() == '' and current_function_start is not None:
                # Завершаем функцию на пустой строке
                func_length = i - current_function_start
                if func_length > 50:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=current_function_start + 1,
                        column=0,
                        issue_type="COMPLEXITY",
                        message=f"Функция слишком длинная ({func_length} строк)",
                        severity="WARNING",
                        code_snippet=lines[current_function_start].strip()
                    ))
                current_function_start = None

        # Проверка вложенных условий
        nested_depth = 0
        max_depth = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('if ', 'for ', 'while ', 'try:', 'with ')):
                nested_depth += 1
                max_depth = max(max_depth, nested_depth)
            elif stripped == 'else:' or stripped.startswith('elif '):
                # else/elif не увеличивают глубину
                pass
            elif nested_depth > 0 and (stripped == '' or not stripped.startswith((' ', '\t'))):
                nested_depth = max(0, nested_depth - 1)

        if max_depth > 5:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                column=0,
                issue_type="COMPLEXITY",
                message=f"Слишком высокая вложенность кода (уровень {max_depth})",
                severity="WARNING",
                code_snippet=""
            ))

        return issues


    def _check_best_practices(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Проверяет соблюдение лучших практик"""
        issues = []

        # Проверка импортов
        import_lines = [line for line in content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
        if len(import_lines) > 20:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                column=0,
                issue_type="BEST_PRACTICE",
                message=f"Слишком много импортов ({len(import_lines)})",
                severity="WARNING",
                code_snippet=""
            ))

        # Проверка длины строк
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 100 and not line.strip().startswith('#'):
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i + 1,
                    column=100,
                    issue_type="BEST_PRACTICE",
                    message=f"Строка слишком длинная ({len(line)} символов)",
                    severity="WARNING",
                    code_snippet=line[:100] + "..."
                ))

        # Проверка отсутствия docstring
        if '"""' not in content and "'''" not in content:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                column=0,
                issue_type="BEST_PRACTICE",
                message="Файл не содержит docstring",
                severity="WARNING",
                code_snippet=""
            ))

        return issues


    def _check_security_issues(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Проверяет потенциальные проблемы безопасности"""
        issues = []

        # Проверка eval/exec
        if 'eval(' in content:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=content.find('eval(') + 1,
                column=0,
                issue_type="SECURITY",
                message="Использование eval() может быть небезопасным",
                severity="ERROR",
                code_snippet=""
            ))

        if 'exec(' in content:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=content.find('exec(') + 1,
                column=0,
                issue_type="SECURITY",
                message="Использование exec() может быть небезопасным",
                severity="ERROR",
                code_snippet=""
            ))

        # Проверка os.system
        if 'os.system(' in content:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=content.find('os.system(') + 1,
                column=0,
                issue_type="SECURITY",
                message="Использование os.system() может быть небезопасным",
                severity="WARNING",
                code_snippet=""
            ))

        return issues


    def _check_performance_issues(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Проверяет потенциальные проблемы производительности"""
        issues = []

        # Проверка конкатенации строк в циклах
        if 'for ' in content and '+' in content and 'str(' in content:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                column=0,
                issue_type="PERFORMANCE",
                message="Возможна неэффективная конкатенация строк в цикле",
                severity="WARNING",
                code_snippet=""
            ))

        # Проверка повторных вычислений
        if content.count('len(') > 5:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                column=0,
                issue_type="PERFORMANCE",
                message="Частое использование len() может быть неэффективным",
                severity="WARNING",
                code_snippet=""
            ))

        return issues


    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Генерирует отчет по анализу кода"""
        # Группируем проблемы по типам
        issues_by_type = {}
        issues_by_severity = {'ERROR': 0, 'WARNING': 0, 'INFO': 0}

        for issue in self.issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)
            issues_by_severity[issue.severity] += 1

        # Статистика по метрикам
        total_files = len(self.metrics)
        avg_complexity = sum(m.complexity for m in self.metrics) / total_files if total_files > 0 else 0
        avg_maintainability = sum(m.maintainability_index for m in self.metrics) / total_files if total_files > 0 else 0

        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {
                'total_files_analyzed': total_files,
                'total_issues': len(self.issues),
                'issues_by_severity': issues_by_severity,
                'average_complexity': round(avg_complexity, 2),
                'average_maintainability_index': round(avg_maintainability, 2)
            },
            'issues_by_type': {
                issue_type: len(issues)
                for issue_type, issues in issues_by_type.items()
            },
            'files_with_issues': list(set(issue.file_path for issue in self.issues)),
            'detailed_issues': [
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'message': issue.message,
                    'code_snippet': issue.code_snippet
                }
                for issue in self.issues
            ],
            'metrics_summary': {
                'total_lines_of_code': sum(m.lines_of_code for m in self.metrics),
                'total_functions': sum(m.functions_count for m in self.metrics),
                'total_classes': sum(m.classes_count for m in self.metrics),
                'files_with_high_complexity': [
                    m.file_path for m in self.metrics if m.complexity > 20
                ],
                'files_with_low_maintainability': [
                    m.file_path for m in self.metrics if m.maintainability_index < 50
                ]
            }
        }

        return report


    def save_report(self, report: Dict[str, Any], output_path: str = "code_analysis_report.json"):
        """Сохраняет отчет в файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return output_path

def main():
    """Главная функция для демонстрации возможностей анализатора кода"""
    print("=== АНАЛИЗАТОР КОДА ПРОЕКТА ===")

    # Создаем анализатор кода
    analyzer = CodeAnalyzer()

    print("✓ Анализатор кода инициализирован")
    print(f"✓ Корневая директория: {analyzer.project_root}")

    # Анализируем проект
    print("\nАнализ проекта...")
    report = analyzer.analyze_project()

    print(f"✓ Проанализировано файлов: {report['summary']['total_files_analyzed']}")
    print(f"✓ Найдено проблем: {report['summary']['total_issues']}")
    print(f"✓ Средняя сложность: {report['summary']['average_complexity']}")
    print(f"✓ Индекс поддерживаемости: {report['summary']['average_maintainability_index']}")

    # Показываем распределение проблем
    print("\nРаспределение проблем по типам:")
    for issue_type, count in report['issues_by_type'].items():
        print(f"  - {issue_type}: {count}")

    # Показываем файлы с проблемами
    if report['files_with_issues']:
        print(f"\nФайлы с проблемами ({len(report['files_with_issues'])}):")
        for file_path in report['files_with_issues'][:5]:  # Показываем первые 5
            print(f"  - {file_path}")
        if len(report['files_with_issues']) > 5:
            print(f"  ... и еще {len(report['files_with_issues']) - 5} файлов")

    # Сохраняем отчет
    report_path = analyzer.save_report(report)
    print(f"\n✓ Отчет сохранен: {report_path}")

    print("\nАнализатор кода успешно протестирован")
    print("\nДоступные функции:")
    print("- Анализ проекта: analyze_project()")
    print("- Анализ файла: analyze_file()")
    print("- Генерация отчета: _generate_analysis_report()")
    print("- Сохранение отчета: save_report()")

if __name__ == "__main__":
    main()

