# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль генерации отчетов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для создания
комплексных отчетов о симуляциях и анализах.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import pdfkit
import html

class ReportGenerator:
    """
    Класс для генерации отчетов
    Создает комплексные отчеты о симуляциях, анализах и
    результатах работы всех компонентов проекта.
    """


    def __init__(self, output_dir: str = "reports"):
        """
        Инициализирует генератор отчетов

        Args:
            output_dir: Директория для сохранения отчетов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # HTML шаблон для отчета
        self.html_template = Template("""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p><strong>Дата генерации:</strong> {{ timestamp }}</p>

        {% if summary %}
        <h2>Общий обзор</h2>
        <div class="metrics-grid">
            {% for key, value in summary.items() %}
            <div class="metric-card">
                <strong>{{ key }}:</strong> {{ value }}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if surface_analysis %}
        <h2>Анализ поверхности</h2>
        <table>
            <tr>
                <th>Метрика</th>
                <th>Значение</th>
            </tr>
            {% for key, value in surface_analysis.items() %}
            <tr>
                <td>{{ key }}</td>
                <td>{{ "%.4f"|format(value) if value is number else value }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if image_analysis %}
        <h2>Анализ изображений</h2>
        <table>
            <tr>
                <th>Признак</th>
                <th>Значение</th>
            </tr>
            {% for key, value in image_analysis.items() %}
            <tr>
                <td>{{ key }}</td>
                <td>{{ "%.4f"|format(value) if value is number else value }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if sstv_analysis %}
        <h2>Анализ SSTV сигналов</h2>
        <table>
            <tr>
                <th>Метрика</th>
                <th>Значение</th>
            </tr>
            {% for key, value in sstv_analysis.items() %}
            <tr>
                <td>{{ key }}</td>
                <td>{{ "%.4f"|format(value) if value is number else value }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if charts %}
        <h2>Визуализации</h2>
        {% for chart_path in charts %}
        <div class="chart-container">
            <img src="{{ chart_path }}" alt="Визуализация" style="max-width: 100%; height: auto;">
        </div>
        {% endfor %}
        {% endif %}

        {% if timeline %}
        <h2>Хронология событий</h2>
        <table>
            <tr>
                <th>Время</th>
                <th>Событие</th>
                <th>Компонент</th>
                <th>Уровень</th>
            </tr>
            {% for event in timeline %}
            <tr>
                <td>{{ event.timestamp }}</td>
                <td>{{ event.message }}</td>
                <td>{{ event.component }}</td>
                <td>{{ event.level }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        <div class="footer">
            <p>Сгенерировано автоматически системой Лаборатории моделирования нанозонда</p>
            <p>© {{ year }} Школа программирования Maestro7IT. Все права защищены.</p>
        </div>
    </div>
</body>
</html>
        """)

    def generate_simulation_report(self,
    """TODO: Add description"""

                                 simulation_data: Dict[str, Any],
                                 title: str = "Отчет о симуляции",
                                 include_charts: bool = True) -> str:
        """
        Генерирует отчет о симуляции

        Args:
            simulation_data: Данные симуляции
            title: Заголовок отчета
            include_charts: Включать ли диаграммы

        Returns:
            Путь к созданному отчету
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Подготовка данных для шаблона
        template_data = {
            'title': title,
            'timestamp': timestamp,
            'year': datetime.now().year,
            'summary': simulation_data.get('summary', {}),
            'surface_analysis': simulation_data.get('surface_analysis', {}),
            'image_analysis': simulation_data.get('image_analysis', {}),
            'sstv_analysis': simulation_data.get('sstv_analysis', {}),
            'charts': simulation_data.get('charts', []),
            'timeline': simulation_data.get('timeline', [])
        }

        # Генерация HTML
        html_content = self.html_template.render(**template_data)

        # Сохранение отчета
        report_filename = f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML отчет сохранен: {report_path}")

        # Также создаем PDF версию
        pdf_filename = report_path.with_suffix('.pdf')
        try:
            pdfkit.from_string(html_content, str(pdf_filename))
            print(f"PDF отчет сохранен: {pdf_filename}")
        except Exception as e:
            print(f"Ошибка создания PDF: {e}")

        return str(report_path)

    """TODO: Add description"""

    def generate_analytics_report(self,
                                analytics_data: Dict[str, Any],
                                title: str = "Аналитический отчет") -> str:
        """
        Генерирует аналитический отчет

        Args:
            analytics_data: Данные аналитики
            title: Заголовок отчета

        Returns:
            Путь к созданному отчету
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Подготовка данных для шаблона
        template_data = {
            'title': title,
            'timestamp': timestamp,
            'year': datetime.now().year,
            'summary': analytics_data.get('summary', {}),
            'surface_analysis': analytics_data.get('surface_analysis', {}),
            'image_analysis': analytics_data.get('image_analysis', {}),
            'sstv_analysis': analytics_data.get('sstv_analysis', {}),
            'charts': analytics_data.get('charts', []),
            'timeline': analytics_data.get('timeline', [])
        }

        # Генерация HTML
        html_content = self.html_template.render(**template_data)

        # Сохранение отчета
        report_filename = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Аналитический отчет сохранен: {report_path}")

        # Создаем PDF версию
        pdf_filename = report_path.with_suffix('.pdf')
        try:
            pdfkit.from_string(html_content, str(pdf_filename))
            print(f"PDF аналитический отчет сохранен: {pdf_filename}")
        except Exception as e:
            print(f"Ошибка создания PDF: {e}")

        return str(report_path)
    """TODO: Add description"""

    def generate_comparison_report(self,
                                 reports_data: List[Dict[str, Any]],
                                 title: str = "Сравнительный отчет") -> str:
        """
        Генерирует сравнительный отчет

        Args:
            reports_data: Список данных для сравнения
            title: Заголовок отчета

        Returns:
            Путь к созданному отчету
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Подготовка данных для сравнения
        comparison_data = {
            'title': title,
            'timestamp': timestamp,
            'year': datetime.now().year,
            'comparisons': []
        }

        for i, report in enumerate(reports_data):
            comparison_data['comparisons'].append({
                'id': i + 1,
                'data': report
            })

        # Создание специального шаблона для сравнительного отчета
        comparison_template = Template("""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .comparison-section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        .highlight {
            background-color: #e8f5e8;
        }
        .difference {
            background-color: #ffe8e8;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p><strong>Дата генерации:</strong> {{ timestamp }}</p>

        {% for comp in comparisons %}
        <div class="comparison-section">
            <h2>Сравнение #{{ comp.id }}</h2>

            {% if comp.data.summary %}
            <h3>Общий обзор</h3>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                </tr>
                {% for key, value in comp.data.summary.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ "%.4f"|format(value) if value is number else value }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}

            {% if comp.data.surface_analysis %}
            <h3>Анализ поверхности</h3>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                </tr>
                {% for key, value in comp.data.surface_analysis.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ "%.4f"|format(value) if value is number else value }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>
        {% endfor %}

        <div class="footer">
            <p>Сгенерировано автоматически системой Лаборатории моделирования нанозонда</p>
            <p>© {{ year }} Школа программирования Maestro7IT. Все права защищены.</p>
        </div>
    </div>
</body>
</html>
        """)

        html_content = comparison_template.render(**comparison_data)

        # Сохранение отчета
        report_filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Сравнительный отчет сохранен: {report_path}")

        # Создаем PDF версию
        pdf_filename = report_path.with_suffix('.pdf')
        try:
            pdfkit.from_string(html_content, str(pdf_filename))
            print(f"PDF сравнительный отчет сохранен: {pdf_filename}")
        except Exception as e:
            print(f"Ошибка создания PDF: {e}")

        return str(report_path)


    def create_summary_statistics(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Создает сводную статистику из списка данных

        Args:
            data_list: Список словарей с данными

        Returns:
            Словарь с сводной статистикой
        """
        if not data_list:
            return {}

        # Объединяем все ключи
        all_keys = set()
        for data in data_list:
            all_keys.update(data.keys())

        summary = {}
        for key in all_keys:
            values = []
            for data in data_list:
                if key in data and isinstance(data[key], (int, float)):
                    values.append(data[key])

            if values:
                summary[f"{key}_mean"] = round(sum(values) / len(values), 4)
                summary[f"{key}_min"] = round(min(values), 4)
                summary[f"{key}_max"] = round(max(values), 4)
                summary[f"{key}_count"] = len(values)

        return summary

def main():
    """Главная функция для демонстрации возможностей генератора отчетов"""
    print("=== ГЕНЕРАТОР ОТЧЕТОВ ПРОЕКТА ===")

    # Создаем генератор отчетов
    report_gen = ReportGenerator()

    # Создаем тестовые данные
    test_simulation_data = {
        'summary': {
            'Время выполнения': '15.23 сек',
            'Компоненты использованы': 'СЗМ, Визуализация',
            'Статус': 'Успешно'
        },
        'surface_analysis': {
            'Средняя высота': 0.1234,
            'Стандартное отклонение': 0.0567,
            'Минимальная высота': -0.2341,
            'Максимальная высота': 0.4567
        },
        'image_analysis': {
            'Средняя интенсивность': 0.7890,
            'Контраст': 0.1234,
            'Энтропия': 5.6789
        },
        'sstv_analysis': {
            'SNR': 25.67,
            'Доминирующая частота': 1200.0
        },
        'charts': [],
        'timeline': [
            {'timestamp': '2023-12-01 10:00:00', 'message': 'Начало симуляции', 'component': 'СЗМ', 'level': 'INFO'},
            {'timestamp': '2023-12-01 10:00:05', 'message': 'Создание поверхности', 'component': 'СЗМ', 'level': 'INFO'},
            {'timestamp': '2023-12-01 10:00:15', 'message': 'Сканирование завершено', 'component': 'СЗМ', 'level': 'INFO'}
        ]
    }

    # Генерируем отчет
    report_path = report_gen.generate_simulation_report(
        test_simulation_data,
        "Тестовый отчет о симуляции"
    )

    print(f"✓ Отчет успешно сгенерирован: {report_path}")

    # Создаем сводную статистику
    test_data_list = [
        {'metric1': 1.0, 'metric2': 2.0},
        {'metric1': 1.2, 'metric2': 2.1},
        {'metric1': 0.9, 'metric2': 1.9}
    ]

    summary_stats = report_gen.create_summary_statistics(test_data_list)
    print(f"✓ Сводная статистика: {summary_stats}")

    print("Генератор отчетов успешно протестирован")

if __name__ == "__main__":
    main()

