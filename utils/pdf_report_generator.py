# -*- coding: utf-8 -*-
"""
Модуль генерации научных PDF отчётов для проекта Nanoprobe Simulation Lab
Поддержка профессиональных отчётов для научных публикаций
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ScientificPDFReport:
    """
    Генератор научных PDF отчётов для публикаций
    Создаёт профессиональные отчёты с графиками, таблицами и анализом данных
    """

    def __init__(self, output_dir: str = "reports/pdf"):
        """
        Инициализация генератора отчётов

        Args:
            output_dir: Директория для сохранения отчётов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Стили
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Настройка стилей для отчёта"""
        # Заголовок отчёта
        if 'ReportTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a2e'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))

        # Подзаголовок
        if 'Subtitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Subtitle',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#16213e'),
                spaceAfter=12,
                alignment=TA_CENTER
            ))

        # Заголовок раздела
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#0f3460'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold'
            ))

        # Заголовок подраздела
        if 'SubSectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubSectionHeader',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#1a1a2e'),
                spaceAfter=10,
                spaceBefore=15,
                fontName='Helvetica-Bold'
            ))

        # Основной текст
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#333333'),
                alignment=TA_JUSTIFY,
                leading=14
            ))

    def generate_surface_analysis_report(
        self,
        surface_data: Dict[str, Any],
        images: List[str] = None,
        title: str = "Анализ поверхности",
        author: str = "Nanoprobe Simulation Lab",
        organization: str = "Школа программирования Maestro7IT"
    ) -> str:
        """
        Генерация отчёта об анализе поверхности

        Args:
            surface_data: Данные анализа поверхности
            images: Пути к изображениям
            title: Заголовок отчёта
            author: Автор
            organization: Организация

        Returns:
            Путь к созданному PDF файлу
        """
        filename = f"surface_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        content = []

        # Титульная страница
        content.extend(self._create_title_page(title, author, organization))
        content.append(PageBreak())

        # Содержание
        content.append(Paragraph("Содержание", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))

        # Основная информация
        content.append(Paragraph("1. Общая информация", self.styles['SectionHeader']))
        content.extend(self._create_info_table(surface_data))
        content.append(Spacer(1, 0.3*inch))

        # Статистика поверхности
        content.append(Paragraph("2. Статистика поверхности", self.styles['SectionHeader']))
        content.extend(self._create_surface_stats(surface_data))
        content.append(Spacer(1, 0.3*inch))

        # Графики и изображения
        if images:
            content.append(Paragraph("3. Визуализация", self.styles['SectionHeader']))
            content.extend(self._add_images(images))

        # Выводы
        content.append(PageBreak())
        content.append(Paragraph("4. Выводы", self.styles['SectionHeader']))
        content.extend(self._create_conclusions(surface_data))

        # Методология
        content.append(PageBreak())
        content.append(Paragraph("5. Методология", self.styles['SectionHeader']))
        content.append(Paragraph(
            "Анализ поверхности выполнен с использованием методов сканирующей зондовой "
            "микроскопии (СЗМ). Статистические характеристики рассчитаны на основе "
            "трёхмерных данных топографии поверхности.",
            self.styles['BodyText']
        ))

        # Дата и подпись
        content.append(Spacer(1, 1*inch))
        content.append(Paragraph(
            f"<i>Отчёт сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}</i>",
            self.styles['BodyText']
        ))

        doc.build(content)
        return str(filepath)

    def _create_title_page(
        self, title: str, author: str, organization: str
    ) -> List:
        """Создание титульной страницы"""
        content = []

        # Логотип (если есть)
        content.append(Spacer(1, 1*inch))

        # Название организации
        content.append(Paragraph(organization, self.styles['Subtitle']))
        content.append(Spacer(1, 0.5*inch))

        # Заголовок отчёта
        content.append(Paragraph(title, self.styles['ReportTitle']))
        content.append(Spacer(1, 1*inch))

        # Автор
        content.append(Paragraph(f"<b>Автор:</b> {author}", self.styles['BodyText']))
        content.append(Spacer(1, 0.2*inch))

        # Дата
        content.append(Paragraph(
            f"<b>Дата:</b> {datetime.now().strftime('%d.%m.%Y')}",
            self.styles['BodyText']
        ))

        content.append(Spacer(1, 2*inch))

        # Копирайт
        content.append(Paragraph(
            "© 2026 Школа программирования Maestro7IT. Все права защищены.",
            self.styles['BodyText']
        ))

        return content

    def _create_info_table(self, data: Dict[str, Any]) -> List:
        """Создание таблицы с общей информацией"""
        content = []

        table_data = [['Параметр', 'Значение']]

        # Основные параметры
        params = {
            'Тип поверхности': data.get('surface_type', 'Не указано'),
            'Размер области (нм)': data.get('scan_size', 'Не указано'),
            'Разрешение (пиксели)': data.get('resolution', 'Не указано'),
            'Дата сканирования': data.get('scan_date', datetime.now().strftime('%d.%m.%Y')),
            'Метод': data.get('method', 'СЗМ'),
        }

        for key, value in params.items():
            table_data.append([key, str(value)])

        table = Table(table_data, colWidths=[4*cm, 10*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        content.append(table)
        return content

    def _create_surface_stats(self, data: Dict[str, Any]) -> List:
        """Создание раздела со статистикой поверхности"""
        content = []

        # Статистические параметры
        stats = {
            'Средняя высота (Ra)': data.get('mean_height', 0),
            'СКО (Rq)': data.get('std_deviation', 0),
            'Максимальная высота (Rt)': data.get('max_height', 0),
            'Асимметрия (Rsk)': data.get('skewness', 0),
            'Эксцесс (Rku)': data.get('kurtosis', 0),
            'Среднее квадратичное': data.get('rms', 0),
        }

        table_data = [['Параметр', 'Значение', 'Ед. изм.']]

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                table_data.append([key, f"{value:.4f}", 'нм'])
            else:
                table_data.append([key, str(value), '-'])

        table = Table(table_data, colWidths=[5*cm, 6*cm, 3*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8f8')]),
        ]))

        content.append(table)
        return content

    def _add_images(self, images: List[str]) -> List:
        """Добавление изображений в отчёт"""
        content = []

        for i, img_path in enumerate(images, 1):
            try:
                img = Image(img_path, width=14*cm, height=10*cm)
                content.append(img)
                content.append(Paragraph(
                    f"<i>Рисунок {i}: {Path(img_path).stem}</i>",
                    self.styles['BodyText']
                ))
                content.append(Spacer(1, 0.3*inch))
            except Exception as e:
                content.append(Paragraph(
                    f"<i>Изображение не загружено: {img_path} ({e})</i>",
                    self.styles['BodyText']
                ))

        return content

    def _create_conclusions(self, data: Dict[str, Any]) -> List:
        """Создание раздела с выводами"""
        content = []

        conclusions = []

        # Автоматическая генерация выводов на основе данных
        if 'mean_height' in data:
            mean = data.get('mean_height', 0)
            if abs(mean) < 10:
                conclusions.append("• Поверхность имеет низкий рельеф с средней высотой менее 10 нм")
            elif abs(mean) < 100:
                conclusions.append("• Поверхность имеет средний рельеф с высотой от 10 до 100 нм")
            else:
                conclusions.append("• Поверхность имеет высокий рельеф с высотой более 100 нм")

        if 'std_deviation' in data:
            std = data.get('std_deviation', 0)
            if std < 5:
                conclusions.append("• Поверхность характеризуется высокой однородностью")
            elif std < 20:
                conclusions.append("• Поверхность имеет умеренную неоднородность")
            else:
                conclusions.append("• Поверхность характеризуется высокой неоднородностью")

        if 'skewness' in data:
            skew = data.get('skewness', 0)
            if skew > 0:
                conclusions.append("• Преобладают выступы над впадинами (положительная асимметрия)")
            elif skew < 0:
                conclusions.append("• Преобладают впадины над выступами (отрицательная асимметрия)")
            else:
                conclusions.append("• Симметричное распределение высот")

        if not conclusions:
            conclusions.append("• Требуется дополнительный анализ данных")

        for conclusion in conclusions:
            content.append(Paragraph(conclusion, self.styles['BodyText']))
            content.append(Spacer(1, 0.1*inch))

        return content

    def generate_defect_analysis_report(
        self,
        defect_data: Dict[str, Any],
        defect_images: List[str] = None,
        title: str = "Анализ дефектов поверхности",
        author: str = "Nanoprobe Simulation Lab",
        organization: str = "Школа программирования Maestro7IT"
    ) -> str:
        """
        Генерация отчёта об анализе дефектов

        Args:
            defect_data: Данные о дефектах
            defect_images: Изображения дефектов
            title: Заголовок
            author: Автор
            organization: Организация

        Returns:
            Путь к PDF файлу
        """
        filename = f"defect_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        content = []

        # Титульная страница
        content.extend(self._create_title_page(title, author, organization))
        content.append(PageBreak())

        # Общая информация
        content.append(Paragraph("1. Общая информация", self.styles['SectionHeader']))
        
        info_data = [
            ['Параметр', 'Значение'],
            ['Количество дефектов', str(defect_data.get('defects_count', 0))],
            ['Тип анализа', defect_data.get('analysis_type', 'AI/ML')],
            ['Модель', defect_data.get('model_name', 'Не указано')],
            ['Достоверность', f"{defect_data.get('confidence', 0):.2%}"],
            ['Время анализа', f"{defect_data.get('processing_time', 0):.2f} сек"],
        ]

        table = Table(info_data, colWidths=[5*cm, 9*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        content.append(table)
        content.append(Spacer(1, 0.3*inch))

        # Детали дефектов
        content.append(Paragraph("2. Детектированные дефекты", self.styles['SectionHeader']))
        
        defects = defect_data.get('defects', [])
        if defects:
            defect_table_data = [['#', 'Тип', 'Координаты', 'Размер (нм)', 'Достоверность']]
            
            for i, defect in enumerate(defects, 1):
                defect_table_data.append([
                    str(i),
                    defect.get('type', 'Неизвестно'),
                    f"({defect.get('x', 0):.1f}, {defect.get('y', 0):.1f})",
                    f"{defect.get('size', 0):.2f}",
                    f"{defect.get('confidence', 0):.2%}"
                ])
            
            defect_table = Table(defect_table_data, colWidths=[0.8*cm, 3*cm, 3.5*cm, 2.5*cm, 2.5*cm])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            content.append(defect_table)
        else:
            content.append(Paragraph("Дефекты не обнаружены", self.styles['BodyText']))

        content.append(Spacer(1, 0.3*inch))

        # Изображения
        if defect_images:
            content.append(Paragraph("3. Визуализация дефектов", self.styles['SectionHeader']))
            content.extend(self._add_images(defect_images))

        # Выводы
        content.append(PageBreak())
        content.append(Paragraph("3. Рекомендации", self.styles['SectionHeader']))
        
        if defects:
            content.append(Paragraph(
                "• Рекомендуется провести дополнительный анализ дефектных областей",
                self.styles['BodyText']
            ))
            content.append(Paragraph(
                "• Возможно требуется корректировка технологического процесса",
                self.styles['BodyText']
            ))
        else:
            content.append(Paragraph(
                "• Поверхность соответствует требованиям качества",
                self.styles['BodyText']
            ))

        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph(
            f"<i>Отчёт сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}</i>",
            self.styles['BodyText']
        ))

        doc.build(content)
        return str(filepath)


# Глобальная функция для быстрой генерации отчётов
def generate_pdf_report(
    report_type: str,
    data: Dict[str, Any],
    images: List[str] = None,
    output_dir: str = "reports/pdf"
) -> str:
    """
    Быстрая генерация PDF отчёта

    Args:
        report_type: Тип отчёта ('surface', 'defect', 'comparison')
        data: Данные для отчёта
        images: Изображения
        output_dir: Директория вывода

    Returns:
        Путь к PDF файлу
    """
    generator = ScientificPDFReport(output_dir)

    if report_type == 'surface':
        return generator.generate_surface_analysis_report(data, images)
    elif report_type == 'defect':
        return generator.generate_defect_analysis_report(data, images)
    else:
        raise ValueError(f"Неизвестный тип отчёта: {report_type}")


if __name__ == "__main__":
    # Тестовая генерация отчёта
    print("=== Генерация тестового PDF отчёта ===")
    
    # Тестовые данные
    test_data = {
        'surface_type': 'Кремниевая подложка',
        'scan_size': '10x10 мкм',
        'resolution': '512x512',
        'mean_height': 12.345,
        'std_deviation': 3.456,
        'max_height': 45.678,
        'skewness': 0.123,
        'kurtosis': 2.345,
        'rms': 3.567,
    }
    
    # Генерация отчёта
    report_path = generate_pdf_report('surface', test_data)
    print(f"✓ Отчёт сгенерирован: {report_path}")
