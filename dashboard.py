#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивная панель управления проектом Лаборатория моделирования нанозонда
Этот модуль предоставляет графический интерфейс для управления всеми компонентами проекта.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from utils.config_manager import ConfigManager
from utils.logger import setup_project_logging
from utils.data_manager import DataManager
from utils.visualizer import ProjectVisualizer
from utils.simulator_orchestrator import SimulationOrchestrator


class NanoprobeDashboard:
    """
    Класс интерактивной панели управления проектом
    Предоставляет графический интерфейс для управления всеми компонентами проекта Лаборатории моделирования нанозонда.
    """
    
    def __init__(self):
        """Инициализирует панель управления"""
        self.root = tk.Tk()
        self.root.title("Лаборатория моделирования нанозонда - Панель управления")
        self.root.geometry("1200x800")
        
        # Инициализируем компоненты
        self.config_manager = ConfigManager()
        self.logger_manager = setup_project_logging(self.config_manager)
        self.data_manager = DataManager()
        self.visualizer = ProjectVisualizer()
        self.orchestrator = SimulationOrchestrator(self.config_manager)
        
        # Состояния
        self.simulation_running = False
        self.background_thread = None
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Логируем запуск
        self.logger_manager.log_system_event("Панель управления запущена", "INFO")
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        # Главный фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем вкладки
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка управления
        self.create_control_tab(notebook)
        
        # Вкладка визуализации
        self.create_visualization_tab(notebook)
        
        # Вкладка логов
        self.create_logs_tab(notebook)
        
        # Вкладка настроек
        self.create_settings_tab(notebook)
    
    def create_control_tab(self, parent):
        """Создает вкладку управления"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Управление")
        
        # Левая часть - элементы управления
        left_frame = ttk.LabelFrame(frame, text="Управление компонентами", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Кнопки управления
        ttk.Button(left_frame, text="Запустить СЗМ симуляцию", 
                  command=self.run_spm_simulation).pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Запустить анализ изображений", 
                  command=self.run_image_analysis).pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Запустить SSTV декодирование", 
                  command=self.run_sstv_decoding).pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Комплексная симуляция", 
                  command=self.run_comprehensive_simulation).pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Непрерывная симуляция", 
                  command=self.run_continuous_simulation).pack(fill=tk.X, pady=5)
        
        ttk.Button(left_frame, text="Остановить симуляцию", 
                  command=self.stop_simulation).pack(fill=tk.X, pady=5)
        
        # Правая часть - информация о состоянии
        right_frame = ttk.LabelFrame(frame, text="Информация о состоянии", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Текстовое поле для информации
        self.status_text = scrolledtext.ScrolledText(right_frame, height=20)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем начальную информацию
        self.update_status_info()
    
    def create_visualization_tab(self, parent):
        """Создает вкладку визуализации"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Визуализация")
        
        # Фрейм для элементов управления
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Загрузить и отобразить данные", 
                  command=self.load_and_visualize_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Создать отчет визуализации", 
                  command=self.create_visualization_report).pack(side=tk.LEFT, padx=5)
        
        # Фрейм для canvas
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_logs_tab(self, parent):
        """Создает вкладку логов"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Логи")
        
        # Текстовое поле для логов
        self.logs_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопка обновления логов
        ttk.Button(frame, text="Обновить логи", 
                  command=self.refresh_logs).pack(pady=5)
    
    def create_settings_tab(self, parent):
        """Создает вкладку настроек"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Настройки")
        
        # Настройки проекта
        project_frame = ttk.LabelFrame(frame, text="Настройки проекта", padding=10)
        project_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(project_frame, text="Директория данных:").pack(anchor=tk.W)
        self.data_dir_var = tk.StringVar(value=self.config_manager.get("paths.data_dir", "data"))
        ttk.Entry(project_frame, textvariable=self.data_dir_var).pack(fill=tk.X, pady=5)
        
        ttk.Label(project_frame, text="Директория вывода:").pack(anchor=tk.W)
        self.output_dir_var = tk.StringVar(value=self.config_manager.get("paths.output_dir", "output"))
        ttk.Entry(project_frame, textvariable=self.output_dir_var).pack(fill=tk.X, pady=5)
        
        # Кнопка сохранения настроек
        ttk.Button(project_frame, text="Сохранить настройки", 
                  command=self.save_settings).pack(pady=10)
    
    def update_status_info(self):
        """Обновляет информацию о состоянии"""
        self.status_text.delete(1.0, tk.END)
        
        # Добавляем информацию о состоянии
        status_info = f"""ИНФОРМАЦИЯ О СОСТОЯНИИ ПРОЕКТА
{'='*40}
Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Симуляция запущена: {'Да' if self.simulation_running else 'Нет'}
Компоненты инициализированы: Да
Последнее обновление: {datetime.now().strftime('%H:%M:%S')}

КОМПОНЕНТЫ ПРОЕКТА:
- СЗМ симулятор: {'Инициализирован' if self.orchestrator.spm_controller else 'Не инициализирован'}
- Анализатор изображений: {'Инициализирован' if self.orchestrator.image_processor else 'Не инициализирован'}
- SSTV декодер: {'Инициализирован' if self.orchestrator.sstv_decoder else 'Не инициализирован'}

ПУТИ:
- Данные: {self.data_manager.data_dir}
- Вывод: {self.data_manager.output_dir}
"""
        
        self.status_text.insert(tk.END, status_info)
        self.status_text.see(tk.END)
    
    def run_spm_simulation(self):
        """Запускает симуляцию СЗМ"""
        def worker():
            try:
                self.logger_manager.log_system_event("Запуск СЗМ симуляции из GUI", "INFO")
                
                # Создаем поверхность и запускаем симуляцию
                surface = self.orchestrator.create_simulation_surface((30, 30))
                results = self.orchestrator.run_spm_simulation(surface)
                
                messagebox.showinfo("Симуляция СЗМ", f"Симуляция завершена!\nДлительность: {results['duration']:.2f} сек")
                self.logger_manager.log_spm_event("СЗМ симуляция завершена успешно", "INFO")
                
            except Exception as e:
                self.logger_manager.log_spm_event(f"Ошибка СЗМ симуляции: {e}", "ERROR")
                messagebox.showerror("Ошибка", f"Ошибка СЗМ симуляции: {e}")
            finally:
                self.update_status_info()
        
        threading.Thread(target=worker, daemon=True).start()
    
    def run_image_analysis(self):
        """Запускает анализ изображений"""
        def worker():
            try:
                # Открываем диалог выбора файла
                file_path = filedialog.askopenfilename(
                    title="Выберите изображение для анализа",
                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
                )
                
                if not file_path:
                    return
                
                self.logger_manager.log_system_event("Запуск анализа изображений из GUI", "INFO")
                
                results = self.orchestrator.run_image_analysis(file_path)
                
                if results.get("success"):
                    messagebox.showinfo("Анализ изображений", f"Анализ завершен!\nШероховатость: {results.get('roughness', 0):.4f}")
                    self.logger_manager.log_analyzer_event("Анализ изображений завершен успешно", "INFO")
                else:
                    messagebox.showerror("Ошибка", f"Ошибка анализа: {results.get('error', 'Unknown error')}")
                    self.logger_manager.log_analyzer_event("Ошибка анализа изображений", "ERROR")
                
            except Exception as e:
                self.logger_manager.log_analyzer_event(f"Ошибка анализа изображений: {e}", "ERROR")
                messagebox.showerror("Ошибка", f"Ошибка анализа изображений: {e}")
            finally:
                self.update_status_info()
        
        threading.Thread(target=worker, daemon=True).start()
    
    def run_sstv_decoding(self):
        """Запускает декодирование SSTV"""
        def worker():
            try:
                # Открываем диалог выбора файла
                file_path = filedialog.askopenfilename(
                    title="Выберите аудиофайл SSTV",
                    filetypes=[("Audio files", "*.wav *.mp3 *.flac *.aac")]
                )
                
                if not file_path:
                    return
                
                self.logger_manager.log_system_event("Запуск SSTV декодирования из GUI", "INFO")
                
                results = self.orchestrator.run_sstv_decoding(file_path)
                
                if results.get("success"):
                    messagebox.showinfo("SSTV декодирование", "Декодирование завершено!")
                    self.logger_manager.log_sstv_event("SSTV декодирование завершено успешно", "INFO")
                else:
                    messagebox.showerror("Ошибка", f"Ошибка декодирования: {results.get('error', 'Unknown error')}")
                    self.logger_manager.log_sstv_event("Ошибка SSTV декодирования", "ERROR")
                
            except Exception as e:
                self.logger_manager.log_sstv_event(f"Ошибка SSTV декодирования: {e}", "ERROR")
                messagebox.showerror("Ошибка", f"Ошибка SSTV декодирования: {e}")
            finally:
                self.update_status_info()
        
        threading.Thread(target=worker, daemon=True).start()
    
    def run_comprehensive_simulation(self):
        """Запускает комплексную симуляцию"""
        def worker():
            try:
                self.logger_manager.log_system_event("Запуск комплексной симуляции из GUI", "INFO")
                
                results = self.orchestrator.coordinate_multi_component_simulation((40, 40))
                
                messagebox.showinfo("Комплексная симуляция", f"Симуляция завершена!\nДлительность: {results['total_duration']:.2f} сек")
                self.logger_manager.log_simulation_event("Комплексная симуляция завершена успешно", "INFO")
                
            except Exception as e:
                self.logger_manager.log_system_event(f"Ошибка комплексной симуляции: {e}", "ERROR")
                messagebox.showerror("Ошибка", f"Ошибка комплексной симуляции: {e}")
            finally:
                self.update_status_info()
        
        threading.Thread(target=worker, daemon=True).start()
    
    def run_continuous_simulation(self):
        """Запускает непрерывную симуляцию"""
        def worker():
            try:
                self.logger_manager.log_system_event("Запуск непрерывной симуляции из GUI", "INFO")
                
                # Запускаем в фоне на 5 минут для тестирования
                self.orchestrator.start_background_simulation(5)
                
                messagebox.showinfo("Непрерывная симуляция", "Непрерывная симуляция запущена (5 минут)")
                
            except Exception as e:
                self.logger_manager.log_system_event(f"Ошибка непрерывной симуляции: {e}", "ERROR")
                messagebox.showerror("Ошибка", f"Ошибка непрерывной симуляции: {e}")
            finally:
                self.update_status_info()
        
        threading.Thread(target=worker, daemon=True).start()
    
    def stop_simulation(self):
        """Останавливает симуляцию"""
        self.orchestrator.stop_simulation()
        self.simulation_running = False
        messagebox.showinfo("Остановка", "Симуляция остановлена")
        self.update_status_info()
    
    def load_and_visualize_data(self):
        """Загружает и визуализирует данные"""
        try:
            file_path = filedialog.askopenfilename(
                title="Выберите файл данных",
                filetypes=[("Data files", "*.txt *.csv *.npy"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Пытаемся загрузить данные
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            else:
                data = np.loadtxt(file_path)
            
            # Визуализируем данные
            self.ax.clear()
            if len(data.shape) == 2:
                im = self.ax.imshow(data, cmap='viridis')
                self.fig.colorbar(im, ax=self.ax)
                self.ax.set_title("Визуализация данных")
            else:
                self.ax.plot(data)
                self.ax.set_title("График данных")
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки данных: {e}")
    
    def create_visualization_report(self):
        """Создает отчет визуализации"""
        try:
            # Создаем пример отчета
            sample_data = np.random.rand(50, 50)
            self.visualizer.visualize_all_for_report(
                surface_data=sample_data,
                original_image=sample_data,
                processed_image=sample_data,
                sstv_image=sample_data
            )
            
            messagebox.showinfo("Отчет", "Отчет визуализации создан в директории output")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания отчета: {e}")
    
    def refresh_logs(self):
        """Обновляет отображение логов"""
        # Для простоты показываем сообщение
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.insert(tk.END, "Логи обновляются...\n")
        self.logs_text.insert(tk.END, f"Время обновления: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def save_settings(self):
        """Сохраняет настройки"""
        try:
            # Обновляем конфигурацию
            self.config_manager.set("paths.data_dir", self.data_dir_var.get())
            self.config_manager.set("paths.output_dir", self.output_dir_var.get())
            
            # Обновляем менеджеры
            self.data_manager = DataManager(self.data_dir_var.get(), self.output_dir_var.get())
            
            messagebox.showinfo("Настройки", "Настройки успешно сохранены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения настроек: {e}")
    
    def run(self):
        """Запускает панель управления"""
        self.root.mainloop()


def main():
    """Главная функция запуска панели управления"""
    print("ЗАПУСК ИНТЕРАКТИВНОЙ ПАНЕЛИ УПРАВЛЕНИЯ")
    print("Инициализация компонентов...")
    
    try:
        dashboard = NanoprobeDashboard()
        print("✓ Панель управления инициализирована")
        print("Запуск интерфейса...")
        
        dashboard.run()
        
    except Exception as e:
        print(f"Ошибка запуска панели управления: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()