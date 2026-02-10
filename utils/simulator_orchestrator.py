# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль оркестратора симуляции для проекта Лаборатория моделирования нанозонда
Этот модуль координирует работу всех компонентов проекта
для комплексной симуляции.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
import queue
from datetime import datetime

from utils.config_manager import ConfigManager
from utils.logger import setup_project_logging
from utils.data_manager import DataManager
from utils.visualizer import ProjectVisualizer
from cpp_spm_hardware_sim.src.spm_simulator import SurfaceModel, SPMController
from py_surface_image_analyzer.src.image_processor import ImageProcessor, calculate_surface_roughness
from py_sstv_groundstation.src.sstv_decoder import SSTVDecoder

class SimulationOrchestrator:
    """
    Класс оркестратора симуляции
    Координирует работу всех компонентов проекта для комплексной симуляции
    процессов, происходящих в нанозондовом микроскопе и связанных системах.
    """


    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Инициализирует оркестратор симуляции

        Args:
            config_manager: Экземпляр менеджера конфигурации (опционально)
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger_manager = setup_project_logging(self.config_manager)
        self.data_manager = DataManager()
        self.visualizer = ProjectVisualizer()

        # Компоненты проекта
        self.spm_controller = None
        self.image_processor = None
        self.sstv_decoder = None

        # Состояния симуляции
        self.simulation_running = False
        self.simulation_thread = None

        # Очереди для межкомпонентного взаимодействия
        self.data_queue = queue.Queue()
        self.event_queue = queue.Queue()

        # Результаты симуляции
        self.simulation_results = {}


    def initialize_components(self):
        """Инициализирует все компоненты проекта"""
        self.logger_manager.log_system_event("Инициализация компонентов симуляции", "INFO")

        # Инициализация СЗМ симулятора
        try:
            self.spm_controller = SPMController()
            self.logger_manager.log_spm_event("СЗМ контроллер инициализирован", "INFO")
        except Exception as e:
            self.logger_manager.log_spm_event(f"Ошибка инициализации СЗМ: {e}", "ERROR")

        # Инициализация анализатора изображений
        try:
            self.image_processor = ImageProcessor()
            self.logger_manager.log_analyzer_event("Анализатор изображений инициализирован", "INFO")
        except Exception as e:
            self.logger_manager.log_analyzer_event(f"Ошибка инициализации анализатора: {e}", "ERROR")

        # Инициализация SSTV станции
        try:
            self.sstv_decoder = SSTVDecoder()
            self.logger_manager.log_sstv_event("SSTV декодер инициализирован", "INFO")
        except Exception as e:
            self.logger_manager.log_sstv_event(f"Ошибка инициализации SSTV: {e}", "ERROR")


    def create_simulation_surface(self, size: tuple = (50, 50)) -> 'SurfaceModel':
        """
        Создает поверхность для симуляции

        Args:
            size: Размер поверхности (ширина, высота)

        Returns:
            Экземпляр модели поверхности
        """
        self.logger_manager.log_spm_event(f"Создание поверхности размером {size}", "INFO")

        surface = SurfaceModel(size[0], size[1])

        # Сохраняем поверхность
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulated_surface_{timestamp}.txt"
        surface.save_to_file(filename)

        self.logger_manager.log_spm_event(f"Поверхность создана и сохранена как {filename}", "INFO")
        return surface


    def run_spm_simulation(self, surface: 'SurfaceModel', duration: float = 10.0) -> Dict[str, Any]:
        """
        Запускает симуляцию сканирования поверхности СЗМ

        Args:
            surface: Модель поверхности для сканирования
            duration: Продолжительность симуляции в секундах

        Returns:
            Словарь с результатами симуляции
        """
        self.logger_manager.log_spm_event("Начало симуляции сканирования СЗМ", "INFO")

        if self.spm_controller is None:
            self.initialize_components()

        # Устанавливаем поверхность
        self.spm_controller.set_surface(surface)

        # Запускаем сканирование
        start_time = time.time()
        self.spm_controller.scanSurface()  # Метод может потребовать другого имени
        end_time = time.time()

        # Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"spm_scan_results_{timestamp}.txt"
        self.spm_controller.save_scan_results(results_filename)

        results = {
            "duration": end_time - start_time,
            "surface_size": (surface.getWidth(), surface.getHeight()),
            "results_file": results_filename,
            "timestamp": timestamp
        }

        self.logger_manager.log_spm_event(f"Симуляция СЗМ завершена за {results['duration']:.2f} сек", "INFO")
        return results


    def run_image_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Запускает анализ изображения

        Args:
            image_path: Путь к изображению для анализа

        Returns:
            Словарь с результатами анализа
        """
        self.logger_manager.log_analyzer_event(f"Начало анализа изображения: {image_path}", "INFO")

        if self.image_processor is None:
            self.initialize_components()

        # Загружаем изображение
        if self.image_processor.load_image(image_path):
            # Применяем фильтрацию
            filtered = self.image_processor.apply_noise_reduction("gaussian")

            # Вычисляем шероховатость
            if filtered is not None:
                roughness = calculate_surface_roughness(filtered)
            else:
                roughness = 0.0

            # Сохраняем результаты
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                "input_image": image_path,
                "roughness": roughness,
                "analysis_timestamp": timestamp,
                "success": True
            }

            # Сохраняем результаты анализа
            analysis_file = f"image_analysis_{timestamp}.json"
            self.data_manager.save_image_analysis_results(results, analysis_file)

            self.logger_manager.log_analyzer_event("Анализ изображения завершен", "INFO")
            return results
        else:
            self.logger_manager.log_analyzer_event("Ошибка загрузки изображения", "ERROR")
            return {"success": False, "error": "Failed to load image"}


    def run_sstv_decoding(self, audio_file: str) -> Dict[str, Any]:
        """
        Запускает декодирование SSTV сигнала

        Args:
            audio_file: Путь к аудиофайлу с SSTV сигналом

        Returns:
            Словарь с результатами декодирования
        """
        self.logger_manager.log_sstv_event(f"Начало декодирования SSTV: {audio_file}", "INFO")

        if self.sstv_decoder is None:
            self.initialize_components()

        # Декодируем сигнал
        decoded_image = self.sstv_decoder.decode_from_audio(audio_file)

        if decoded_image:
            # Сохраняем декодированное изображение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sstv_decoded_{timestamp}.png"
            success = self.sstv_decoder.save_decoded_image(output_file)

            results = {
                "input_audio": audio_file,
                "decoded_image": output_file if success else None,
                "decoding_timestamp": timestamp,
                "success": success
            }

            self.logger_manager.log_sstv_event("Декодирование SSTV завершено", "INFO")
            return results
        else:
            self.logger_manager.log_sstv_event("Ошибка декодирования SSTV", "ERROR")
            return {"success": False, "error": "Failed to decode SSTV signal"}


    def coordinate_multi_component_simulation(self, surface_size: tuple = (50, 50)) -> Dict[str, Any]:
        """
        Координирует симуляцию с участием нескольких компонентов

        Args:
            surface_size: Размер поверхности для симуляции

        Returns:
            Словарь с результатами комплексной симуляции
        """
        self.logger_manager.log_simulation_event("Начало комплексной симуляции", "INFO")

        start_time = time.time()

        # Создаем поверхность
        surface = self.create_simulation_surface(surface_size)

        # Запускаем симуляцию СЗМ
        spm_results = self.run_spm_simulation(surface)

        # Создаем визуализацию результатов
        try:
            # Для демонстрации используем случайные данные
            sample_surface_data = np.random.rand(surface_size[1], surface_size[0])
            self.visualizer.surface_viz.plot_surface_2d(
                sample_surface_data,
                "Результаты комплексной симуляции",
                f"output/comprehensive_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка визуализации: {e}", "WARNING")

        end_time = time.time()

        comprehensive_results = {
            "total_duration": end_time - start_time,
            "spm_results": spm_results,
            "timestamp": datetime.now().isoformat(),
            "components_involved": ["SPM", "Visualization"]
        }

        # Сохраняем метаданные симуляции
        self.data_manager.save_simulation_metadata(comprehensive_results,
                                                f"comprehensive_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        self.logger_manager.log_simulation_event("Комплексная симуляция завершена", "INFO")
        return comprehensive_results


    def run_continuous_simulation(self, duration_minutes: int = 10):
        """
        Запускает непрерывную симуляцию в течение заданного времени

        Args:
            duration_minutes: Продолжительность симуляции в минутах
        """
        self.logger_manager.log_simulation_event(f"Начало непрерывной симуляции на {duration_minutes} минут", "INFO")

        self.simulation_running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        cycle_count = 0

        while time.time() < end_time and self.simulation_running:
            try:
                # Выполняем цикл симуляции
                cycle_results = self.coordinate_multi_component_simulation()

                cycle_count += 1
                self.logger_manager.log_simulation_event(f"Цикл симуляции #{cycle_count} завершен", "INFO")

                # Ждем перед следующим циклом
                time.sleep(5)  # 5 секунд между циклами

            except Exception as e:
                self.logger_manager.log_system_event(f"Ошибка в цикле симуляции: {e}", "ERROR")
                time.sleep(1)  # Пауза перед повторной попыткой

        self.simulation_running = False
        self.logger_manager.log_simulation_event("Непрерывная симуляция завершена", "INFO")


    def stop_simulation(self):
        """Останавливает текущую симуляцию"""
        self.simulation_running = False
        self.logger_manager.log_system_event("Симуляция остановлена пользователем", "INFO")


    def start_background_simulation(self, duration_minutes: int = 10):
        """
        Запускает симуляцию в фоновом потоке

        Args:
            duration_minutes: Продолжительность симуляции в минутах
        """
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.logger_manager.log_system_event("Симуляция уже запущена", "WARNING")
            return False

        self.simulation_thread = threading.Thread(
            target=self.run_continuous_simulation,
            args=(duration_minutes,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

        self.logger_manager.log_system_event("Фоновая симуляция запущена", "INFO")
        return True


    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Возвращает статус текущей симуляции

        Returns:
            Словарь с информацией о статусе симуляции
        """
        return {
            "simulation_running": self.simulation_running,
            "thread_active": self.simulation_thread.is_alive() if self.simulation_thread else False,
            "results_count": len(self.simulation_results),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Главная функция для демонстрации возможностей оркестратора"""
    print("=== ОРКЕСТРАТОР СИМУЛЯЦИИ ПРОЕКТА ===")

    # Создаем оркестратор
    orchestrator = SimulationOrchestrator()

    # Инициализируем компоненты
    orchestrator.initialize_components()

    print("✓ Оркестратор симуляции инициализирован")
    print("Тестирование основных функций...")

    # Тестируем создание поверхности
    try:
        surface = orchestrator.create_simulation_surface((20, 20))
        print("✓ Поверхность успешно создана")
    except Exception as e:
        print(f"✗ Ошибка создания поверхности: {e}")

    # Проверяем статус симуляции
    status = orchestrator.get_simulation_status()
    print(f"Статус симуляции: {status}")

    print("Оркестратор симуляции готов к работе")

if __name__ == "__main__":
    main()

