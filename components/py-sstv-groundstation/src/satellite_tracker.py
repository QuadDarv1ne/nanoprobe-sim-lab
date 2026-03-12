# -*- coding: utf-8 -*-
"""
Модуль отслеживания спутников для SSTV станции.
TLE данные, предсказание пролётов, автозапись.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class Satellite:
    """Класс для представления спутника."""
    
    def __init__(self, name: str, tle_line1: str, tle_line2: str):
        """
        Инициализация спутника.
        
        Args:
            name: Название спутника
            tle_line1: TLE строка 1
            tle_line2: TLE строка 2
        """
        self.name = name
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.epoch = self._parse_epoch(tle_line1)
        
    def _parse_epoch(self, line1: str) -> datetime:
        """Парсит эпоху из TLE."""
        try:
            year = int(line1[18:20])
            year += 2000 if year < 80 else 1900
            day_fraction = float(line1[20:32])
            epoch = datetime(year, 1, 1) + timedelta(days=day_fraction - 1)
            return epoch
        except Exception:
            return datetime.now()


class SatelliteTracker:
    """Трекер спутников с TLE данными."""
    
    # TLE данные для популярных спутников
    DEFAULT_SATELLITES = {
        'iss': Satellite(
            'ISS (ZARYA)',
            '1 25544U 98067A   24055.50416667  .00010600  00000-0  19200-3 0  9992',
            '2 25544  51.6400 200.0000 0006000  50.0000 310.0000 15.50000000123456'
        ),
        'noaa_15': Satellite(
            'NOAA 15',
            '1 25338U 98030A   24055.50416667  .00000100  00000-0  10000-3 0  9991',
            '2 25338  98.7000 200.0000 0014000  50.0000 310.0000 14.25000000123456'
        ),
        'noaa_18': Satellite(
            'NOAA 18',
            '1 28654U 05018A   24055.50416667  .00000100  00000-0  10000-3 0  9992',
            '2 28654  99.0000 200.0000 0014000  50.0000 310.0000 14.10000000123456'
        ),
        'noaa_19': Satellite(
            'NOAA 19',
            '1 33591U 09005A   24055.50416667  .00000100  00000-0  10000-3 0  9993',
            '2 33591  99.1000 200.0000 0014000  50.0000 310.0000 14.15000000123456'
        ),
        'meteor_m2': Satellite(
            'METEOR-M 2',
            '1 40069U 14037A   24055.50416667  .00000100  00000-0  10000-3 0  9994',
            '2 40069  98.5000 200.0000 0014000  50.0000 310.0000 13.80000000123456'
        ),
    }
    
    # Частоты SSTV спутников (МГц)
    SSTV_FREQUENCIES = {
        'iss': 145.800,
        'noaa_15': 137.620,
        'noaa_18': 137.9125,
        'noaa_19': 137.100,
        'meteor_m2': 137.900,
    }
    
    def __init__(self, ground_station_lat: float = 55.75, ground_station_lon: float = 37.61):
        """
        Инициализация трекера.
        
        Args:
            ground_station_lat: Широта наземной станции
            ground_station_lon: Долгота наземной станции
        """
        self.ground_station_lat = ground_station_lat
        self.ground_station_lon = ground_station_lon
        self.satellites = self.DEFAULT_SATELLITES.copy()
        self.tle_file = Path("data/tle_data.json")
        
    def load_tle(self, filepath: str) -> int:
        """
        Загружает TLE данные из файла.
        
        Args:
            filepath: Путь к TLE файлу
            
        Returns:
            int: Количество загруженных спутников
        """
        try:
            with open(filepath, 'r') as f:
                tle_data = json.load(f)
            
            count = 0
            for name, tle in tle_data.items():
                if 'line1' in tle and 'line2' in tle:
                    self.satellites[name] = Satellite(name, tle['line1'], tle['line2'])
                    count += 1
            
            print(f"Загружено {count} спутников из {filepath}")
            return count
        except Exception as e:
            print(f"Ошибка загрузки TLE: {e}")
            return 0
    
    def save_tle(self, filepath: str = None) -> bool:
        """
        Сохраняет TLE данные в файл.
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            bool: True если успешно
        """
        filepath = filepath or self.tle_file
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            tle_data = {}
            for name, sat in self.satellites.items():
                tle_data[name] = {
                    'line1': sat.tle_line1,
                    'line2': sat.tle_line2
                }
            
            with open(filepath, 'w') as f:
                json.dump(tle_data, f, indent=2)
            
            print(f"TLE сохранены: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения TLE: {e}")
            return False
    
    def get_pass_predictions(self, satellite_name: str, 
                             hours_ahead: int = 24,
                             min_elevation: float = 10.0) -> List[Dict]:
        """
        Получает предсказания пролётов спутника.
        
        Args:
            satellite_name: Название спутника
            hours_ahead: На сколько часов вперёд
            min_elevation: Минимальная высота (градусы)
            
        Returns:
            List[Dict]: Список пролётов
        """
        if satellite_name not in self.satellites:
            print(f"Спутник '{satellite_name}' не найден")
            return []
        
        satellite = self.satellites[satellite_name]
        passes = []
        
        # Упрощённое предсказание (для реального нужна библиотека sgp4)
        # ISS: ~15.5 орбит в день, период ~92 минуты
        if 'iss' in satellite_name.lower():
            period_minutes = 92
            passes_per_day = 15.5
        elif 'noaa' in satellite_name.lower():
            period_minutes = 102
            passes_per_day = 14.1
        elif 'meteor' in satellite_name.lower():
            period_minutes = 104
            passes_per_day = 13.8
        else:
            period_minutes = 95
            passes_per_day = 15.1
        
        now = datetime.now()
        end_time = now + timedelta(hours=hours_ahead)
        
        # Генерируем предсказания
        current_time = now
        pass_num = 0
        
        while current_time < end_time:
            # Добавляем период
            current_time += timedelta(minutes=period_minutes)
            
            # Простая эвристика видимости
            # В реальности нужно использовать SGP4
            is_visible = (pass_num % 3) == 0  # Каждый 3-й пролёт видимый
            
            if is_visible:
                aos_time = current_time
                los_time = current_time + timedelta(minutes=8)  # 8 минут пролёт
                max_elevation = 30 + (pass_num % 60)  # 30-90 градусов
                
                passes.append({
                    'satellite': satellite_name,
                    'aos': aos_time,  # Acquisition of Signal
                    'los': los_time,  # Loss of Signal
                    'max_elevation': max_elevation,
                    'frequency': self.SSTV_FREQUENCIES.get(satellite_name.lower(), 0),
                    'duration_minutes': 8
                })
            
            pass_num += 1
        
        return passes
    
    def get_current_position(self, satellite_name: str) -> Optional[Dict]:
        """
        Получает текущую позицию спутника.
        
        Args:
            satellite_name: Название спутника
            
        Returns:
            Dict: Позиция спутника или None
        """
        if satellite_name not in self.satellites:
            return None
        
        # Упрощённая позиция (для реальной нужен SGP4)
        satellite = self.satellites[satellite_name]
        
        # ISS орбита: ~400-420 км
        if 'iss' in satellite_name.lower():
            altitude_km = 420
            velocity_kmh = 27600
        elif 'noaa' in satellite_name.lower():
            altitude_km = 850
            velocity_kmh = 27000
        else:
            altitude_km = 800
            velocity_kmh = 27200
        
        return {
            'satellite': satellite_name,
            'altitude_km': altitude_km,
            'velocity_kmh': velocity_kmh,
            'latitude': (datetime.now().minute % 90) * (1 if datetime.now().second < 30 else -1),
            'longitude': (datetime.now().hour * 15) % 360 - 180,
            'footprint_km': 4500 if altitude_km > 800 else 2500
        }
    
    def is_satellite_visible(self, satellite_name: str, 
                             min_elevation: float = 10.0) -> bool:
        """
        Проверяет видимость спутника.
        
        Args:
            satellite_name: Название спутника
            min_elevation: Минимальная высота
            
        Returns:
            bool: True если виден
        """
        position = self.get_current_position(satellite_name)
        if not position:
            return False
        
        # Простая проверка (для реальной нужна тригонометрия)
        # Считаем что спутник виден если он над горизонтом
        return abs(position['latitude'] - self.ground_station_lat) < 30
    
    def get_next_pass(self, satellite_name: str) -> Optional[Dict]:
        """
        Получает следующий пролёт спутника.
        
        Args:
            satellite_name: Название спутника
            
        Returns:
            Dict: Информация о пролёте или None
        """
        passes = self.get_pass_predictions(satellite_name, hours_ahead=24)
        if passes:
            return passes[0]
        return None
    
    def get_all_satellites(self) -> List[str]:
        """
        Получает список всех спутников.
        
        Returns:
            List[str]: Список названий
        """
        return list(self.satellites.keys())
    
    def get_sstv_schedule(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Получает расписание SSTV передач.
        
        Args:
            hours_ahead: На сколько часов вперёд
            
        Returns:
            List[Dict]: Расписание
        """
        schedule = []
        
        for sat_name in self.SSTV_FREQUENCIES.keys():
            passes = self.get_pass_predictions(sat_name, hours_ahead)
            for pass_info in passes:
                schedule.append(pass_info)
        
        # Сортируем по времени
        schedule.sort(key=lambda x: x['aos'])
        
        return schedule
