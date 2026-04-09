"""
Модуль отслеживания спутников для SSTV станции.
Реальные TLE данные из CelesTrak, SGP4 propagation, предсказание пролётов.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import requests
from sgp4.api import Satrec
from sgp4.api import jday
import numpy as np


class Satellite:
    """Класс для представления спутника с SGP4."""

    def __init__(self, name: str, tle_line1: str, tle_line2: str):
        """
        Инициализация спутника.

        Args:
            name: Название спутника
            tle_line1: TLE строка 1
            tle_line2: TLE строка 2
        """
        self.name = name
        self.tle_line1 = tle_line1.strip()
        self.tle_line2 = tle_line2.strip()
        self.epoch = self._parse_epoch(tle_line1)
        
        # Инициализация SGP4
        try:
            self.satellite = Satrec.twoline2rv(self.tle_line1, self.tle_line2)
            self.sgp4_initialized = True
        except Exception as e:
            print(f"⚠ Ошибка инициализации SGP4 для {name}: {e}")
            self.satellite = None
            self.sgp4_initialized = False

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
    
    def get_position(self, dt: datetime = None) -> Optional[Dict]:
        """
        Получает позицию спутника через SGP4.
        
        Args:
            dt: Время (по умолчанию сейчас)
            
        Returns:
            Dict с position и velocity или None
        """
        if not self.sgp4_initialized or not self.satellite:
            return None
        
        if dt is None:
            dt = datetime.utcnow()
        
        try:
            jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
            e, r, v = self.satellite.sgp4(jd, fr)
            
            if e == 0:  # Успех
                # r = [x, y, z] в km, v = [vx, vy, vz] в km/s
                altitude_km = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - 6371.0  # Earth radius
                velocity_kmh = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2) * 3600
                
                # Latitude/Longitude (упрощённо)
                lat = np.degrees(np.arctan2(r[2], np.sqrt(r[0]**2 + r[1]**2)))
                lon = np.degrees(np.arctan2(r[1], r[0]))
                
                return {
                    'position_km': r,
                    'velocity_km_s': v,
                    'altitude_km': altitude_km,
                    'velocity_kmh': velocity_kmh,
                    'latitude': lat,
                    'longitude': lon,
                    'footprint_km': 2 * np.sqrt(2 * 6371.0 * altitude_km)  # Approximation
                }
            else:
                print(f"⚠ SGP4 error {e} for {self.name}")
                return None
        except Exception as e:
            print(f"⚠ SGP4 computation error: {e}")
            return None


class SatelliteTracker:
    """Трекер спутников с реальными TLE данными из CelesTrak."""

    # NORAD ID для спутников
    SATELLITE_NORAD_IDS = {
        'iss': 25544,
        'noaa_15': 25338,
        'noaa_18': 28654,
        'noaa_19': 33591,
        'meteor_m2_3': 57067,  # Метеор-М2-3 (актуальный)
    }

    # Начальные TLE (будут обновлены из CelesTrak)
    DEFAULT_SATELLITES = {
        'iss': Satellite(
            'ISS (ZARYA)',
            '1 25544U 98067A   25088.50416667  .00015000  00000-0  27000-3 0  9991',
            '2 25544  51.6420 120.5000 0005000  80.0000 280.0000 15.50000000500001'
        ),
        'noaa_15': Satellite(
            'NOAA 15',
            '1 25338U 98030A   25088.50416667  .00000200  00000-0  12000-3 0  9992',
            '2 25338  98.7500 250.0000 0013000  60.0000 300.0000 14.25000000500002'
        ),
        'noaa_18': Satellite(
            'NOAA 18',
            '1 28654U 05018A   25088.50416667  .00000200  00000-0  12000-3 0  9993',
            '2 28654  99.0500 260.0000 0013500  70.0000 290.0000 14.10000000500003'
        ),
        'noaa_19': Satellite(
            'NOAA 19',
            '1 33591U 09005A   25088.50416667  .00000200  00000-0  12000-3 0  9994',
            '2 33591  99.1500 270.0000 0014000  80.0000 280.0000 14.15000000500004'
        ),
        'meteor_m2_3': Satellite(
            'METEOR-M 2-3',
            '1 57067U 23085A   25088.50416667  .00000200  00000-0  12000-3 0  9995',
            '2 57067  98.6000 240.0000 0013000  50.0000 310.0000 13.85000000500005'
        ),
    }

    # Частоты SSTV спутников (МГц)
    SSTV_FREQUENCIES = {
        'iss': 145.800,
        'noaa_15': 137.620,
        'noaa_18': 137.9125,
        'noaa_19': 137.100,
        'meteor_m2_3': 137.900,
    }
    
    def fetch_tle_from_celestrak(self, norad_id: int) -> Optional[Tuple[str, str]]:
        """
        Загружает TLE с CelesTrak API.
        
        Args:
            norad_id: NORAD ID спутника
            
        Returns:
            Tuple (line1, line2) или None
        """
        try:
            url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id:05d}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            if len(lines) >= 3:
                line1 = lines[1].strip()
                line2 = lines[2].strip()
                return line1, line2
        except Exception as e:
            print(f"⚠ Ошибка загрузки TLE с CelesTrak: {e}")
        
        return None
    
    def update_tle_from_celestrak(self) -> int:
        """
        Обновляет все TLE из CelesTrak.
        
        Returns:
            int: Количество обновлённых спутников
        """
        print("\n📡 Обновление TLE из CelesTrak...")
        updated = 0
        
        for name, norad_id in self.SATELLITE_NORAD_IDS.items():
            tle_data = self.fetch_tle_from_celestrak(norad_id)
            if tle_data:
                line1, line2 = tle_data
                self.satellites[name] = Satellite(name, line1, line2)
                updated += 1
                print(f"  ✓ {name}: TLE обновлён")
            else:
                print(f"  ⚠ {name}: не удалось обновить TLE")
        
        print(f"✓ Обновлены TLE: {updated}/{len(self.SATELLITE_NORAD_IDS)}")
        return updated

    def _elevation_from_position(self, pos: Dict, dt: datetime) -> float:
        """
        Вычисляет угол возвышения спутника над горизонтом наблюдателя.
        Учитывает вращение Земли через GMST.
        
        Args:
            pos: Позиция спутника (ECI координаты)
            dt: Время UTC
            
        Returns:
            float: Угол возвышения в градусах
        """
        # GMST (Greenwich Mean Sidereal Time) для перевода ECI → ECEF
        jd = (dt - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 + 2451545.0
        gmst_rad = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360
        gmst_rad = np.radians(gmst_rad)

        # Позиция наблюдателя в ECEF (км)
        R_earth = 6371.0
        lat_r = np.radians(self.ground_station_lat)
        lon_r = np.radians(self.ground_station_lon) + gmst_rad
        obs_x = R_earth * np.cos(lat_r) * np.cos(lon_r)
        obs_y = R_earth * np.cos(lat_r) * np.sin(lon_r)
        obs_z = R_earth * np.sin(lat_r)

        # Вектор от наблюдателя к спутнику (ECI)
        r = pos['position_km']
        dx = r[0] - obs_x
        dy = r[1] - obs_y
        dz = r[2] - obs_z

        # Единичный вектор "вверх" для наблюдателя
        up_x = np.cos(lat_r) * np.cos(lon_r)
        up_y = np.cos(lat_r) * np.sin(lon_r)
        up_z = np.sin(lat_r)

        # Elevation = arcsin(dot(range_vec, up_vec) / |range_vec|)
        range_mag = np.sqrt(dx**2 + dy**2 + dz**2)
        if range_mag < 1e-6:
            return 0.0
        dot = (dx * up_x + dy * up_y + dz * up_z) / range_mag
        return float(np.degrees(np.arcsin(np.clip(dot, -1.0, 1.0))))

    def __init__(self, ground_station_lat: float = 55.7558, ground_station_lon: float = 37.6173):
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

        # Загружаем TLE из кэша или обновляем с CelesTrak
        self._load_or_refresh_tle()

    def _load_or_refresh_tle(self, max_age_hours: int = 12):
        """
        Загружает TLE из кэша если свежие, иначе обновляет с CelesTrak.
        
        Args:
            max_age_hours: Максимальный возраст кэша в часах (по умолчанию 12)
        """
        tle_file = Path("data/tle_data.json")
        
        # Проверяем возраст кэша
        if tle_file.exists():
            import time
            age_hours = (time.time() - tle_file.stat().st_mtime) / 3600
            if age_hours < max_age_hours:
                loaded = self.load_tle(str(tle_file))
                if loaded > 0:
                    print(f"✓ TLE загружены из кэша ({age_hours:.1f}ч назад)")
                    return
        
        # Кэш устарел или отсутствует — обновляем
        updated = self.update_tle_from_celestrak()
        if updated > 0:
            self.save_tle(str(tle_file))
        else:
            print("⚠ Используются встроенные TLE (CelesTrak недоступен)")

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
        save_path = Path(filepath) if filepath else self.tle_file
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            tle_data = {}
            for name, sat in self.satellites.items():
                tle_data[name] = {
                    'line1': sat.tle_line1,
                    'line2': sat.tle_line2
                }

            with open(save_path, 'w') as f:
                json.dump(tle_data, f, indent=2)

            print(f"TLE сохранены: {save_path}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения TLE: {e}")
            return False

    def get_pass_predictions(self, satellite_name: str,
                             hours_ahead: int = 24,
                             min_elevation: float = 10.0) -> List[Dict]:
        """
        Получает предсказания пролётов спутника через SGP4.
        
        Реальное предсказание на основе позиции спутника и наземной станции.

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

        sat = self.satellites[satellite_name]
        if not sat.sgp4_initialized:
            print(f"⚠ SGP4 не инициализирован для {satellite_name}")
            return []

        passes = []
        now = datetime.utcnow()
        end_time = now + timedelta(hours=hours_ahead)
        
        # Шаг 30 секунд для точности
        time_step = timedelta(seconds=30)
        current_time = now
        
        in_pass = False
        current_pass = None
        
        while current_time < end_time:
            pos = sat.get_position(current_time)

            if pos:
                elevation = self._elevation_from_position(pos, current_time)

                if elevation >= min_elevation:
                    if not in_pass:
                        in_pass = True
                        current_pass = {
                            'satellite': satellite_name,
                            'aos': current_time,
                            'max_elevation': elevation,
                            'frequency': self.SSTV_FREQUENCIES.get(satellite_name, 0),
                        }
                    else:
                        current_pass['max_elevation'] = max(current_pass['max_elevation'], elevation)
                else:
                    if in_pass:
                        current_pass['los'] = current_time
                        duration_minutes = (current_time - current_pass['aos']).total_seconds() / 60.0
                        if duration_minutes >= 2:
                            passes.append({**current_pass, 'duration_minutes': duration_minutes})
                        in_pass = False
                        current_pass = None

            current_time += time_step

        if in_pass and current_pass:
            current_pass['los'] = current_time
            duration_minutes = (current_time - current_pass['aos']).total_seconds() / 60.0
            passes.append({**current_pass, 'duration_minutes': duration_minutes})

        # Добавляем time_until_aos для каждого пролёта
        now_local = datetime.utcnow()
        for p in passes:
            delta = (p['aos'] - now_local).total_seconds()
            sign = "" if delta >= 0 else "-"
            delta = abs(int(delta))
            h, rem = divmod(delta, 3600)
            m, s = divmod(rem, 60)
            p['time_until_aos'] = f"{sign}{h:02d}:{m:02d}:{s:02d}"

        return passes

    def get_current_position(self, satellite_name: str) -> Optional[Dict]:
        """
        Получает текущую позицию спутника через SGP4.

        Args:
            satellite_name: Название спутника

        Returns:
            Dict: Позиция спутника или None
        """
        if satellite_name not in self.satellites:
            return None

        sat = self.satellites[satellite_name]
        pos = sat.get_position()
        
        if pos:
            return {
                'satellite': satellite_name,
                **pos,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return None

    def is_satellite_visible(self, satellite_name: str,
                             min_elevation: float = 10.0) -> bool:
        pos = self.get_current_position(satellite_name)
        if not pos:
            return False
        elevation = self._elevation_from_position(pos, datetime.utcnow())
        return elevation >= min_elevation

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
