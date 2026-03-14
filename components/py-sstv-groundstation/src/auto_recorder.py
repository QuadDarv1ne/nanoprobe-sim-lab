"""
Автоматическая запись при пролёте спутника.
Мониторинг расписания, автозапуск за 5 мин до AOS, остановка после LOS.
"""

import time
import threading
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ScheduledRecording:
    """Запланированная запись"""
    satellite: str
    aos: datetime  # Acquisition of Signal
    los: datetime  # Loss of Signal
    frequency: float  # MHz
    duration_minutes: int
    output_dir: str = "output/auto_recordings"
    status: str = "scheduled"  # scheduled, recording, completed, failed
    recording_process: Optional[subprocess.Popen] = None
    metadata: Dict = field(default_factory=dict)


class AutoRecordingScheduler:
    """Планировщик автоматической записи спутников."""

    def __init__(
        self,
        ground_station_lat: float = 55.75,
        ground_station_lon: float = 37.61,
        pre_pass_minutes: int = 5,
        post_pass_minutes: int = 2
    ):
        """
        Инициализация планировщика.

        Args:
            ground_station_lat: Широта наземной станции
            ground_station_lon: Долгота наземной станции
            pre_pass_minutes: Начинать запись за N минут до AOS
            post_pass_minutes: Заканчивать запись через N минут после LOS
        """
        self.lat = ground_station_lat
        self.lon = ground_station_lon
        self.pre_pass_minutes = pre_pass_minutes
        self.post_pass_minutes = post_pass_minutes

        self.recordings: List[ScheduledRecording] = []
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callback: Optional[Callable] = None

        # Путь к main.py для запуска записи
        self.main_script = Path(__file__).parent / "main.py"

        # Создаём директорию для записей
        Path("output/auto_recordings").mkdir(parents=True, exist_ok=True)

    def set_callback(self, callback: Callable):
        """Установка callback для уведомлений."""
        self.callback = callback

    def _notify(self, event_type: str, data: Dict):
        """Отправка уведомления через callback."""
        if self.callback:
            self.callback(event_type, data)

    def load_schedule(self, hours_ahead: int = 24) -> List[ScheduledRecording]:
        """
        Загружает расписание пролётов.

        Args:
            hours_ahead: На сколько часов вперёд

        Returns:
            List[ScheduledRecording]: Список записей
        """
        from satellite_tracker import SatelliteTracker

        tracker = SatelliteTracker(
            ground_station_lat=self.lat,
            ground_station_lon=self.lon
        )

        # Получаем расписание для всех SSTV спутников
        sstv_satellites = ['iss', 'noaa_15', 'noaa_18', 'noaa_19', 'meteor_m2']

        self.recordings = []

        for sat_name in sstv_satellites:
            passes = tracker.get_pass_predictions(sat_name, hours_ahead=hours_ahead)

            for pass_info in passes:
                # Пропускаем пролёты с низкой высотой
                if pass_info.get('max_elevation', 0) < 20:
                    continue

                # Рассчитываем время начала/окончания записи
                aos = pass_info['aos']
                los = pass_info['los']

                start_time = aos - timedelta(minutes=self.pre_pass_minutes)
                end_time = los + timedelta(minutes=self.post_pass_minutes)

                duration = int((end_time - start_time).total_seconds() / 60)

                recording = ScheduledRecording(
                    satellite=sat_name.upper(),
                    aos=start_time,
                    los=end_time,
                    frequency=pass_info.get('frequency', 0),
                    duration_minutes=duration,
                    metadata={
                        'max_elevation': pass_info.get('max_elevation', 0),
                        'original_aos': aos,
                        'original_los': los
                    }
                )

                self.recordings.append(recording)

        # Сортируем по времени
        self.recordings.sort(key=lambda x: x.aos)

        self._notify('schedule_loaded', {
            'count': len(self.recordings),
            'next_pass': self.recordings[0].satellite if self.recordings else None
        })

        return self.recordings

    def start_monitoring(self):
        """Запускает мониторинг расписания."""
        if self.is_running:
            print("Мониторинг уже запущен")
            return

        self.is_running = True

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self._notify('monitoring_started', {'time': datetime.now().isoformat()})
        print(f"Мониторинг запущен. Запланировано записей: {len(self.recordings)}")

    def stop_monitoring(self):
        """Останавливает мониторинг."""
        self.is_running = False

        # Останавливаем все активные записи
        for recording in self.recordings:
            if recording.status == 'recording' and recording.recording_process:
                self._stop_recording(recording)

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self._notify('monitoring_stopped', {'time': datetime.now().isoformat()})
        print("Мониторинг остановлен")

    def _monitor_loop(self):
        """Основной цикл мониторинга."""
        while self.is_running:
            now = datetime.now()

            for recording in self.recordings:
                if recording.status == 'scheduled':
                    # Проверка времени начала
                    if now >= recording.aos:
                        self._start_recording(recording)

                elif recording.status == 'recording':
                    # Проверка времени окончания
                    if now >= recording.los:
                        self._stop_recording(recording)

            # Проверка каждые 10 секунд
            time.sleep(10)

    def _start_recording(self, recording: ScheduledRecording):
        """Начинает запись."""
        try:
            recording.status = 'recording'

            # Создаём директорию для записи
            timestamp = recording.aos.strftime('%Y%m%d_%H%M%S')
            output_dir = Path(recording.output_dir) / f"{recording.satellite}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Формируем команду
            cmd = [
                'python', str(self.main_script),
                '--sdr',
                '-f', str(recording.frequency),
                '--duration', str(recording.duration_minutes),
                '--output-audio', str(output_dir / f"{recording.satellite}_{timestamp}.wav"),
                '--output-image', str(output_dir / f"{recording.satellite}_{timestamp}.png"),
                '--auto-decode',
                '--bias-tee',
                '--agc'
            ]

            # Запускаем процесс
            recording.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            recording.metadata['start_time'] = datetime.now().isoformat()
            recording.metadata['output_dir'] = str(output_dir)

            self._notify('recording_started', {
                'satellite': recording.satellite,
                'frequency': recording.frequency,
                'output_dir': str(output_dir)
            })

            print(f"🔴 ЗАПИСЬ НАЧАТА: {recording.satellite} на {recording.frequency} MHz")

        except Exception as e:
            recording.status = 'failed'
            recording.metadata['error'] = str(e)

            self._notify('recording_failed', {
                'satellite': recording.satellite,
                'error': str(e)
            })

            print(f"❌ Ошибка начала записи: {e}")

    def _stop_recording(self, recording: ScheduledRecording):
        """Останавливает запись."""
        try:
            if recording.recording_process:
                # Останавливаем процесс
                recording.recording_process.terminate()
                try:
                    recording.recording_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    recording.recording_process.kill()
                    recording.recording_process.wait(timeout=5)

                recording.recording_process = None

            recording.status = 'completed'
            recording.metadata['end_time'] = datetime.now().isoformat()

            # Сохраняем метаданные
            self._save_metadata(recording)

            self._notify('recording_completed', {
                'satellite': recording.satellite,
                'output_dir': recording.metadata.get('output_dir', '')
            })

            print(f"✅ ЗАПИСЬ ЗАВЕРШЕНА: {recording.satellite}")

        except Exception as e:
            recording.status = 'failed'
            recording.metadata['error'] = str(e)

            self._notify('recording_failed', {
                'satellite': recording.satellite,
                'error': str(e)
            })

            print(f"❌ Ошибка остановки записи: {e}")

    def _save_metadata(self, recording: ScheduledRecording):
        """Сохраняет метаданные записи."""
        output_dir = recording.metadata.get('output_dir', '')
        if not output_dir:
            return

        metadata_file = Path(output_dir) / "recording_metadata.json"

        metadata = {
            'satellite': recording.satellite,
            'frequency_mhz': recording.frequency,
            'aos': recording.aos.isoformat(),
            'los': recording.los.isoformat(),
            'duration_minutes': recording.duration_minutes,
            'ground_station': {
                'latitude': self.lat,
                'longitude': self.lon
            },
            'recording': {
                'start_time': recording.metadata.get('start_time', ''),
                'end_time': recording.metadata.get('end_time', ''),
                'status': recording.status
            },
            'pass_info': {
                'max_elevation': recording.metadata.get('max_elevation', 0),
                'original_aos': recording.metadata.get('original_aos', '').isoformat() if recording.metadata.get('original_aos') else '',
                'original_los': recording.metadata.get('original_los', '').isoformat() if recording.metadata.get('original_los') else ''
            }
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_next_pass(self) -> Optional[ScheduledRecording]:
        """Получает следующий пролёт для записи."""
        now = datetime.now()

        for recording in self.recordings:
            if recording.status == 'scheduled' and recording.aos > now:
                return recording

        return None

    def get_status(self) -> Dict:
        """Получает статус планировщика."""
        active = [r for r in self.recordings if r.status == 'recording']
        scheduled = [r for r in self.recordings if r.status == 'scheduled']
        completed = [r for r in self.recordings if r.status == 'completed']
        failed = [r for r in self.recordings if r.status == 'failed']

        return {
            'is_running': self.is_running,
            'active_recordings': len(active),
            'scheduled_recordings': len(scheduled),
            'completed_recordings': len(completed),
            'failed_recordings': len(failed),
            'next_pass': scheduled[0].satellite if scheduled else None,
            'next_pass_time': scheduled[0].aos.isoformat() if scheduled else None
        }
