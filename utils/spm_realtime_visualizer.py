"""
Модуль Real-time визуализации СЗМ для проекта Nanoprobe Simulation Lab
Интерактивная визуализация в реальном времени с поддержкой анимации
"""

import base64
import json
import logging
import time
from datetime import datetime, timezone
from io import BytesIO
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class StreamingDataBuffer:
    """
    Буфер для потоковой передачи данных сканирования
    Оптимизирован для работы с большими объёмами данных в реальном времени
    """

    def __init__(self, max_size: int = 1000):
        """
        Инициализация буфера

        Args:
            max_size: Максимальный размер буфера
        """
        self.max_size = max_size
        self.buffer = []
        self.lock = Lock()
        self.total_items_added = 0

    def add_frame(self, frame_data: np.ndarray, timestamp: float = None):
        """
        Добавление кадра в буфер

        Args:
            frame_data: Данные кадра
            timestamp: Временная метка
        """
        with self.lock:
            if timestamp is None:
                timestamp = time.time()

            self.buffer.append(
                {
                    "data": frame_data.copy(),
                    "timestamp": timestamp,
                    "frame_id": self.total_items_added,
                }
            )

            self.total_items_added += 1

            # Ограничение размера буфера
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Получение последнего кадра"""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1].copy()

    def get_frames(self, count: int = 10) -> List[Dict[str, Any]]:
        """Получение последних кадров"""
        with self.lock:
            return self.buffer[-count:].copy()

    def get_fps(self) -> float:
        """Расчёт FPS"""
        with self.lock:
            if len(self.buffer) < 2:
                return 0

            time_diff = self.buffer[-1]["timestamp"] - self.buffer[0]["timestamp"]
            if time_diff <= 0:
                return 0

            return len(self.buffer) / time_diff

    def clear(self):
        """Очистка буфера"""
        with self.lock:
            self.buffer = []
            self.total_items_added = 0

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики буфера"""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "total_frames": self.total_items_added,
                "fps": self.get_fps(),
                "buffer_usage": (
                    len(self.buffer) / self.max_size * 100 if self.max_size > 0 else 0
                ),
            }


class RealTimeSPMVisualizer:
    """
    Визуализатор СЗМ в реальном времени
    Поддержка анимации сканирования, интерактивного управления
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 10),
        colormap: str = "viridis",
        update_interval: int = 100,
    ):
        """
        Инициализация визуализатора

        Args:
            figsize: Размер фигуры
            colormap: Цветовая карта
            update_interval: Интервал обновления (мс)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib не установлен")

        self.figsize = figsize
        self.colormap = colormap
        self.update_interval = update_interval

        self.fig = None
        self.ax = None
        self.im = None
        self.colorbar = None
        self.anim = None

        self.data_lock = Lock()
        self.current_data = None
        self.data_history = []
        self.max_history = 100

        # Параметры сканирования
        self.scan_params = {
            "scan_size": 100,  # нм
            "scan_rate": 1.0,  # Гц
            "resolution": 256,
            "z_range": (0, 10),  # нм
        }

        # Статистика в реальном времени
        self.runtime_metrics = {
            "frames_displayed": 0,
            "last_update": None,
            "fps": 0,
        }

    def create_figure(self, title: str = "СЗМ Real-time Визуализация"):
        """Создание фигуры для визуализации"""
        plt.close("all")

        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle(title, fontsize=14, fontweight="bold")

        # Основная область для изображения
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        self.ax = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_stats = self.fig.add_subplot(gs[0:2, 2])
        self.ax_profile = self.fig.add_subplot(gs[2, 0:2])

        # Настройка осей статистики
        self.ax_stats.axis("off")
        self.stats_text = self.ax_stats.text(
            0.1,
            0.9,
            "",
            transform=self.ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )

        self.fig.canvas.manager.set_window_title("Nanoprobe SPM Real-time Viewer")

    def initialize_plot(self, initial_data: np.ndarray = None):
        """
        Инициализация графика

        Args:
            initial_data: Начальные данные
        """
        if self.fig is None:
            self.create_figure()

        if initial_data is None:
            initial_data = np.zeros(
                (self.scan_params["resolution"], self.scan_params["resolution"])
            )

        with self.data_lock:
            self.current_data = initial_data

        vmin, vmax = self.scan_params["z_range"]

        self.im = self.ax.imshow(
            initial_data,
            cmap=self.colormap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            interpolation="bilinear",
            animated=True,
        )

        self.colorbar = self.fig.colorbar(self.im, ax=self.ax, label="Высота (нм)")

        # Инициализация профиля
        (self.profile_line,) = self.ax_profile.plot([], [], "b-", linewidth=2)
        self.ax_profile.set_xlabel("Позиция X (пиксели)")
        self.ax_profile.set_ylabel("Высота (нм)")
        self.ax_profile.set_title("Профиль поверхности")
        self.ax_profile.grid(True, alpha=0.3)

        self._update_stats({})

    def update_data(self, new_data: np.ndarray, metadata: Dict = None):
        """
        Обновление данных

        Args:
            new_data: Новые данные поверхности
            metadata: Метаданные
        """
        with self.data_lock:
            self.current_data = new_data.copy()

            # Сохранение в историю
            self.data_history.append(
                {
                    "data": new_data.copy(),
                    "timestamp": datetime.now(timezone.utc),
                    "metadata": metadata or {},
                }
            )

            # Ограничение истории
            if len(self.data_history) > self.max_history:
                self.data_history.pop(0)

    def _update_display(self, frame: int) -> list:
        """
        Обновление отображения (для анимации)

        Args:
            frame: Номер кадра

        Returns:
            Список обновлённых художественных объектов
        """
        with self.data_lock:
            if self.current_data is None:
                return [self.im]

            data = self.current_data

        # Обновление изображения
        self.im.set_array(data)

        # Обновление профиля (среднее по Y)
        profile = np.mean(data, axis=0)
        self.profile_line.set_data(np.arange(len(profile)), profile)

        # Авто масштабирование профиля
        self.ax_profile.relim()
        self.ax_profile.autoscale_view()

        # Обновление статистики
        stats = self._calculate_stats(data)
        self._update_stats(stats)

        # Обновление метрик
        self.runtime_metrics["frames_displayed"] += 1
        self.runtime_metrics["last_update"] = datetime.now(timezone.utc)

        return [self.im, self.profile_line, self.stats_text]

    def _calculate_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Расчёт статистики данных"""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "rms": float(np.sqrt(np.mean(data**2))),
        }

    def _update_stats(self, stats: Dict[str, float]):
        """Обновление отображения статистики"""
        if not stats:
            stats_text = "Ожидание данных..."
        else:
            stats_text = (
                f"Статистика поверхности:\n"
                f"─────────────────────\n"
                f"Среднее:     {stats.get('mean', 0):.4f} нм\n"
                f"СКО:         {stats.get('std', 0):.4f} нм\n"
                f"Мин:         {stats.get('min', 0):.4f} нм\n"
                f"Макс:        {stats.get('max', 0):.4f} нм\n"
                f"RMS:         {stats.get('rms', 0):.4f} нм\n"
                f"─────────────────────\n"
                f"Кадры:       {self.runtime_metrics['frames_displayed']}\n"
                f"FPS:         {self.runtime_metrics['fps']:.1f}"
            )

        self.stats_text.set_text(stats_text)

    def start_animation(self, data_source: Callable = None):
        """
        Запуск анимации

        Args:
            data_source: Функция-источник данных (опционально)
        """
        if self.fig is None:
            self.initialize_plot()

        def animate(frame):
            """
            Функция анимации для обновления данных

            Args:
                frame: Номер текущего кадра
            """
            if data_source:
                try:
                    new_data = data_source(frame)
                    self.update_data(new_data)
                except Exception as e:
                    print(f"Ошибка получения данных: {e}")
            return self._update_display(frame)

        self.anim = animation.FuncAnimation(
            self.fig,
            animate,
            interval=self.update_interval,
            blit=True,
            cache_frame_data=False,
        )

        plt.show()

    def stop_animation(self):
        """Остановка анимации"""
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None

    def save_frame(self, filepath: str = None) -> str:
        """
        Сохранение текущего кадра

        Args:
            filepath: Путь для сохранения

        Returns:
            Путь сохранённого файла
        """
        if filepath is None:
            filepath = (
                f"output/spm_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
            )

        if self.fig:
            self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
            return filepath
        return ""

    def export_data(self, filepath: str = None) -> str:
        """
        Экспорт текущих данных

        Args:
            filepath: Путь для экспорта

        Returns:
            Путь сохранённого файла
        """
        if filepath is None:
            filepath = f"output/spm_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.npy"

        with self.data_lock:
            if self.current_data is not None:
                np.save(filepath, self.current_data)

                # Сохранение метаданных
                meta_path = filepath.replace(".npy", "_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "scan_params": self.scan_params,
                            "shape": self.current_data.shape,
                        },
                        f,
                        indent=2,
                    )

                return filepath
        return ""


class SPMScanSimulator:
    """
    Симулятор сканирования СЗМ в реальном времени
    Генерация данных сканирования с прогрессом
    """

    def __init__(
        self,
        surface_generator: Callable = None,
        scan_speed: float = 1.0,
        noise_level: float = 0.01,
    ):
        """
        Инициализация симулятора

        Args:
            surface_generator: Функция генерации поверхности
            scan_speed: Скорость сканирования
            noise_level: Уровень шума
        """
        self.surface_generator = surface_generator or self._default_surface
        self.scan_speed = scan_speed
        self.noise_level = noise_level

        self.resolution = 256
        self.scan_progress = 0
        self.is_scanning = False

        self._surface = None
        self._scanned_area = None

    def _default_surface(self, size: int = 256) -> np.ndarray:
        """Поверхность по умолчанию"""
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)

        # Комбинация различных функций
        Z = (
            np.sin(3 * np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2) / 2)
            + 0.3 * np.cos(5 * X) * np.sin(5 * Y)
            + 0.1 * np.random.randn(size, size)
        )

        return Z

    def start_scan(self, resolution: int = 256):
        """
        Начало сканирования

        Args:
            resolution: Разрешение сканирования
        """
        self.resolution = resolution
        self.scan_progress = 0
        self.is_scanning = True

        # Генерация полной поверхности
        self._surface = self.surface_generator(resolution)
        self._scanned_area = np.zeros((resolution, resolution))

    def get_next_frame(self) -> np.ndarray:
        """
        Получение следующего кадра сканирования

        Returns:
            Данные текущего кадра
        """
        if not self.is_scanning or self._surface is None:
            return np.zeros((self.resolution, self.resolution))

        # Имитация прогресса сканирования (строка за строкой)
        lines_per_frame = max(1, int(self.scan_speed * 5))
        start_line = int(self.scan_progress * self.resolution / 100)
        end_line = min(start_line + lines_per_frame, self.resolution)

        # Копирование отсканированных строк
        self._scanned_area[start_line:end_line] = self._surface[start_line:end_line]

        # Добавление шума
        noise = np.random.randn(*self._scanned_area.shape) * self.noise_level
        result = self._scanned_area + noise

        # Обновление прогресса
        self.scan_progress = 100 * end_line / self.resolution

        if self.scan_progress >= 100:
            self.is_scanning = False

        return result

    def get_scan_progress(self) -> float:
        """Получение прогресса сканирования"""
        return self.scan_progress


class WebSocketVisualizer:
    """
    Визуализатор для веб-интерфейса (WebSocket)
    Отправка данных в веб-панель в реальном времени
    """

    def __init__(self, socketio_client=None):
        """
        Инициализация

        Args:
            socketio_client: Клиент Socket.IO
        """
        self.socketio = socketio_client
        self.update_interval = 0.1  # секунды
        self.last_emit = 0

    def emit_surface_data(
        self, data: np.ndarray, event: str = "surface_update", downsample: int = 4
    ):
        """
        Отправка данных поверхности

        Args:
            data: Данные поверхности
            event: Название события
            downsample: Фактор уменьшения разрешения
        """
        if self.socketio is None:
            return

        # Уменьшение разрешения для передачи
        if downsample > 1:
            h, w = data.shape
            small_data = data[::downsample, ::downsample]
        else:
            small_data = data

        # Преобразование в список
        data_list = small_data.tolist()

        # Статистика
        stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Отправка
        self.socketio.emit(
            event,
            {
                "data": data_list,
                "stats": stats,
                "shape": small_data.shape,
            },
        )

    def emit_scan_progress(self, progress: float):
        """
        Отправка прогресса сканирования

        Args:
            progress: Прогресс (0-100)
        """
        if self.socketio:
            self.socketio.emit("scan_progress", {"progress": progress})


# Глобальные функции для быстрой визуализации
def visualize_spm_realtime(
    data_source: Callable = None,
    title: str = "СЗМ Real-time",
    colormap: str = "viridis",
):
    """
    Быстрая визуализация СЗМ в реальном времени

    Args:
        data_source: Источник данных
        title: Заголовок
        colormap: Цветовая карта
    """
    visualizer = RealTimeSPMVisualizer(colormap=colormap)
    visualizer.create_figure(title)

    if data_source:
        visualizer.start_animation(data_source)
    else:
        # Симулятор по умолчанию
        simulator = SPMScanSimulator()
        simulator.start_scan()
        visualizer.start_animation(lambda f: simulator.get_next_frame())


def create_spm_visualization(
    surface_data: np.ndarray, save_path: str = None, show_profile: bool = True
) -> plt.Figure:
    """
    Создание статической визуализации СЗМ

    Args:
        surface_data: Данные поверхности
        save_path: Путь для сохранения
        show_profile: Показывать профиль

    Returns:
        Figure matplotlib
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib не установлен")

    fig, axes = plt.subplots(1, 2 if show_profile else 1, figsize=(14, 6))

    if show_profile:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # 2D визуализация
    im = ax1.imshow(surface_data, cmap="viridis", aspect="equal")
    ax1.set_title("2D Визуализация")
    ax1.set_xlabel("X (пиксели)")
    ax1.set_ylabel("Y (пиксели)")
    plt.colorbar(im, ax=ax1, label="Высота (нм)")

    if show_profile:
        # Профиль
        profile_x = np.mean(surface_data, axis=0)
        profile_y = np.mean(surface_data, axis=1)

        ax2.plot(profile_x, label="Среднее по Y")
        ax2.plot(profile_y, label="Среднее по X")
        ax2.set_title("Профиль поверхности")
        ax2.set_xlabel("Позиция")
        ax2.set_ylabel("Высота (нм)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Тестирование
    print("=== Тестирование Real-time визуализации СЗМ ===")

    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib не установлен. pip install matplotlib")
    else:
        # Создание визуализатора
        visualizer = RealTimeSPMVisualizer(update_interval=100)

        # Симулятор
        simulator = SPMScanSimulator(scan_speed=2.0, noise_level=0.02)
        simulator.start_scan(resolution=256)

        print("\nЗапуск визуализации...")
        print("Закройте окно для завершения")

        # Запуск с симулятором
        visualizer.create_figure("СЗМ Real-time Визуализация (Тест)")
        visualizer.start_animation(lambda f: simulator.get_next_frame())


class RealTimeSPMWebSocketAdapter:
    """
    Адаптер для передачи данных СЗМ через WebSocket
    Конвертация данных в формат для веб-клиентов
    """

    def __init__(self, visualizer: RealTimeSPMVisualizer = None):
        """
        Инициализация адаптера

        Args:
            visualizer: Визуализатор для подключения
        """
        self.visualizer = visualizer
        self.buffer = StreamingDataBuffer(max_size=500)
        self.connected_clients = set()

    def attach_visualizer(self, visualizer: RealTimeSPMVisualizer):
        """Подключение визуализатора"""
        self.visualizer = visualizer

    def process_frame(self, frame_data: np.ndarray, timestamp: float = None):
        """
        Обработка кадра для передачи

        Args:
            frame_data: Данные кадра
            timestamp: Временная метка
        """
        # Добавление в буфер
        self.buffer.add_frame(frame_data, timestamp)

        # Отправка клиентам
        if self.connected_clients:
            message = self._create_frame_message(frame_data, timestamp)
            self._broadcast(message)

    def _create_frame_message(self, frame_data: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Создание сообщения для клиента"""
        # Сжатие данных (уменьшение разрешения для передачи)
        if frame_data.shape[0] > 128:
            step = frame_data.shape[0] // 128
            compressed = frame_data[::step, ::step]
        else:
            compressed = frame_data

        # Нормализация
        normalized = (compressed - compressed.min()) / (compressed.max() - compressed.min() + 1e-10)

        # Конвертация в base64 для передачи
        img_buffer = BytesIO()
        img = Image.fromarray((normalized * 255).astype(np.uint8))
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        return {
            "type": "spm_frame",
            "timestamp": timestamp or time.time(),
            "frame_id": self.buffer.total_items_added - 1,
            "image_base64": img_base64,
            "shape": list(compressed.shape),
            "min_value": float(compressed.min()),
            "max_value": float(compressed.max()),
            "stats": {
                "mean": float(np.mean(compressed)),
                "std": float(np.std(compressed)),
                "rms": float(np.sqrt(np.mean(compressed**2))),
            },
        }

    def _broadcast(self, message: Dict[str, Any]):
        """Рассылка сообщения клиентам"""
        # Заглушка для реальной реализации WebSocket

    def add_client(self, client_id: str):
        """Добавление клиента"""
        self.connected_clients.add(client_id)

    def remove_client(self, client_id: str):
        """Удаление клиента"""
        self.connected_clients.discard(client_id)

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Получение статистики буфера"""
        return self.buffer.get_stats()

    def get_latest_frame_data(self) -> Optional[Dict[str, Any]]:
        """Получение последнего кадра"""
        return self.buffer.get_latest_frame()

    def export_frame_to_json(self, frame_id: int = None) -> Optional[str]:
        """
        Экспорт кадра в JSON

        Args:
            frame_id: ID кадра (None для последнего)

        Returns:
            JSON строка
        """
        if frame_id is None:
            frame = self.buffer.get_latest_frame()
        else:
            frames = self.buffer.get_frames(frame_id + 1)
            frame = frames[0] if frames and frames[0]["frame_id"] == frame_id else None

        if not frame:
            return None

        result = {
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "shape": list(frame["data"].shape),
            "min": float(frame["data"].min()),
            "max": float(frame["data"].max()),
            "mean": float(np.mean(frame["data"])),
            "std": float(np.std(frame["data"])),
        }

        return json.dumps(result, indent=2)
