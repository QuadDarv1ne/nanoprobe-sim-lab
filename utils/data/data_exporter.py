"""Модуль экспорта данных для проекта Nanoprobe Simulation Lab."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class DataExporter:
    """Экспорт данных в различные форматы."""

    SUPPORTED_FORMATS = ['csv', 'hdf5', 'json', 'npy']

    def __init__(self, output_dir: str = "output"):
        """Инициализирует экспортер данных."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        data: Union[Dict, List, np.ndarray],
        filename: str,
        fmt: str = 'csv',
        **kwargs
    ) -> Path:
        """
        Экспортирует данные в файл.

        Args:
            data: Данные для экспорта
            filename: Имя файла
            fmt: Формат экспорта (csv, hdf5, json, npy)
            **kwargs: Дополнительные параметры для экспорта

        Returns:
            Path: Путь к сохранённому файлу

        Raises:
            ValueError: Если формат не поддерживается или данные некорректны
        """
        if not filename:
            raise ValueError("Имя файла не может быть пустым")

        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {fmt}. Доступны: {self.SUPPORTED_FORMATS}")

        if data is None:
            raise ValueError("Данные для экспорта не могут быть пустыми")

        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(f'.{fmt}')

        # Проверяем расширение файла
        valid_extensions = {'.csv', '.hdf5', '.h5', '.json', '.npy'}
        if filepath.suffix.lower() not in valid_extensions:
            raise ValueError(f"Неподдерживаемое расширение: {filepath.suffix}")

        try:
            if fmt == 'csv':
                self._export_csv(data, filepath, **kwargs)
            elif fmt == 'hdf5':
                self._export_hdf5(data, filepath, **kwargs)
            elif fmt == 'json':
                self._export_json(data, filepath, **kwargs)
            elif fmt == 'npy':
                self._export_npy(data, filepath, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Ошибка экспорта в {fmt}: {e}")

        return filepath

    def _export_csv(self, data: Union[Dict, List, np.ndarray], filepath: Path, **kwargs):
        """Экспорт в CSV."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for CSV export")

        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type for CSV: {type(data)}")

        df.to_csv(filepath, index=False, **kwargs)

    def _export_hdf5(self, data: Union[Dict, np.ndarray], filepath: Path, **kwargs):
        """Экспорт в HDF5."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for HDF5 export")

        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
            df.to_hdf(filepath, key='data', mode='w', **kwargs)
        elif isinstance(data, dict):
            with pd.HDFStore(filepath, mode='w') as store:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        store.put(key, pd.DataFrame(value))
                    else:
                        store.put(key, pd.Series([value]))
        else:
            raise ValueError(f"Unsupported data type for HDF5: {type(data)}")

    def _export_json(self, data: Union[Dict, List], filepath: Path, **kwargs):
        """Экспорт в JSON."""
        def convert_to_serializable(obj):
            """
            Преобразует объекты в JSON-сериализуемый формат.

            Args:
                obj: Объект для преобразования

            Returns:
            Сериализуемое значение
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable_data = convert_to_serializable(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, **kwargs)

    def _export_npy(self, data: np.ndarray, filepath: Path, **kwargs):
        """Экспорт в NumPy .npy формат."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        np.save(filepath, data, **kwargs)

    def export_surface_data(
        self,
        surface_data: np.ndarray,
        metadata: Optional[Dict] = None,
        filename: str = None,
        fmt: str = 'hdf5'
    ) -> Path:
        """
        Экспортирует данные поверхности с метаданными.

        Args:
            surface_data: Данные поверхности (2D массив)
            metadata: Метаданные (размеры, тип, timestamp)
            filename: Имя файла
            fmt: Формат экспорта

        Returns:
            Path: Путь к файлу
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"surface_{timestamp}"

        export_data = {
            'surface': surface_data,
            'metadata': metadata or {
                'width': surface_data.shape[1],
                'height': surface_data.shape[0],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }

        return self.export(export_data, filename, fmt=fmt)

    def export_scan_results(
        self,
        scan_data: np.ndarray,
        scan_params: Optional[Dict] = None,
        filename: str = None,
        fmt: str = 'csv'
    ) -> Path:
        """
        Экспортирует результаты сканирования.

        Args:
            scan_data: Данные сканирования
            scan_params: Параметры сканирования
            filename: Имя файла
            fmt: Формат экспорта

        Returns:
            Path: Путь к файлу
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"scan_results_{timestamp}"

        if fmt == 'csv':
            export_data = {
                'x': np.repeat(np.arange(scan_data.shape[1]), scan_data.shape[0]),
                'y': np.tile(np.arange(scan_data.shape[0]), scan_data.shape[1]),
                'value': scan_data.flatten()
            }
        else:
            export_data = {
                'scan_data': scan_data,
                'scan_params': scan_params or {}
            }

        return self.export(export_data, filename, fmt=fmt)


class DataImporter:
    """Импорт данных из различных форматов."""

    SUPPORTED_FORMATS = ['csv', 'hdf5', 'json', 'npy']

    def __init__(self, input_dir: str = "data"):
        """Инициализирует импортер данных."""
        self.input_dir = Path(input_dir)

    def import_file(self, filepath: Union[str, Path], fmt: str = None) -> Any:
        """
        Импортирует данные из файла.

        Args:
            filepath: Путь к файлу
            fmt: Формат файла (автоопределение если None)

        Returns:
            Данные
        """
        filepath = Path(filepath)

        if fmt is None:
            fmt = filepath.suffix.lstrip('.').lower()

        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {fmt}")

        if fmt == 'csv':
            return self._import_csv(filepath)
        elif fmt == 'hdf5':
            return self._import_hdf5(filepath)
        elif fmt == 'json':
            return self._import_json(filepath)
        elif fmt == 'npy':
            return self._import_npy(filepath)

    def _import_csv(self, filepath: Path) -> np.ndarray:
        """Импорт из CSV."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for CSV import")
        df = pd.read_csv(filepath)
        return df.values

    def _import_hdf5(self, filepath: Path) -> Dict[str, np.ndarray]:
        """Импорт из HDF5."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for HDF5 import")

        result = {}
        with pd.HDFStore(filepath, mode='r') as store:
            for key in store.keys():
                result[key.lstrip('/')] = store[key].values
        return result

    def _import_json(self, filepath: Path) -> Dict:
        """Импорт из JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _import_npy(self, filepath: Path) -> np.ndarray:
        """Импорт из NumPy .npy формата."""
        return np.load(filepath)
