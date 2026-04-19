"""
Экспорт данных SSTV/SDR в научные форматы

Поддерживаемые форматы:
- FITS (Flexible Image Transport System) - астрономия
- HDF5 (Hierarchical Data Format) - общие научные данные
- CSV с метаданными - простой анализ
- JSON-LD (Linked Data) - семантическая веб
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from astropy.io import fits

    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

logger = logging.getLogger(__name__)


class SSTVDataExporter:
    """
    Экспорт SSTV/SDR данных в научные форматы.
    """

    def __init__(self, output_dir: str = "output/scientific"):
        """
        Инициализация экспортера.

        Args:
            output_dir: Директория для экспортированных файлов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_sstv_recording(
        self,
        recording_data: Dict[str, Any],
        format: str = "fits",
        output_filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Экспорт SSTV записи.

        Args:
            recording_data: Данные записи
                - image: numpy array изображения
                - iq_data: I/Q данные (complex64)
                - metadata: метаданные записи
            format: Формат экспорта (fits, hdf5, csv, jsonld)
            output_filename: Имя файла (авто если None)

        Returns:
            Путь к файлу или None при ошибке
        """
        if not output_filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_filename = f"sstv_{timestamp}"

        if format == "fits":
            return self._export_to_fits(recording_data, output_filename)
        elif format == "hdf5":
            return self._export_to_hdf5(recording_data, output_filename)
        elif format == "csv":
            return self._export_to_csv(recording_data, output_filename)
        elif format == "jsonld":
            return self._export_to_jsonld(recording_data, output_filename)
        else:
            logger.error(f"Неподдерживаемый формат: {format}")
            return None

    def _export_to_fits(
        self,
        data: Dict[str, Any],
        filename: str,
    ) -> Optional[str]:
        """Экспорт в FITS формат"""
        if not FITS_AVAILABLE:
            logger.error("astropy не установлен: pip install astropy")
            return None

        try:
            output_path = self.output_dir / f"{filename}.fits"

            # Создаём HDU список
            hdul = fits.HDUList()

            # Primary HDU - изображение SSTV
            if "image" in data:
                image_data = np.array(data["image"])
                primary_hdu = fits.PrimaryHDU(image_data)
                primary_hdu.header["OBJECT"] = "SSTV"
                primary_hdu.header["DATE-OBS"] = data.get("metadata", {}).get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                )
                primary_hdu.header["TELESCOP"] = "RTL-SDR V4"
                primary_hdu.header["INSTRUME"] = "SSTV Decoder"

                # Метаданные из записи
                metadata = data.get("metadata", {})
                if "frequency" in metadata:
                    primary_hdu.header["FREQ"] = metadata["frequency"]
                if "mode" in metadata:
                    primary_hdu.header["SSTV_MODE"] = metadata["mode"]
                if "satellite" in metadata:
                    primary_hdu.header["SAT_NAME"] = metadata["satellite"]

                hdul.append(primary_hdu)

            # Extension HDU - I/Q данные
            if "iq_data" in data:
                iq_data = data["iq_data"]
                iq_hdu = fits.ImageHDU(data=np.stack([iq_data.real, iq_data.imag], axis=-1))
                iq_hdu.header["EXTNAME"] = "IQ_DATA"
                iq_hdu.header["BUNIT"] = "complex64"
                hdul.append(iq_hdu)

            # Extension HDU - спектр
            if "spectrum" in data:
                spectrum_hdu = fits.ImageHDU(data=np.array(data["spectrum"]))
                spectrum_hdu.header["EXTNAME"] = "SPECTRUM"
                hdul.append(spectrum_hdu)

            # Сохраняем
            hdul.writeto(str(output_path), overwrite=True)
            logger.info(f"FITS файл сохранён: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка экспорта в FITS: {e}")
            return None

    def _export_to_hdf5(
        self,
        data: Dict[str, Any],
        filename: str,
    ) -> Optional[str]:
        """Экспорт в HDF5 формат"""
        if not HDF5_AVAILABLE:
            logger.error("h5py не установлен: pip install h5py")
            return None

        try:
            output_path = self.output_dir / f"{filename}.h5"

            with h5py.File(str(output_path), "w") as f:
                # Группа метаданных
                meta_group = f.create_group("metadata")
                metadata = data.get("metadata", {})

                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    else:
                        meta_group.attrs[key] = str(value)

                # Системные метаданные
                meta_group.attrs["export_timestamp"] = datetime.now(timezone.utc).isoformat()
                meta_group.attrs["format_version"] = "1.0"
                meta_group.attrs["software"] = "Nanoprobe Sim Lab"

                # Группа изображения
                if "image" in data:
                    img_group = f.create_group("sstv_image")
                    img_group.create_dataset(
                        "data",
                        data=np.array(data["image"]),
                        compression="gzip",
                    )
                    img_group.attrs["format"] = "RGB"

                # Группа I/Q данных
                if "iq_data" in data:
                    iq_group = f.create_group("iq_data")
                    iq_data = data["iq_data"]
                    iq_group.create_dataset(
                        "real",
                        data=iq_data.real,
                        compression="gzip",
                    )
                    iq_group.create_dataset(
                        "imag",
                        data=iq_data.imag,
                        compression="gzip",
                    )
                    iq_group.attrs["dtype"] = "complex64"

                # Группа спектра
                if "spectrum" in data:
                    spec_group = f.create_group("spectrum")
                    spec_group.create_dataset(
                        "power",
                        data=np.array(data["spectrum"]),
                        compression="gzip",
                    )

                # Группа waterfall
                if "waterfall" in data:
                    wf_group = f.create_group("waterfall")
                    wf_group.create_dataset(
                        "data",
                        data=np.array(data["waterfall"]),
                        compression="gzip",
                    )

            logger.info(f"HDF5 файл сохранён: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка экспорта в HDF5: {e}")
            return None

    def _export_to_csv(
        self,
        data: Dict[str, Any],
        filename: str,
    ) -> Optional[str]:
        """Экспорт в CSV формат"""
        try:
            output_path = self.output_dir / f"{filename}.csv"

            # Экспортируем спектральные данные
            if "spectrum" in data:
                spectrum = data["spectrum"]

                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    # Заголовок
                    writer.writerow(
                        [
                            "frequency_hz",
                            "power_db",
                            "timestamp",
                        ]
                    )

                    # Данные
                    for i, power in enumerate(spectrum):
                        freq = i * (
                            data.get("metadata", {}).get("sample_rate", 2400000) / len(spectrum)
                        )
                        writer.writerow(
                            [
                                freq,
                                power,
                                data.get("metadata", {}).get(
                                    "timestamp", datetime.now(timezone.utc).isoformat()
                                ),
                            ]
                        )

            # Экспортируем метаданные в отдельный JSON
            metadata_path = self.output_dir / f"{filename}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(data.get("metadata", {}), f, indent=2)

            logger.info(f"CSV файл сохранён: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка экспорта в CSV: {e}")
            return None

    def _export_to_jsonld(
        self,
        data: Dict[str, Any],
        filename: str,
    ) -> Optional[str]:
        """
        Экспорт в JSON-LD (Linked Data).

        Совместим с семантической вебом и научными репозиториями.
        """
        try:
            output_path = self.output_dir / f"{filename}.jsonld"

            # JSON-LD контекст
            jsonld = {
                "@context": {
                    "@vocab": "https://schema.org/",
                    "sstv": "https://nanoprobe.sim.lab/sstv/",
                    "sdr": "https://nanoprobe.sim.lab/sdr/",
                    "frequency": {"@id": "sdr:frequency", "@type": "Number"},
                    "sample_rate": {"@id": "sdr:sampleRate", "@type": "Number"},
                    "timestamp": {"@id": "sstv:timestamp", "@type": "DateTime"},
                },
                "@type": "sstv:Recording",
                "name": f"SSTV Recording {filename}",
                "description": data.get("metadata", {}).get(
                    "description", "SSTV recording from RTL-SDR"
                ),
                "dateCreated": data.get("metadata", {}).get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "instrument": {
                    "@type": "sdr:Device",
                    "name": "RTL-SDR V4",
                    "sdr:sampleRate": data.get("metadata", {}).get("sample_rate", 2400000),
                    "sdr:centerFrequency": data.get("metadata", {}).get("frequency", 145800000),
                },
            }

            # Добавляем специфичные метаданные
            metadata = data.get("metadata", {})
            if "satellite" in metadata:
                jsonld["about"] = {
                    "@type": "sstv:Satellite",
                    "name": metadata["satellite"],
                }

            if "mode" in metadata:
                jsonld["sstv:mode"] = metadata["mode"]

            # Добавляем ссылки на данные
            image_data = data.get("image")
            image_shape = list(image_data.shape) if hasattr(image_data, "shape") else []

            jsonld["sstv:data"] = {
                "image_shape": image_shape,
                "iq_samples": len(data.get("iq_data", [])) if "iq_data" in data else 0,
            }

            # Сохраняем
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(jsonld, f, indent=2, ensure_ascii=False)

            logger.info(f"JSON-LD файл сохранён: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка экспорта в JSON-LD: {e}")
            return None

    def export_batch(
        self,
        recordings: List[Dict[str, Any]],
        format: str = "hdf5",
    ) -> List[str]:
        """
        Пакетный экспорт нескольких записей.

        Args:
            recordings: Список записей
            format: Формат экспорта

        Returns:
            Список путей к файлам
        """
        exported_files = []

        for i, recording in enumerate(recordings):
            filename = f"batch_{i:04d}"
            result = self.export_sstv_recording(
                recording_data=recording,
                format=format,
                output_filename=filename,
            )

            if result:
                exported_files.append(result)

        logger.info(
            f"Пакетный экспорт завершён: " f"{len(exported_files)}/{len(recordings)} файлов"
        )

        return exported_files


# CLI интерфейс
def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Экспорт SSTV/SDR данных в научные форматы")
    parser.add_argument("--input", required=True, help="Входной JSON файл с данными")
    parser.add_argument(
        "--format",
        choices=["fits", "hdf5", "csv", "jsonld"],
        default="hdf5",
        help="Формат экспорта",
    )
    parser.add_argument("--output", help="Имя выходного файла")
    parser.add_argument("--output-dir", default="output/scientific", help="Директория для вывода")

    args = parser.parse_args()

    # Загрузка данных
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Файл не найден: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Экспорт
    exporter = SSTVDataExporter(output_dir=args.output_dir)
    result = exporter.export_sstv_recording(
        recording_data=data,
        format=args.format,
        output_filename=args.output,
    )

    if result:
        logger.info(f"Экспорт успешен: {result}")
    else:
        logger.error("Ошибка экспорта")


if __name__ == "__main__":
    main()
