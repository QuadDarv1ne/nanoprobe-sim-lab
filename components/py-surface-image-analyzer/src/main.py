"""Анализатор изображений поверхности для обработки AFM-изображений."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_processor import ImageProcessor, calculate_surface_roughness

# Try to import database
try:
    project_root = Path(__file__).parent.parent.parent.parent
    utils_path = project_root / "utils"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from database import get_database

    HAS_DB = True
except (ImportError, Exception):
    HAS_DB = False


def main():
    """Основная функция анализатора изображений."""
    parser = argparse.ArgumentParser(description="Анализатор изображений поверхности")
    parser.add_argument("--image", "-i", type=str, help="Путь к изображению")
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        default="gaussian",
        choices=["gaussian", "median", "bilateral"],
        help="Тип фильтра для шумоподавления",
    )
    parser.add_argument("--edges", "-e", action="store_true", help="Обнаружить края")
    parser.add_argument("--roughness", "-r", action="store_true", help="Рассчитать шероховатость")
    parser.add_argument("--stats", "-s", action="store_true", help="Показать статистику")
    parser.add_argument("--output", "-o", type=str, help="Путь для сохранения результата")
    parser.add_argument("--output-image", type=str, help="Сохранить обработанное изображение")
    parser.add_argument("--no-db", action="store_true", help="Не сохранять в БД")

    args = parser.parse_args()

    print("=" * 60)
    print("    АНАЛИЗАТОР ИЗОБРАЖЕНИЙ ПОВЕРХНОСТИ")
    print("    Surface Image Analyzer")
    print("=" * 60)

    processor = ImageProcessor()

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Ошибка: Файл '{image_path}' не найден")
            sys.exit(1)

        print(f"Загрузка изображения: {image_path}")
        if not processor.load_image(str(image_path)):
            print("Ошибка загрузки изображения")
            sys.exit(1)

        # Статистика
        if args.stats:
            stats = processor.get_statistics()
            if stats:
                print("\nСтатистика изображения:")
                print(f"  Размер: {stats['shape']}")
                print(f"  Среднее: {stats['mean']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

        # Фильтр
        if args.filter:
            print(f"\nПрименение фильтра: {args.filter}")
            processor.apply_noise_reduction(args.filter)

        # Края
        if args.edges:
            print("Обнаружение краев...")
            edges = processor.detect_edges()
            if edges is not None and args.output:
                edges_path = Path(args.output).with_name(f"edges_{Path(args.output).name}")
                processor.save_image(str(edges_path), edges)
                print(f"Края сохранены: {edges_path}")

        # Шероховатость
        if args.roughness:
            roughness_params = calculate_surface_roughness(
                processor.processed_image
                if processor.processed_image is not None
                else processor.image
            )
            print("\nПараметры шероховатости:")
            print(f"  Ra (среднее): {roughness_params['ra']:.4f}")
            print(f"  Rq (среднеквадратичное): {roughness_params['rq']:.4f}")
            print(f"  Rz (по 10 точкам): {roughness_params['rz']:.4f}")

        # Сохранение обработанного изображения
        if args.output_image:
            processor.save_image(args.output_image)

        # Сохранение в БД
        if HAS_DB and not args.no_db:
            try:
                db = get_database()
                stats = processor.get_statistics()
                roughness = (
                    calculate_surface_roughness(
                        processor.processed_image
                        if processor.processed_image is not None
                        else processor.image
                    )
                    if args.roughness
                    else None
                )

                metadata = {
                    "stats": stats,
                    "roughness": roughness,
                    "filter": args.filter,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }

                scan_id = db.add_scan_result(
                    scan_type="image",
                    surface_type="analyzed",
                    width=stats["shape"][1] if len(stats["shape"]) > 1 else stats["shape"][0],
                    height=stats["shape"][0],
                    file_path=str(image_path),
                    metadata=metadata,
                )
                print(f"\nРезультаты сохранены в БД (ID: {scan_id})")
            except Exception as e:
                print(f"Предупреждение: Не удалось сохранить в БД: {e}")

        print("\nАнализ завершен")

    else:
        print("\nРежим демонстрации:")
        print("  python main.py --image surface.png")
        print("  python main.py -i surface.png -f median -r -s")
        print("  python main.py -i surface.png --edges --output edges.png")


if __name__ == "__main__":
    main()
