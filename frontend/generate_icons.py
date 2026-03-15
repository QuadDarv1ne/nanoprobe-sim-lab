"""
Генерация иконок для PWA

Создаёт полный набор иконок всех требуемых размеров.
Требует Pillow: pip install Pillow
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Конфигурация
OUTPUT_DIR = "frontend/public/icons"
SIZES = [
    (72, 72),    # Android Chrome
    (96, 96),    # Android Chrome
    (128, 128),  # Chrome Web Store
    (144, 144),  # Android Chrome
    (152, 152),  # iOS Safari
    (192, 192),  # Android Chrome (home screen)
    (384, 384),  # Android Chrome (splash screen)
    (512, 512),  # Android Chrome (Play Store)
    (1024, 1024) # Play Store feature graphic
]

# Для manifest.json
ICON_SIZES_FOR_MANIFEST = [72, 96, 128, 144, 152, 192, 384, 512]

# Badge sizes
BADGE_SIZES = [72, 96, 128, 192]

# Маска для иконок (круг/квадрат)
MASK_SHAPE = "square"  # "square", "circle", "squircle"


def create_icon(size, filename, variant="main"):
    """
    Создание иконки заданного размера.
    
    Args:
        size: Кортеж (width, height)
        filename: Имя файла
        variant: "main", "maskable", "badge"
    """
    width, height = size
    
    # Создаём изображение с прозрачным фоном
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Параметры дизайна
    padding = width // 8
    inner_size = width - (padding * 2)
    
    if variant == "main":
        # Градиентный фон (синий)
        for i in range(height):
            r = int(59 + (30 * i / height))
            g = int(130 + (20 * i / height))
            b = int(246 + (10 * i / height))
            draw.rectangle([(0, i), (width, i+1)], fill=(r, g, b, 255))
        
        # Белый текст "N" или логотип
        try:
            font_size = int(inner_size * 0.6)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Рисуем букву "N" (Nanoprobe)
        text = "N"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        
    elif variant == "maskable":
        # Маскабируемая иконка (безопасная зона 40%)
        safe_zone = int(width * 0.4)
        
        # Фон
        for i in range(height):
            r = int(59 + (30 * i / height))
            g = int(130 + (20 * i / height))
            b = int(246 + (10 * i / height))
            draw.rectangle([(0, i), (width, i+1)], fill=(r, g, b, 255))
        
        # Логотип в безопасной зоне
        try:
            font_size = int(safe_zone * 0.5)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = "🔬"  # Иконка микроскопа для Nanoprobe
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, font=font)
        
    elif variant == "badge":
        # Badge для notifications (72x72)
        # Круглый фон
        center = width // 2
        radius = center - 4
        draw.ellipse(
            [(center-radius, center-radius), (center+radius, center+radius)],
            fill=(59, 130, 246, 255)
        )
        
        # Белая буква "N"
        try:
            font_size = int(width * 0.5)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = "N"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2 - 2
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    # Сохранение
    img.save(filename, 'PNG')
    print(f"✓ Создано: {filename} ({width}x{height})")


def generate_all_icons():
    """Генерация всех иконок"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Генерация иконок для PWA...")
    print(f"Директория: {OUTPUT_DIR}")
    print()
    
    # Main icons
    print("=== Main Icons ===")
    for size in SIZES:
        filename = os.path.join(OUTPUT_DIR, f"icon-{size[0]}x{size[1]}.png")
        create_icon(size, filename, "main")
    
    # Maskable icons
    print("\n=== Maskable Icons ===")
    for size in SIZES:
        filename = os.path.join(OUTPUT_DIR, f"icon-maskable-{size[0]}x{size[1]}.png")
        create_icon(size, filename, "maskable")
    
    # Badge icons
    print("\n=== Badge Icons ===")
    for size in BADGE_SIZES:
        filename = os.path.join(OUTPUT_DIR, f"badge-{size}x{size}.png")
        create_icon((size, size), filename, "badge")
    
    # Shortcut icons
    print("\n=== Shortcut Icons ===")
    for shortcut in ["dashboard", "sstv", "analysis", "simulations"]:
        filename = os.path.join(OUTPUT_DIR, f"{shortcut}.png")
        create_icon((192, 192), filename, "main")
        print(f"✓ Создано: {filename}")
    
    print("\n✅ Готово!")
    print(f"\nСоздано иконок: {len(SIZES) * 2 + len(BADGE_SIZES) + 4}")


def generate_manifest_icons():
    """Генерация manifest.json с правильными путями"""
    manifest = {
        "name": "Nanoprobe Sim Lab",
        "short_name": "Nanoprobe Lab",
        "description": "Лаборатория моделирования нанозонда - SSTV Ground Station, СЗМ симулятор, AI/ML анализ",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": "#3b82f6",
        "orientation": "portrait-primary",
        "icons": [],
        "categories": ["science", "education", "productivity"],
        "shortcuts": [
            {
                "name": "Dashboard",
                "short_name": "Dashboard",
                "description": "Панель управления",
                "url": "/",
                "icons": [{"src": "/icons/dashboard.png", "sizes": "192x192", "type": "image/png"}]
            },
            {
                "name": "SSTV Station",
                "short_name": "SSTV",
                "description": "SSTV Ground Station",
                "url": "/sstv",
                "icons": [{"src": "/icons/sstv.png", "sizes": "192x192", "type": "image/png"}]
            }
        ],
        "share_target": {
            "action": "/share",
            "method": "POST",
            "enctype": "multipart/form-data",
            "params": {
                "title": "title",
                "text": "text",
                "files": [
                    {
                        "name": "images",
                        "accept": "image/*"
                    }
                ]
            }
        }
    }
    
    # Добавляем основные иконки
    for size in ICON_SIZES_FOR_MANIFEST:
        manifest["icons"].append({
            "src": f"/icons/icon-{size}x{size}.png",
            "sizes": f"{size}x{size}",
            "type": "image/png",
            "purpose": "any"
        })
    
    # Маскабируемые иконки
    for size in [192, 512]:
        manifest["icons"].append({
            "src": f"/icons/icon-maskable-{size}x{size}.png",
            "sizes": f"{size}x{size}",
            "type": "image/png",
            "purpose": "maskable"
        })
    
    # Badge
    manifest["icons"].append({
        "src": "/icons/badge-72x72.png",
        "sizes": "72x72",
        "type": "image/png",
        "purpose": "any"
    })
    
    import json
    manifest_path = os.path.join("frontend/public/manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Обновлён manifest.json")


if __name__ == "__main__":
    generate_all_icons()
    generate_manifest_icons()
