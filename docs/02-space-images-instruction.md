# Инструкция: Загрузка и печать изображений Hubble

## Содержание
1. [Быстрый старт](#быстрый-старт)
2. [Установка зависимостей](#установка-зависимостей)
3. [Поиск изображений](#поиск-изображений)
4. [Загрузка и обработка](#загрузка-и-обработка)
5. [Печать фотографий](#печать-фотографий)
6. [Примеры использования](#примеры-использования)
7. [Возможные проблемы](#возможные-проблемы)

---

## Быстрый старт

### 1. Запуск одной командой

```bash
cd C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab
python -c "from utils.space_image_downloader import download_and_print_hubble_image; download_and_print_hubble_image('M31')"
```

**Результат:**
- Изображение загружено из Hubble
- Созданы 3 версии для печати в папке `output/print/`

---

## Установка зависимостей

### Основные зависимости

```bash
pip install requests numpy Pillow matplotlib pandas
```

### Для работы с FITS (опционально)

```bash
pip install astropy
```

### Проверка установки

```bash
python -c "from utils.space_image_downloader import SpaceImageDownloader; print('OK')"
```

---

## Поиск изображений

### Hubble Space Telescope

```python
from utils.space_image_downloader import SpaceImageDownloader

downloader = SpaceImageDownloader()

# Поиск по названию объекта
results = downloader.search_hubble(target="M31", pagesize=10)

# Поиск по типу наблюдения
results = downloader.search_hubble(
    target="Crab Nebula",
    observation_type="IMAGE"
)

# Просмотр результатов
for i, item in enumerate(results[:5]):
    print(f"{i+1}. {item.get('target_name', 'Unknown')}")
```

**Популярные объекты:**
| Объект | Запрос | Описание |
|--------|--------|----------|
| M31 | `"M31"` | Туманность Андромеды |
| M42 | `"M42"` | Туманность Ориона |
| M51 | `"M51"` | Галактика Водоворот |
| Crab Nebula | `"Crab Nebula"` | Крабовидная туманность |
| Eagle Nebula | `"Eagle Nebula"` | Туманность Орёл |
| Pillars of Creation | `"Pillars of Creation"` | Столпы творения |

### NASA Image Library

```python
# Поиск в архиве NASA
results = downloader.search_nasa(
    query="Hubble galaxy",
    year_start=2020,
    year_end=2024,
    pagesize=10
)
```

---

## Загрузка и обработка

### Шаг 1: Загрузка изображения

```python
from utils.space_image_downloader import SpaceImageDownloader

downloader = SpaceImageDownloader(download_dir="data/space_images")

# Загрузка по URL
image_path = downloader.download_image(
    url="https://example.com/image.png",
    filename="my_image.png"
)
```

### Шаг 2: Обработка изображения

```python
from utils.space_image_downloader import SpaceImageProcessor

processor = SpaceImageProcessor()

# Загрузка
processor.load_image("data/space_images/my_image.png")

# Улучшение контраста
processor.enhance_contrast(factor=1.5)

# Улучшение яркости
processor.enhance_brightness(factor=1.2)

# Сохранение
processor.save_image("output/enhanced.png", quality=95)
```

### Шаг 3: Создание версий для печати

```python
# Автоматическое создание всех версий
processor.load_image("data/space_images/my_image.png")

versions = processor.create_print_versions(
    base_filename="hubble_m31",
    output_dir="output/print"
)

# Результат:
# versions['color'] -> output/print/hubble_m31_color.png
# versions['black_white'] -> output/print/hubble_m31_bw.png
# versions['false_color'] -> output/print/hubble_m31_false_color.png
```

---

## Печать фотографий

### Windows

```bash
# Печать через командную строку
print /D:"\\ИМЯ_ПРИНТЕРА" output\print\hubble_m31_color.png

# Открыть для просмотра
start output\print\hubble_m31_color.png

# Печать через PowerShell
Start-Process -FilePath "output\print\hubble_m31_color.png" -VerbName Print
```

### Linux

```bash
# Печать
lp -d printer_name output/print/hubble_m31_color.png

# Проверка очереди
lpstat -o

# Открыть для просмотра
xdg-open output/print/hubble_m31_color.png
```

### macOS

```bash
# Печать
lp -d printer_name output/print/hubble_m31_color.png

# Проверка очереди
lpstat -o

# Открыть для просмотра
open output/print/hubble_m31_color.png
```

### Рекомендации по печати

| Параметр | Значение |
|----------|----------|
| Размер бумаги | A4 или Letter |
| Качество | Высокое (300+ DPI) |
| Тип бумаги | Фотобумага |
| Цветовой профиль | sRGB |
| Поля | Минимальные |

---

## Примеры использования

### Пример 1: Полная цепочка для M31

```python
from utils.space_image_downloader import download_and_print_hubble_image

# Одна функция - полный результат
results = download_and_print_hubble_image(
    target="M31",
    output_dir="output/print"
)

print("Файлы готовы:")
for version, path in results.items():
    print(f"  {version}: {path}")
```

### Пример 2: Пакетная загрузка нескольких объектов

```python
from utils.space_image_downloader import SpaceImageDownloader, SpaceImageProcessor

downloader = SpaceImageDownloader()
processor = SpaceImageProcessor()

# Список объектов для загрузки
targets = ["M31", "M42", "M51", "Crab Nebula"]

for target in targets:
    print(f"Загрузка {target}...")
    
    # Поиск
    results = downloader.search_hubble(target, pagesize=1)
    if not results:
        print(f"  Не найдено: {target}")
        continue
    
    # Загрузка
    if 'url' in results[0]:
        path = downloader.download_image(results[0]['url'])
        
        # Обработка
        processor.load_image(str(path))
        processor.create_print_versions(target.lower(), "output/print")
```

### Пример 3: Работа с FITS файлами

```python
from utils.space_image_downloader import FITSReader

fits_reader = FITSReader()

# Чтение FITS файла
if fits_reader.read("data/m31.fits"):
    # Конвертация в изображение
    image = fits_reader.to_image()
    image.save("output/m31_from_fits.png")
    print("FITS конвертирован!")
```

### Пример 4: Применение цветовых карт

```python
from utils.space_image_downloader import SpaceImageProcessor

processor = SpaceImageProcessor()
processor.load_image("data/space_images/image.png")

# Различные цветовые карты
for cmap in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
    processor.apply_color_map(cmap)
    processor.save_image(f"output/{cmap}_colormap.png")
```

---

## Возможные проблемы

### Ошибка: "astropy не установлен"

**Решение:**
```bash
pip install astropy
```

Или используйте упрощённое чтение FITS (без astropy).

---

### Ошибка: "Изображения не найдены"

**Причины:**
- Неправильное название объекта
- Временная недоступность API

**Решение:**
```python
# Попробуйте альтернативное название
results = downloader.search_hubble("Andromeda")  # вместо "M31"

# Или используйте NASA API
results = downloader.search_nasa(query="M31 galaxy")
```

---

### Ошибка: "Таймаут загрузки"

**Решение:**
```python
# Увеличьте таймаут (по умолчанию 60с)
import requests
requests.adapters.DEFAULT_RETRIES = 5

# Или повторите загрузку
path = downloader.download_image(url)
if not path:
    path = downloader.download_image(url)  # Повтор
```

---

### Ошибка: "Порт занят" (для веб-дашборда)

**Решение:**
```bash
# Используйте другой порт
python src/web/web_dashboard.py --port 5001
```

---

### Изображение слишком тёмное

**Решение:**
```python
processor.load_image("image.png")
processor.enhance_brightness(factor=1.5)  # Увеличьте яркость
processor.enhance_contrast(factor=1.3)     # Добавьте контраст
processor.save_image("output/bright.png")
```

---

### Чёрно-белое изображение нечёткое

**Решение:**
```python
from PIL import ImageEnhance

processor.load_image("image.png")
processor.convert_to_grayscale()

# Увеличьте резкость
enhancer = ImageEnhance.Sharpness(processor.processed_image)
sharpened = enhancer.enhance(1.5)
sharpened.save("output/sharp_bw.png")
```

---

## Дополнительные ресурсы

- **Hubble Site**: https://hubblesite.org/
- **ESA Hubble**: https://esahubble.org/
- **NASA Image Library**: https://images.nasa.gov/
- **MAST Portal**: https://mast.stsci.edu/
- **JWST**: https://www.stsci.edu/jwst/

---

## Поддержка

При возникновении проблем:
1. Проверьте логи в папке `logs/`
2. Убедитесь, что все зависимости установлены
3. Проверьте подключение к интернету
4. Попробуйте альтернативный источник данных (NASA вместо Hubble)
