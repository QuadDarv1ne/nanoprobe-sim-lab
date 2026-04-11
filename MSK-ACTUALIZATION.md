# Автоматическая актуализация данных по МСК

## Обзор изменений

Реализована система автоматической актуализации данных по МСК (Московское время) для **всех расчётов** в проекте. Все модули теперь используют единый источник данных о местоположении и часовом поясе.

## Что было сделано

### 1. Унификация источников
- **Главный менеджер**: `utils/location_manager.py` - единая точка истины для всех данных о местоположении и времени МСК
- **Обёртка для совместимости**: `components/py-sstv-groundstation/src/geolocation.py` теперь импортирует всё из `location_manager.py` вместо дублирования кода

### 2. Обновлённые модули
Следующие файлы обновлены для использования единого location_manager:
- ✅ `components/py-sstv-groundstation/src/geolocation.py` - обёртка с fallback
- ✅ `components/py-sstv-groundstation/src/satellite_tracker.py` - трекинг спутников
- ✅ `components/py-sstv-groundstation/src/main.py` - главный файл SSTV
- ✅ `components/py-sstv-groundstation/src/auto_recorder.py` - автоматическая запись
- ✅ `rtl_sdr_tools/iss_tracker.py` - уже использовал location_manager
- ✅ `api/routes/weather.py` - уже использовал location_manager

### 3. Автоматическое обновление кэша

Добавлены **два механизма** актуализации:

#### 3.1. Фоновое обновление (`auto_refresh=True` по умолчанию)
```python
from utils.location_manager import get_location

# При вызове get_location() кэш проверяется и обновляется в фоне если устарел
loc = get_location()  # auto_refresh=True по умолчанию
```

#### 3.2. Принудительное обновление
```python
from utils.location_manager import refresh_msk_data

# Мгновенное обновление данных МСК
location = refresh_msk_data()
```

### 4. Новые функции

#### `refresh_msk_data()`
Принудительно обновляет данные МСК (координаты и часовой пояс). Используется для гарантированной актуализации всех расчётов.

```python
from utils.location_manager import refresh_msk_data

location = refresh_msk_data()
# Вывод:
# [MSK] Обновление данных геолокации...
# [MSK] ✓ Местоположение обновлено: Odintsovo, Russia
# [MSK]   Координаты: 55.6763°N, 37.2617°E
# [MSK]   Часовой пояс: Moscow (UTC+3)
```

#### Параметр `auto_refresh` в `get_location()`
```python
get_location(force_detect=False, use_env=True, auto_refresh=True)
```
- `force_detect` - принудительное определение по IP
- `use_env` - проверять переменные окружения
- `auto_refresh` - **НОВОЕ**: автоматически обновлять кэш если он устарел (по умолчанию `True`)

## Как это работает

### Приоритет источников данных
1. **Переменные окружения** `GROUND_STATION_LAT` / `GROUND_STATION_LON` (если `use_env=True`)
2. **Кэш геолокации** `data/location_cache.json` (TTL 24 часа)
3. **Автоопределение по IP** (ip-api.com, ipapi.co)
4. **Дефолт**: Москва, МСК (UTC+3)

### Механизм автообновления
```
Вызов get_location()
    ↓
Проверка кэша
    ↓
Кэш устарел? → Да → Запуск фонового обновления
    ↓              ↓
Использовать кэш  Обновление в фоне
                   ↓
                Сохранение в кэш
```

## Использование в коде

### Базовое использование
```python
from utils.location_manager import get_location, now_msk, utc_to_msk

# Получить местоположение (автообновление включено)
loc = get_location()
print(f"Координаты: {loc['lat']}, {loc['lon']}")
print(f"Часовой пояс: {loc['timezone'].name} (UTC+{loc['timezone'].utc_offset})")

# Получить текущее время МСК
msk_time = now_msk()
print(f"Текущее время МСК: {msk_time.strftime('%H:%M:%S')}")

# Конвертировать UTC → МСК
from datetime import datetime, timezone
utc_now = datetime.now(timezone.utc)
msk_time = utc_to_msk(utc_now)
```

### Принудительное обновление
```python
from utils.location_manager import refresh_msk_data

# Обновить данные МСК перед расчётами
location = refresh_msk_data()
if location:
    print(f"✓ Данные обновлены: {location['city']}")
```

### Отключение автообновления
```python
# Если нужно использовать только кэш без фонового обновления
loc = get_location(auto_refresh=False)
```

## Тестирование

Запуск комплексного теста:
```bash
python test_msk_actualization.py
```

Тест проверяет:
1. ✅ Основной location_manager
2. ✅ Обёртку geolocation.py
3. ✅ SatelliteTracker
4. ✅ AutoRecordingScheduler
5. ✅ Механизм кэширования

## Файлы изменений

### Изменённые файлы
- `utils/location_manager.py` - добавлены `refresh_msk_data()`, `_background_location_refresh()`, параметр `auto_refresh`
- `components/py-sstv-groundstation/src/geolocation.py` - переделан на обёртку импорта
- `components/py-sstv-groundstation/src/satellite_tracker.py` - обновлён импорт
- `components/py-sstv-groundstation/src/main.py` - обновлён импорт
- `components/py-sstv-groundstation/src/auto_recorder.py` - обновлён импорт

### Новые файлы
- `test_msk_actualization.py` - комплексный тест системы
- `MSK-ACTUALIZATION.md` - этот файл документации

## Преимущества

1. **Единый источник истины** - все модули используют одни и те же данные
2. **Автоматическая актуализация** - кэш обновляется автоматически при каждом вызове
3. **Фоновое обновление** - не блокирует основной поток выполнения
4. **Обратная совместимость** - старый код продолжает работать без изменений
5. **Отказоустойчивость** - fallback на дефолтные данные если API недоступен
6. **Гибкое управление** - можно отключить автообновление или принудительно обновить данные

## Технические детали

### Кэширование
- **Файл кэша**: `data/location_cache.json`
- **TTL**: 24 часа
- **Формат**: JSON с timestamp в ISO формате
- **Фоновое обновление**: threading.Thread с daemon=True

### Timezone
- **МСК**: UTC+3 (захардкожено в MSK_TZ)
- **Автоопределение**: через zoneinfo для локальных поясов
- **Конвертация**: timedelta-based (не использует pytz)

### Источники геолокации
1. **ip-api.com** - основной (fields: lat, lon, city, country, timezone)
2. **ipapi.co** - fallback (fields: latitude, longitude, city, country_name, utc_offset)

## Примеры использования во всех расчётах

Все расчёты в проекте автоматически используют актуальные данные МСК:

### SSTV Ground Station
```python
# components/py-sstv-groundstation/src/main.py
from utils.location_manager import get_location, now_msk

loc = get_location()  # Автообновление включено
print(f"Время станции: {now_msk()}")
```

### Satellite Tracker
```python
# components/py-sstv-groundstation/src/satellite_tracker.py
tracker = SatelliteTracker(
    ground_station_lat=loc["lat"],    # Актуальная широта
    ground_station_lon=loc["lon"]     # Актуальная долгота
)
# Все расчёты пролётов используют актуальные координаты
```

### Auto Recorder
```python
# components/py-sstv-groundstation/src/auto_recorder.py
scheduler = AutoRecordingScheduler()
# Автоматически использует get_location() с автообновлением
```

### Weather API
```python
# api/routes/weather.py
from utils.location_manager import get_location
_station = get_location()
# Прогноз погоды для актуальных координат станции
```

## Обратная совместимость

Весь старый код продолжает работать без изменений. Обёртка `geolocation.py` предоставляет те же функции:
- `get_location()`
- `now_msk()`
- `utc_to_msk()`
- `force_detect_and_save()`
- `get_location_info()`
- `refresh_msk_data()` **(НОВАЯ)**

## Будущие улучшения

Возможные улучшения:
- [ ] Добавить webhook для уведомления об обновлении кэша
- [ ] Логирование всех обновлений геолокации
- [ ] Поддержка нескольких источников координат одновременно
- [ ] Валидация координат перед сохранением в кэш
- [ ] Настройка TTL через переменные окружения

## Поддержка

При возникновении проблем:
1. Проверьте логи на предмет ошибок геолокации
2. Удалите `data/location_cache.json` для принудительного обновления
3. Запустите `python test_msk_actualization.py` для диагностики
4. Проверьте доступ к ip-api.com и ipapi.co

---

**Дата создания**: 2026-04-11  
**Автор**: Дуплей Максим Игоревич  
**Версия**: 1.0
