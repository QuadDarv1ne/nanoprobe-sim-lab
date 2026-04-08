# Зависимости проекта Nanoprobe Sim Lab

## Структура файлов

| Файл | Размер | Назначение |
|------|--------|------------|
| `requirements.txt` | ~120 пакетов | **Основные** - Backend API + научные пакеты |
| `requirements-full.txt` | ~135 пакетов | **Всё включено** - + Flask, SSTV, dev утилиты |
| `requirements-api.txt` | ~35 пакетов | **Минимальный API** - только FastAPI сервер |
| `requirements-dev.txt` | ~50 пакетов | **Разработка** - линтеры, тесты, профилирование |
| `requirements-sstv.txt` | ~10 пакетов | **SSTV Ground Station** - RTL-SDR + спутники |
| `requirements-flask.txt` | ~10 пакетов | **Flask Frontend** - legacy дашборд |

## Быстрый старт

```bash
# Вариант 1: Полный проект (рекомендуется для разработки)
pip install -r requirements.txt

# Вариант 2: Только API сервер (production, минимальный)
pip install -r requirements-api.txt

# Вариант 3: Всё включено (SSTV, Flask, dev инструменты)
pip install -r requirements-full.txt

# Вариант 4: + инструменты разработки
pip install -r requirements-dev.txt

# Вариант 5: SSTV Ground Station (нужно RTL-SDR устройство)
pip install -r requirements-sstv.txt
```

## Python версии

Поддерживаемые версии: **Python 3.11, 3.12, 3.13, 3.14**

## Почему несколько файлов?

1. **`requirements.txt`** - основной файл для большинства пользователей
2. **`requirements-api.txt`** - минимальный для Docker продакшн деплоя
3. **`requirements-full.txt`** - полный набор для разработки
4. **`requirements-dev.txt`** - только инструменты разработки (через `-r requirements.txt`)
5. **`requirements-sstv.txt`** - опционально для SSTV (требует RTL-SDR hardware)
6. **`requirements-flask.txt`** - опционально для legacy Flask frontend

## Устаревшие файлы (удалить)

- `requirements_ru.txt` - устарел, версии 2-3 года давности
- `requirements_no_rtlsdr.txt` - больше не нужен (RTL-SDR не включён по умолчанию)

## Компоненты

Каждый компонент имеет собственные зависимости:

- `components/py-sstv-groundstation/requirements.txt` - SSTV standalone
- `components/cpp-spm-hardware-sim/requirements.txt` - СЗМ симулятор
- `components/py-surface-image-analyzer/requirements.txt` - Анализ поверхностей

## Changelog зависимостей

### 2026-04-08
- ✅ Обновлены все версии до последних стабильных
- ✅ Добавлены недостающие: `pyotp`, `qrcode`, `pyyaml`, `httpx`, `starlette`
- ✅ Исправлен: `rtlsdr` → `pyrtlsdr` (более поддерживаемый пакет)
- ✅ Добавлен диапазон для numpy: `>=1.26.0,<2.0.0` (избежать breaking changes)
- ✅ RTL-SDR **НЕ** включён по умолчанию (опционально через `requirements-sstv.txt`)
- ✅ Удалены: `pdfkit`, `wkhtmltopdf` из основных (опционально)
