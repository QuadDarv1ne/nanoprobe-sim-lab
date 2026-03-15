# Отчёты и JSON данные

**Структура папок для организованных данных проекта**

---

## 📁 Структура

```
reports/
├── errors/           # Отчёты об ошибках
│   └── error_report_YYYYMMDD_HHMMSS.json
├── exports/          # Экспортированные данные
│   ├── scans/        # Экспорт сканирований
│   ├── simulations/  # Экспорт симуляций
│   └── analysis/     # Экспорт анализов
└── analytics/        # Аналитические отчёты
    ├── daily/        # Ежедневные отчёты
    ├── weekly/       # Еженедельные отчёты
    └── monthly/      # Ежемесячные отчёты
```

---

## 🔧 Настройка

### 1. Обновите `.gitignore`

```gitignore
# Отчёты (игнорируем содержимое)
reports/errors/*.json
reports/exports/**/*.json
reports/analytics/**/*.json

# Но игнорируем саму структуру
!reports/
!reports/.gitkeep
```

### 2. Обновите конфигурацию логирования

В `utils/error_handler.py`:

```python
ERROR_REPORTS_DIR = Path("reports/errors")
ERROR_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
```

### 3. Обновите экспорт данных

В `utils/data_exporter.py`:

```python
EXPORTS_DIR = Path("reports/exports")
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
```

---

## 📊 Типы отчётов

### Errors (`reports/errors/`)

**Формат имени:** `error_report_YYYYMMDD_HHMMSS.json`

**Содержимое:**
```json
{
  "timestamp": "2026-03-15T14:30:22.123456",
  "error_type": "DatabaseError",
  "message": "Failed to connect",
  "traceback": "...",
  "context": {
    "user": "admin",
    "endpoint": "/api/v1/scans",
    "method": "POST"
  },
  "resolved": false
}
```

### Exports (`reports/exports/`)

**Формат имени:** `{type}_export_YYYYMMDD_HHMMSS.{json,csv,pdf}`

**Примеры:**
- `scans_export_20260315_143022.json`
- `simulations_export_20260315_143022.csv`
- `analysis_report_20260315_143022.pdf`

### Analytics (`reports/analytics/`)

**Формат имени:** `{period}_analytics_YYYYMMDD.json`

**Примеры:**
- `daily_analytics_20260315.json`
- `weekly_analytics_20260311.json`
- `monthly_analytics_202603.json`

---

## 🗂️ Автоматическая очистка

### Скрипт очистки (`cleanup_reports.py`)

```python
#!/usr/bin/env python3
"""
Автоматическая очистка старых отчётов
"""

from pathlib import Path
from datetime import datetime, timedelta

REPORTS_DIR = Path("reports")
MAX_AGE_DAYS = 30  # Хранить 30 дней

def cleanup_old_reports():
    """Удаление отчётов старше MAX_AGE_DAYS"""
    cutoff = datetime.now() - timedelta(days=MAX_AGE_DAYS)
    
    for pattern in ["errors/*.json", "exports/**/*.json", "analytics/**/*.json"]:
        for file in REPORTS_DIR.glob(pattern):
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime < cutoff:
                file.unlink()
                print(f"Deleted: {file}")

if __name__ == "__main__":
    cleanup_old_reports()
```

### Планировщик (cron)

```bash
# Запуск очистки каждый день в 3:00
0 3 * * * cd /path/to/nanoprobe-sim-lab && python cleanup_reports.py
```

---

## 📈 Мониторинг

### Проверка размера папки

```bash
# Linux/macOS
du -sh reports/

# Windows
dir reports /s
```

### Проверка количества файлов

```bash
# Linux/macOS
find reports -name "*.json" | wc -l

# Windows
dir reports\*.json /s /b | find /c ".json"
```

---

## 🔒 Безопасность

### .gitignore для reports

```gitignore
# Игнорируем все JSON файлы в reports
reports/**/*.json

# Но сохраняем структуру
!reports/
!reports/.gitkeep
!reports/errors/.gitkeep
!reports/exports/.gitkeep
!reports/analytics/.gitkeep
```

### Исключения (что можно коммитить)

```gitignore
# README для документации
reports/README.md

# Шаблоны отчётов
reports/templates/*.json
```

---

## 📊 Dashboard для просмотра

### API endpoint

```python
@router.get("/reports/list")
async def list_reports(
    type: str = Query("errors"),
    days: int = Query(7)
):
    """Получить список отчётов за последние N дней"""
    reports_dir = REPORTS_DIR / type
    cutoff = datetime.now() - timedelta(days=days)
    
    reports = []
    for file in reports_dir.glob("*.json"):
        if datetime.fromtimestamp(file.stat().st_mtime) >= cutoff:
            reports.append({
                "filename": file.name,
                "size": file.stat().st_size,
                "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat()
            })
    
    return {"reports": reports, "total": len(reports)}
```

---

**Организованное хранение JSON файлов!** 📁
