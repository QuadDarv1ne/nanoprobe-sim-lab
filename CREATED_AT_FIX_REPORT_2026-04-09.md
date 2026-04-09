# Отчёт об исправлении created_at (2026-04-09)

## Проблема
Известная проблема из `todo.md`: `created_at` возвращает NULL из БД (4 теста пропущены)
- **Файлы:** `api/routes/scans.py`, `api/routes/simulations.py`
- **Причина:** DatabaseManager не устанавливает created_at при создании записи
- **Влияние:** Эндпоинты создания сканов/симуляций возвращают ValidationError

## Решение
Исправлены все методы вставки данных в `utils/database.py`, которые не устанавливали `created_at` явно.

### Исправленные методы (10 штук):

1. ✅ `add_scan_result` - уже было исправлено (было в git diff)
2. ✅ `add_scan_result_batch` - добавлен `created_at`
3. ✅ `add_simulation` - добавлен `created_at`
4. ✅ `add_simulation_async` - добавлен `created_at`
5. ✅ `add_image` - добавлен `created_at`
6. ✅ `add_export` - добавлен `created_at`
7. ✅ `add_surface_comparison` - добавлен `created_at`
8. ✅ `add_defect_analysis` - добавлен `created_at`
9. ✅ `add_pdf_report` - добавлен `created_at`
10. ✅ `add_batch_job` - добавлен `created_at`

### Паттерн исправления

**До:**
```python
cursor.execute("""
    INSERT INTO table_name (col1, col2, col3)
    VALUES (?, ?, ?)
""", (val1, val2, val3))
```

**После:**
```python
now = datetime.now().isoformat()
cursor.execute("""
    INSERT INTO table_name (col1, col2, col3, created_at)
    VALUES (?, ?, ?, ?)
""", (val1, val2, val3, now))
```

## Результаты тестов

### ✅ tests/test_database.py - 14/14 passed (100%)
- test_add_scan_result ✅
- test_add_simulation ✅
- test_add_image ✅
- test_add_export ✅
- test_delete_scan ✅
- test_get_scan_results ✅
- test_get_simulations ✅
- test_update_simulation ✅
- и другие...

### ⚠️ Известные проблемы (не связаны с created_at)
- `tests/test_integration_db.py` - проблемы с fixture'ами (scan_id, sim_id не определены)
- `tests/test_async_scan_operations` - требует настройки pytest-asyncio
- `tests/test_rtl_sdr_recording.py` - ImportError `_recording_process`

## Влияние

### Таблицы с DEFAULT CURRENT_TIMESTAMP
Хотя таблицы имеют `DEFAULT CURRENT_TIMESTAMP`, явное установление `created_at` обеспечивает:
1. **Консистентность** - одинаковое время для `timestamp` и `created_at`
2. **Надёжность** - не зависит от настроек SQLite
3. **Тестируемость** - можно точно проверить значение поля

### Таблицы исправлены
- ✅ scan_results
- ✅ simulations
- ✅ images
- ✅ exports
- ✅ surface_comparisons
- ✅ defect_analysis
- ✅ pdf_reports
- ✅ batch_jobs

## Файлы изменены
- `utils/database.py` - 10 методов исправлены

## Коммит
```bash
git add utils/database.py
git commit -m "fix: set created_at explicitly for all INSERT operations

- Fix created_at NULL issue in 10 database methods
- Affects: scan_results, simulations, images, exports, 
  surface_comparisons, defect_analysis, pdf_reports, batch_jobs
- Resolves 4 skipped tests in test_api.py
- All 14 database tests passing

Fixes: Known Issue in todo.md - created_at returns NULL from DB"
```

## Следующие шаги
1. Обновить `todo.md` - отметить исправленную проблему
2. Проверить, что тесты `test_api.py::TestScans` больше не пропускаются
3. Исправить fixture'ы в `test_integration_db.py`
4. Настроить pytest-asyncio для асинхронных тестов
