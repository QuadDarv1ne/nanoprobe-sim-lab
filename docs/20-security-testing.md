# Security Testing Guide

**Руководство по тестированию безопасности Nanoprobe Sim Lab API**

---

## 📋 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Виды тестов](#виды-тестов)
3. [Примеры использования](#примеры-использования)
4. [Интерпретация результатов](#интерпретация-результатов)
5. [Устранение уязвимостей](#устранение-уязвимостей)

---

## 🚀 Быстрый старт

### Требования

```bash
pip install requests
```

### Запуск теста

```bash
# Полный тест безопасности
python tests/security_test.py

# Быстрый тест (основные проверки)
python tests/security_test.py --quick

# Тест с сохранением отчёта
python tests/security_test.py --report

# Тест удалённого API
python tests/security_test.py --url https://api.nanoprobe-lab.ru
```

---

## 🔍 Виды тестов

### 1. Security Headers

**Проверяемые заголовки:**
- X-Frame-Options (защита от clickjacking)
- X-Content-Type-Options (MIME sniffing защита)
- X-XSS-Protection (XSS защита)
- Referrer-Policy (referrer контроль)
- Content-Security-Policy (CSP)
- Strict-Transport-Security (HSTS)

**Уязвимость:** CWE-1021, CWE-693, CWE-79, CWE-319

---

### 2. SQL Injection

**Тестируемые эндпоинты:**
- POST /api/v1/scans/
- POST /api/v1/simulations/

**Payload примеры:**
```sql
' OR '1'='1
'; DROP TABLE users; --
' UNION SELECT NULL, NULL, NULL --
```

**Уязвимость:** CWE-89 (CVSS: 9.8)

---

### 3. XSS (Cross-Site Scripting)

**Тестируемые эндпоинты:**
- POST /api/v1/scans/
- POST /api/v1/simulations/

**Payload примеры:**
```html
<script>alert('XSS')</script>
<img src=x onerror=alert('XSS')>
<svg onload=alert('XSS')>
```

**Уязвимость:** CWE-79 (CVSS: 7.5)

---

### 4. Authentication Bypass

**Проверка защищённых эндпоинтов:**
- /api/v1/admin/users
- /api/v1/admin/settings
- /api/v1/scans/
- /api/v1/simulations/

**Уязвимость:** CWE-287 (CVSS: 9.0)

---

### 5. Rate Limiting

**Методика:**
- 20 запросов за 5 секунд
- Проверка на 429 Too Many Requests
- Проверка заголовков rate limiting

**Уязвимость:** CWE-770

---

### 6. CORS Misconfiguration

**Тестирование с malicious origins:**
- https://evil.com
- https://attacker.com
- null

**Уязвимость:** CWE-942

---

### 7. Sensitive Data Exposure

**Поиск чувствительных данных:**
- Пароли
- Секреты (secret)
- API ключи
- Токены
- Приватные ключи

**Уязвимость:** CWE-200

---

## 📊 Примеры использования

### 1. Быстрый тест

```bash
python tests/security_test.py --quick
```

**Результат:**
```
======================================================================
🔒 Security Testing: Nanoprobe Sim Lab API
======================================================================
Base URL: http://localhost:8000
Start Time: 2026-03-14 16:00:00
======================================================================

🔐 Аутентификация...
  ℹ️  Аутентификация не удалась (тесты без токена)

📋 Тест 1: Security Headers
  ✅ Все security headers присутствуют

📋 Тест 2: SQL Injection
  ✅ SQL Injection уязвимостей не найдено

📋 Тест 3: XSS (Cross-Site Scripting)
  ✅ XSS уязвимостей не найдено

📋 Тест 4: Authentication Bypass
  ✅ Защищено: /api/v1/admin/users (401)
  ✅ Защищено: /api/v1/admin/settings (401)

📋 Тест 5: Rate Limiting
  ✅ Rate Limiting заголовки присутствуют

📋 Тест 6: CORS Configuration
  ✅ CORS конфигурация безопасна

📋 Тест 7: Sensitive Data Exposure
  ✅ Утечек чувствительных данных не найдено

======================================================================
📊 ИТОГОВЫЙ ОТЧЁТ ПО БЕЗОПАСНОСТИ
======================================================================

📈 Статистика:
   Тестов пройдено: 7/7
   Найдено уязвимостей: 0
   Всего запросов: 45
   Длительность: 12.34с

🎯 Оценка безопасности: ✅ SECURE
💡 Рекомендация: Отличный уровень безопасности!
======================================================================
```

---

### 2. Полный тест с отчётом

```bash
python tests/security_test.py --full --report
```

**Выходные файлы:**
- `tests/security_report.json` - Детальный отчёт

---

### 3. Тест production API

```bash
python tests/security_test.py --url https://api.nanoprobe-lab.ru --timeout 30
```

---

## 📈 Интерпретация результатов

### Уровни критичности

| Уровень | Значок | Описание | Действие |
|---------|--------|----------|----------|
| **CRITICAL** | 🔴 | Критическая уязвимость | Немедленное исправление |
| **HIGH** | 🟠 | Высокая уязвимость | Исправление в течение 24ч |
| **MEDIUM** | 🟡 | Средняя уязвимость | Исправление в течение недели |
| **LOW** | 🟢 | Низкая уязвимость | Исправление по возможности |
| **INFO** | ℹ️ | Информация | Рекомендуется к исправлению |

### Оценка безопасности

| Оценка | Критерий |
|--------|----------|
| ✅ SECURE | 0 уязвимостей |
| 🟢 LOW RISK | Только LOW уязвимости |
| 🟡 MEDIUM RISK | Есть MEDIUM уязвимости |
| 🟠 HIGH RISK | Есть HIGH уязвимости |
| ❌ CRITICAL | Есть CRITICAL уязвимости |

---

## 🔧 Устранение уязвимостей

### 1. Missing Security Headers

**Проблема:** Отсутствуют заголовки безопасности

**Решение:**
```python
# api/security_headers.py уже добавлен
from api.security_headers import setup_security_headers
setup_security_headers(app, production=True)
```

**Проверка:**
```bash
curl -I http://localhost:8000/health
```

---

### 2. SQL Injection

**Проблема:** Прямая интерполяция SQL

**Решение:**
```python
# ❌ НЕПРАВИЛЬНО
cursor.execute(f"SELECT * FROM scans WHERE id = {scan_id}")

# ✅ ПРАВИЛЬНО
cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))

# ✅ ПРАВИЛЬНО (SQLAlchemy)
session.query(Scan).filter(Scan.id == scan_id).first()
```

---

### 3. XSS (Cross-Site Scripting)

**Проблема:** Недостаточное экранирование

**Решение:**
```python
# ✅ Экранирование в JSON ответах
from markupsafe import escape

response_data = {
    "value": escape(user_input)
}

# ✅ Content-Type header
headers["Content-Type"] = "application/json"
```

---

### 4. Authentication Bypass

**Проблема:** Доступ без токена

**Решение:**
```python
# ✅ Dependency injection для аутентификации
from api.dependencies import get_current_user

@app.get("/api/v1/admin/users")
async def get_users(current_user = Depends(get_current_user)):
    # Токлько для авторизованных
```

---

### 5. Rate Limiting

**Проблема:** Нет ограничений на запросы

**Решение:**
```python
# ✅ Rate limiting уже добавлен
from api.rate_limiter import setup_rate_limiter
setup_rate_limiter(app)
```

**Настройка:**
```env
# .env
RATE_LIMIT_DEFAULT=100/minute
RATE_LIMIT_STRICT=10/minute
```

---

### 6. CORS Misconfiguration

**Проблема:** Доступ с любых доменов

**Решение:**
```python
# ✅ Ограничьте доверенные домены
CORS_ORIGINS = [
    "https://nanoprobe-lab.ru",
    "https://www.nanoprobe-lab.ru",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 7. Sensitive Data Exposure

**Проблема:** Чувствительные данные в ответах

**Решение:**
```python
# ✅ Исключите чувствительные поля
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    # password исключён
    
    class Config:
        # Исключить password при сериализации
        exclude = {"password"}
```

---

## 📁 Выходные файлы

### security_report.json

**Структура:**
```json
{
  "timestamp": "2026-03-14T16:00:00",
  "base_url": "http://localhost:8000",
  "summary": {
    "total_tests": 7,
    "passed_tests": 7,
    "total_findings": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "findings": [],
  "results": [
    {
      "test_name": "Security Headers",
      "passed": true,
      "duration_seconds": 0.5,
      "requests_made": 5
    }
  ]
}
```

---

## 🎯 Best Practices

### Перед запуском тестов

1. **Убедитесь, что API доступен:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Запустите тестовую базу данных:**
   ```bash
   python -c "from utils.database import DatabaseManager; print('DB OK')"
   ```

3. **Проверьте зависимости:**
   ```bash
   pip install requests
   ```

### После тестов

1. **Изучите отчёт:**
   ```bash
   cat tests/security_report.json | python -m json.tool
   ```

2. **Исправьте критические уязвимости:**
   - Приоритет: CRITICAL → HIGH → MEDIUM → LOW

3. **Запустите повторно:**
   ```bash
   python tests/security_test.py --report
   ```

---

## 🔗 Дополнительные ресурсы

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [CVSS Calculator](https://www.first.org/cvss/calculator/3.1)

---

**Last Updated:** 2026-03-14  
**Version:** 1.0.0
