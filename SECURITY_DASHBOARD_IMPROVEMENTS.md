# Security & Dashboard Improvements

## Выполненные улучшения (2026-03-15)

---

## 🔐 Security Hardening

### 1. Argon2 Password Hashing ✅

**Файлы:**
- `api/routes/auth.py` — обновлён
- `requirements-api.txt` — обновлён

**Изменения:**
```python
# Было (bcrypt):
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Стало (Argon2 + bcrypt fallback):
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    default="argon2",
    deprecated="auto",
    argon2__memory_cost=65536,  # 64 MB
    argon2__time_cost=3,        # 3 итерации
    argon2__parallelism=4,      # 4 потока
    argon2__type="id",          # Argon2id (гибрид)
)
```

**Преимущества Argon2:**
- 🏆 Победитель Password Hashing Competition
- 💪 Более устойчив к GPU/ASIC атакам чем bcrypt
- 🔧 Гибкая настройка параметров (memory, time, parallelism)
- 🔄 Автоматическая миграция с bcrypt при входе пользователя

**Требования:**
```bash
pip install argon2-cffi passlib[argon2,bcrypt]
```

---

### 2. Audit Logging ✅

**Файлы:**
- `api/routes/auth.py` — добавлен audit logger
- `api/logging_config.py` — добавлен audit handler

**Логгируемые события:**
```python
class AuditEventType(str, Enum):
    LOGIN_SUCCESS = "login.success"
    LOGIN_FAILURE = "login.failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token.refresh"
    TOKEN_REVOKED = "token.revoked"
    _2FA_ENABLED = "2fa.enabled"
    _2FA_DISABLED = "2fa.disabled"
    _2FA_VERIFICATION_FAILED = "2fa.verification_failed"
```

**Формат audit лога (JSON):**
```json
{
  "timestamp": "2026-03-15T10:30:00Z",
  "event_type": "login.success",
  "username": "admin",
  "ip": "192.168.1.1",
  "user_agent": "Mozilla/5.0...",
  "extra": {
    "user_id": 1,
    "role": "admin",
    "auth_method": "password"
  }
}
```

**Файл логов:**
- `logs/api/audit_security.log` — отдельный файл для security событий
- Ротация: 50 MB, 30 файлов (1.5 GB всего)

---

### 3. JWT Refresh Token Rotation ✅

**Статус:** Уже было реализовано, улучшено audit logging

**Функционал:**
- Уникальный `jti` (JWT ID) для каждого refresh токена
- Redis storage для валидации токенов
- Автоматическая ревокация старого токена при refresh
- In-memory fallback если Redis недоступен

**Процесс rotation:**
```
1. Клиент отправляет refresh_token
2. Сервер проверяет jti в Redis
3. Если валиден — создаёт новую пару токенов
4. Старый jti удаляется из Redis (revoke)
5. Новый jti сохраняется в Redis
```

---

## 📊 CLI Dashboard: Модульная Архитектура

### Созданные файлы:

```
src/cli/dashboard/
├── __init__.py                 # Пакет
├── core.py                     # UnifiedDashboard класс
├── widgets/
│   ├── __init__.py
│   ├── base.py                 # Базовый класс Widget
│   ├── system_monitor.py       # CPU, RAM, Disk, Net
│   ├── component_status.py     # API, Frontend, Redis, DB
│   ├── log_viewer.py           # Просмотр логов
│   ├── metrics.py              # API метрики
│   └── activity.py             # Timeline активности
└── layouts/
    ├── __init__.py
    ├── standard.py             # CRITICAL + HIGH виджеты
    ├── enhanced.py             # Все виджеты
    └── minimal.py              # Только CRITICAL
```

### Режимы отображения:

| Режим | Видимые виджеты | Использование |
|-------|----------------|---------------|
| **MINIMAL** | System Monitor, Component Status | Быстрый статус, low bandwidth |
| **STANDARD** | + Log Viewer | Повседневное использование |
| **ENHANCED** | + Metrics, Activity | Полная информация |

### Пример использования:

```python
from src.cli.dashboard import UnifiedDashboard, WidgetMode

# Создание dashboard
dashboard = UnifiedDashboard(
    mode=WidgetMode.ENHANCED,
    theme=DashboardTheme.DARK,
    refresh_interval=5
)

# Запуск
await dashboard.start()

# Смена режима
dashboard.set_mode(WidgetMode.STANDARD)
```

### Запуск из CLI:

```bash
# Enhanced режим (по умолчанию)
python -m src.cli.dashboard.core

# Standard режим
python -m src.cli.dashboard.core standard

# Minimal режим
python -m src.cli.dashboard.core minimal
```

### Виджеты:

#### 1. System Monitor Widget (CRITICAL)
- CPU usage с прогресс баром
- Memory usage + available
- Disk usage + free space
- Network traffic (sent/received)
- Цветовая индикация (зелёный/жёлтый/красный)

#### 2. Component Status Widget (CRITICAL)
- API Server (port 8000)
- Flask Frontend (port 5000)
- Next.js Frontend (port 3000)
- Redis (port 6379)
- PostgreSQL (port 5432)
- Автопроверка доступности

#### 3. Log Viewer Widget (HIGH)
- Последние 10 строк логов
- Фильтрация по уровню (INFO, WARNING, ERROR)
- Подсчёт ошибок

#### 4. Metrics Widget (NORMAL)
- Requests total/per minute
- Average response time
- Cache hit rate
- Active users

#### 5. Activity Widget (NORMAL)
- Timeline последней активности
- Типы: simulation, scan, analysis, system, user
- Relative время ("5m ago")

---

## 🧪 Тесты

### Security Tests:
```bash
pytest tests/test_security_improvements.py -v
```

**Тесты:**
- Argon2 hash format
- Argon2 verification
- bcrypt → Argon2 migration
- Audit event structure
- Audit event types
- JWT token rotation
- Redis token storage
- Password strength validation

### Dashboard Tests:
```bash
pytest tests/test_cli_dashboard.py -v
```

**Тесты:**
- Widget creation
- Widget visibility (modes)
- System monitor refresh
- Component status check
- Log viewer filtering
- Dashboard mode switching
- Widget registration/unregistration

---

## 📋 Checklist

### Security ✅
- [x] Argon2 password hashing
- [x] bcrypt fallback для совместимости
- [x] Audit logging для всех auth событий
- [x] JWT refresh token rotation
- [x] Redis token storage
- [x] Password strength validation

### Dashboard ✅
- [x] Базовый класс Widget
- [x] WidgetPriority (CRITICAL, HIGH, NORMAL, LOW)
- [x] WidgetMode (MINIMAL, STANDARD, ENHANCED)
- [x] SystemMonitorWidget
- [x] ComponentStatusWidget
- [x] LogViewerWidget
- [x] MetricsWidget
- [x] ActivityWidget
- [x] UnifiedDashboard класс
- [x] Layouts (standard, enhanced, minimal)

---

## 🚀 Следующие шаги

Осталось реализовать:

1. **PWA** — offline страница + usePWA hook
2. **PWA** — полный набор иконок
3. **Docker** — docker-compose.prod.yml + nginx
4. **Docker** — Frontend Dockerfile (Next.js)
5. **Database** — Query Analyzer с EXPLAIN
6. **Database** — Composite indexes
7. **Monitoring** — Prometheus/Grafana/Loki
8. **NASA API** — полный клиент
9. **Utils** — реорганизация

---

*Обновлено: 2026-03-15*
