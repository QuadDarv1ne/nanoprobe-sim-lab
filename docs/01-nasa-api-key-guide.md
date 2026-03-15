# NASA API Key - Руководство по получению

**Версия:** 1.0  
**Дата:** 2026-03-15  
**Статус:** ✅ Готово к выполнению

---

## 📋 Обзор

Проект использует NASA API для:
- 🛰️ Astronomy Picture of the Day (APOD)
- 🌌 NASA Image and Video Library
- 🌍 Earth Observatory
- 🚀 ISS Tracking Data

**Текущий статус:** Используется `DEMO_KEY` (ограничения: 30 запросов/час)

**Цель:** Получить production ключ (1000 запросов/час)

---

## 🔑 Как получить NASA API ключ

### Шаг 1: Перейдите на портал NASA API

🔗 **URL:** https://api.nasa.gov/

### Шаг 2: Заполните форму регистрации

1. Нажмите **"Sign Up"** или **"Get API Key"**
2. Заполните форму:

```
Full Name: [Ваше имя]
Email Address: [Ваш email]
Organization: Nanoprobe Sim Lab (или ваша организация)
Organization Type: Academic/Research/Commercial
API Key Use: Research and Education
```

3. **API Key Use Case** (пример):
```
Используется для образовательного проекта Nanoprobe Sim Lab - 
лаборатория моделирования нанозонда и приёма данных с МКС.
API применяется для получения астрономических изображений и 
данных о спутниках в научных целях.
```

### Шаг 3: Получите ключ

После отправки формы вы получите:
- **API Key** на email
- Или мгновенно на экране

**Формат ключа:** `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` (40 символов)

---

## ⚙️ Настройка в проекте

### 1. Обновите .env файл

```bash
# Скопируйте .env.example в .env
cp .env.example .env

# Отредактируйте .env
nano .env  # или ваш редактор
```

**Добавьте/обновите:**
```bash
# ==================== External Services ====================
NASA_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  # Ваш ключ!

# Опционально: дополнительные NASA endpoints
NASA_IMAGE_LIBRARY_URL=https://images-api.nasa.gov
NASA_EARTH_OBSERVATORY_URL=https://eoimages.gsfc.nasa.gov
```

### 2. Проверьте ключ

```bash
# Тестовый запрос
curl "https://api.nasa.gov/planetary/apod?api_key=ВАШ_КЛЮЧ"

# Или через API проекта
curl http://localhost:8000/api/v1/external/nasa/apod
```

**Ожидаемый ответ:**
```json
{
  "date": "2026-03-15",
  "explanation": "This is the Astronomy Picture of the Day...",
  "title": "Amazing Space Image",
  "url": "https://apod.nasa.gov/apod/image/..."
}
```

---

## 📊 Сравнение ключей

| Параметр | DEMO_KEY | Production Key |
|----------|----------|----------------|
| Лимит запросов | 30/час | 1000/час |
| Лимит в день | ~720 | ~24,000 |
| Требуется регистрация | ❌ | ✅ |
| Срок действия | Бессрочно | Бессрочно |
| Поддержка | ❌ | ✅ |

---

## 🔒 Безопасность

### .env файл

**НИКОГДА** не коммитьте `.env` в git!

```bash
# Проверьте .gitignore
cat .gitignore | grep env

# Должно быть:
.env
.env.local
.env.production
```

### Production развёртывание

**Вариант 1: Environment Variables**
```bash
export NASA_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
python run_api.py
```

**Вариант 2: Docker Secrets**
```yaml
# docker-compose.yml
services:
  api:
    environment:
      - NASA_API_KEY_FILE=/run/secrets/nasa_api_key
    secrets:
      - nasa_api_key

secrets:
  nasa_api_key:
    file: ./secrets/nasa_api_key.txt
```

**Вариант 3: CI/CD Secrets**
```yaml
# .github/workflows/deploy.yml
env:
  NASA_API_KEY: ${{ secrets.NASA_API_KEY }}
```

---

## 🧪 Тестирование

### Unit тесты

```python
# tests/test_nasa_api.py
import os
import pytest

def test_nasa_api_key_configured():
    """Проверка наличия NASA API ключа"""
    api_key = os.getenv("NASA_API_KEY")
    assert api_key is not None, "NASA_API_KEY не настроен"
    assert api_key != "DEMO_KEY", "Используется DEMO_KEY!"
    assert len(api_key) == 40, "Неверная длина ключа"
```

### Integration тесты

```bash
# Запуск тестов
pytest tests/test_external_services.py -v

# Проверка лимитов
curl -I "https://api.nasa.gov/planetary/apod?api_key=ВАШ_КЛЮЧ"
```

---

## 📈 Мониторинг использования

### Проверка лимитов

NASA API возвращает заголовки:

```bash
curl -I "https://api.nasa.gov/planetary/apod?api_key=ВАШ_КЛЮЧ"
```

**Заголовки:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1647356400
```

### Логирование

Добавьте мониторинг в код:

```python
# api/routes/external_services.py
import logging

logger = logging.getLogger(__name__)

async def get_nasa_apod(date: Optional[str] = None):
    api_key = os.getenv("NASA_API_KEY")
    
    if api_key == "DEMO_KEY":
        logger.warning("⚠️ WARNING: Используется DEMO_KEY!")
    
    # ... остальной код
```

---

## 🐛 Troubleshooting

### Ошибка: "Invalid API Key"

**Причины:**
1. Ключ скопирован с пробелами
2. Ключ не активирован
3. Превышен лимит

**Решение:**
```bash
# Проверьте ключ
echo $NASA_API_KEY | wc -c  # Должно быть 40+1 (newline)

# Пересоздайте ключ на api.nasa.gov
```

### Ошибка: "Rate Limit Exceeded"

**Решение:**
1. Подождите 1 час (сброс лимита)
2. Включите кэширование Redis
3. Уменьшите частоту запросов

```python
# Включите Redis кэш
@cache(prefix="nasa", expire=3600)
async def get_nasa_apod(date: Optional[str] = None):
    # ...
```

### Ошибка: "API Unavailable"

**Причины:**
1. NASA API недоступен
2. Circuit breaker открыт

**Решение:**
```bash
# Проверьте статус NASA API
curl https://api.nasa.gov/

# Проверьте circuit breaker
curl http://localhost:8000/api/v1/external/status
```

---

## 📚 Дополнительные ресурсы

### Официальная документация

- 🔗 https://api.nasa.gov/
- 🔗 https://api.nasa.gov/manual.html
- 🔗 https://images-api.nasa.gov/

### Примеры использования

- 🔗 https://github.com/nasa/api-nasa
- 🔗 https://apod.nasa.gov/apod/astropix.html

### Альтернативные источники данных

| Сервис | URL | Лимит |
|--------|-----|-------|
| NASA Image Library | https://images-api.nasa.gov | 1000/час |
| ESA/Hubble | https://www.spacetelescope.org/images/ | Безлимитно |
| SDSS SkyServer | http://skyserver.sdss.org/ | 10000/час |
| NOAO Data Lab | https://datalab.noao.edu/ | 5000/час |

---

## ✅ Чек-лист выполнения

- [ ] Перейти на https://api.nasa.gov/
- [ ] Заполнить регистрационную форму
- [ ] Получить API ключ на email
- [ ] Обновить `.env` файл
- [ ] Протестировать запрос
- [ ] Включить кэширование Redis
- [ ] Добавить мониторинг лимитов
- [ ] Обновить документацию проекта

---

## 🎯 Следующие шаги

После получения ключа:

1. **Обновите .env.production** для production среды
2. **Настройте мониторинг** использования API
3. **Включите кэширование** для уменьшения запросов
4. **Добавьте fallback** на случай недоступности

---

**Контакт для получения помощи:** support@api.nasa.gov
