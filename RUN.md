# Универсальный запуск проекта

## 🚀 Быстрый старт

### Интерактивный режим (выбор версии):
```bash
python start.py
```

### Конкретная версия:
```bash
# Flask frontend (v1.0)
python start.py flask

# Next.js frontend (v2.0)
python start.py nextjs

# Только Backend API
python start.py api-only
```

---

## 📌 Версии

### 🔹 Flask Dashboard (v1.0)
- **Порт:** http://localhost:5000
- **Технологии:** Flask + Jinja2 + Socket.IO
- **Запуск:** `python start.py flask`

### 🔹 Next.js Dashboard (v2.0)
- **Порт:** http://localhost:3000
- **Технологии:** Next.js 14 + TypeScript + Tailwind CSS
- **Запуск:** `python start.py nextjs`

### 🔹 Backend API
- **Порт:** http://localhost:8000
- **Swagger:** http://localhost:8000/docs
- **Запуск:** `python start.py api-only`

---

## 📖 Подробная документация

См. [`FRONTEND_VERSIONS.md`](FRONTEND_VERSIONS.md) - полное сравнение версий.
