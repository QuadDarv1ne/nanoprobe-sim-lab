# Frontend Версии - Сравнение

В проекте существуют **две версии frontend**, которые работают параллельно.

---

## 📌 Flask Dashboard (VERSION 1.0 - Legacy/Stable)

**Расположение:** `templates/dashboard.html`  
**Порт:** `http://localhost:5000`  
**Статус:** ✅ Стабильная, проверенная версия

### Технологии:
- Flask (Python)
- Jinja2 шаблоны
- Socket.IO для real-time
- Chart.js 4.x
- Vanilla JavaScript
- Font Awesome иконки

### Файлы:
```
templates/
└── dashboard.html (2536 строк)
```

### Запуск:
```bash
# Через основной скрипт
python start_all.py

# Или напрямую (если доступен web_dashboard.py)
python web_dashboard.py
```

### Особенности:
- ✅ Полностью рабочий
- ✅ Протестированный
- ✅ Многофункциональный
- ❌ Нет TypeScript
- ❌ Нет SSR
- ❌ Большой bundle size

---

## 🚀 Next.js Dashboard (VERSION 2.0 - Modern/Production)

**Расположение:** `frontend/`  
**Порт:** `http://localhost:3000`  
**Статус:** ✅ Новая, современная версия

### Технологии:
- Next.js 14
- TypeScript
- Tailwind CSS
- Zustand (state management)
- Chart.js 4.x + react-chartjs-2
- Lucide React иконки
- WebSocket для real-time
- Sonner (notifications)

### Файлы:
```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── scans/page.tsx
│   │   └── simulations/page.tsx
│   ├── components/
│   │   ├── dashboard-layout.tsx
│   │   ├── header.tsx
│   │   ├── sidebar.tsx
│   │   ├── stats-grid.tsx
│   │   ├── system-health.tsx
│   │   ├── activity-chart.tsx
│   │   ├── recent-activity.tsx
│   │   └── quick-actions.tsx
│   ├── components/ui/
│   │   ├── button.tsx
│   │   ├── alert.tsx
│   │   ├── progress.tsx
│   │   ├── skeleton.tsx
│   │   └── toaster.tsx
│   ├── providers/
│   │   └── theme-provider.ts
│   ├── stores/
│   │   └── dashboard-store.ts
│   └── lib/
│       ├── utils.ts
│       └── config.ts
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

### Запуск:
```bash
cd frontend
npm install
npm run dev
```

### Особенности:
- ✅ Современный UI/UX
- ✅ TypeScript для типобезопасности
- ✅ SSR для лучшей производительности
- ✅ Code splitting
- ✅ Меньший bundle size (~200 KB)
- ✅ Hot reload
- ⚠️ Требует Node.js 18+

---

## 📊 Сравнительная таблица

| Характеристика | Flask (v1.0) | Next.js (v2.0) |
|----------------|--------------|----------------|
| **Порт** | 5000 | 3000 |
| **Язык** | Python + JS | TypeScript + TSX |
| **Рендеринг** | Server-side | Hybrid (SSR + CSR) |
| **Bundle Size** | ~500 KB | ~200 KB |
| **Hot Reload** | ❌ | ✅ |
| **TypeScript** | ❌ | ✅ |
| **Code Splitting** | ❌ | ✅ |
| **SEO** | ❌ | ✅ |
| **PWA Ready** | ❌ | ✅ (готов к добавлению) |
| **Сложность** | Проще | Современнее |
| **Требования** | Python 3.13+ | Node.js 18+ |

---

## 🎯 Какую версию использовать?

### Flask (v1.0) - Используйте если:
- ✅ Нужна проверенная стабильная версия
- ✅ Не хотите устанавливать Node.js
- ✅ Работаете только с Python
- ✅ Не критична производительность frontend

### Next.js (v2.0) - Используйте если:
- ✅ Нужна максимальная производительность
- ✅ Хотите современный UI/UX
- ✅ Планируете расширять frontend
- ✅ Важна типобезопасность (TypeScript)
- ✅ Нужен SSR для SEO

---

## 🔄 Backend (общий для обеих версий)

Обе версии используют **один Backend API**:

**Порт:** `http://localhost:8000`  
**Технологии:** FastAPI + SQLite + Redis  
**Файлы:** `api/`

### Запуск Backend:
```bash
python -m uvicorn api.main:app --reload --port 8000
```

### API Endpoints:
- REST API: `http://localhost:8000/api/v1/...`
- Swagger UI: `http://localhost:8000/docs`
- WebSocket: `ws://localhost:8000/api/v1/dashboard/ws/metrics`

---

## 📁 Архитектура проекта

```
nanoprobe-sim-lab/
│
├── api/                          # FastAPI Backend (ОБЩИЙ)
│   ├── main.py                   # Основное приложение
│   ├── routes/
│   │   ├── enhanced_dashboard.py ← Новые эндпоинты
│   │   ├── scans.py
│   │   ├── simulations.py
│   │   └── ...
│   └── ...
│
├── templates/                    # Flask Frontend (v1.0)
│   └── dashboard.html            # 2536 строк
│
├── frontend/                     # Next.js Frontend (v2.0)
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── providers/
│   │   └── stores/
│   └── package.json
│
└── FRONTEND_VERSIONS.md          # Этот файл
```

---

**Обе версии полностью рабочие и могут использоваться параллельно!** 🎉
