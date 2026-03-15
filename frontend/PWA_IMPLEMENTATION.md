# PWA Implementation Guide

## Progressive Web App для Nanoprobe Sim Lab

**Статус:** ✅ Реализовано (2026-03-15)

---

## 📦 Установленные зависимости

```json
{
  "@ducanh2912/next-pwa": "^10.2.16"
}
```

**Установка:**
```bash
cd frontend
npm install @ducanh2912/next-pwa
```

---

## 🎁 Возможности

### 1. Offline Режим ✅

**Компоненты:**
- `OfflineBanner` — баннер вверху при отсутствии соединения
- `OfflinePage` (`/offline`) — страница offline с информацией

**Использование:**
```tsx
import { OfflineBanner } from '@/components/OfflineBanner';

function App() {
  return (
    <>
      <OfflineBanner />
      {/* остальной контент */}
    </>
  );
}
```

### 2. usePWA Hook ✅

**Файл:** `src/hooks/usePWA.ts`

**API:**
```typescript
const {
  // Install
  canInstall,      // Можно ли установить
  installApp,      // Функция установки
  isInstalled,     // Установлено ли
  platform,        // Платформа (Android, iOS, etc.)
  
  // Online status
  isOnline,        // Статус подключения
  
  // Service Worker
  updateAvailable, // Доступно обновление
  updateServiceWorker, // Функция обновления
  waitingWorker,   // Ожидающий worker
  
  // Push Notifications
  isSupported,     // Поддержка push
  permission,      // Разрешение
  subscription,    // Подписка
  requestPermission, // Запрос разрешения
  subscribeToPush,   // Подписка на push
  unsubscribeFromPush, // Отписка
} = usePWA();
```

**Пример использования:**
```tsx
'use client';

import { usePWA } from '@/hooks/usePWA';

export function PWAControls() {
  const { 
    canInstall, 
    installApp, 
    isOnline,
    updateAvailable,
    updateServiceWorker 
  } = usePWA();

  return (
    <div>
      {/* Кнопка установки */}
      {canInstall && (
        <button onClick={installApp}>
          Установить приложение
        </button>
      )}

      {/* Индикатор online статуса */}
      <div>{isOnline ? '🟢 Online' : '🔴 Offline'}</div>

      {/* Кнопка обновления */}
      {updateAvailable && (
        <button onClick={updateServiceWorker}>
          Обновить приложение
        </button>
      )}
    </div>
  );
}
```

### 3. InstallPrompt Component ✅

**Файл:** `src/components/InstallPrompt.tsx`

**Автоматически отображается** через 5 секунд после загрузки если:
- Приложение можно установить
- Пользователь ещё не устанавливал
- Пользователь не отклонял ранее

**Использование в layout:**
```tsx
import { InstallPrompt } from '@/components/InstallPrompt';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <InstallPrompt />
        {children}
      </body>
    </html>
  );
}
```

### 4. Service Worker ✅

**Файл:** `public/sw.js`

**Функционал:**
- Пре-кэширование статики
- Кэширование API ответов (Cache First + Network fallback)
- Background Sync
- Push Notifications
- Отслеживание последней синхронизации

**Кэшируемые ресурсы:**
```javascript
const STATIC_ASSETS = [
  '/',
  '/offline',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
];

const API_ENDPOINTS = [
  '/api/v1/dashboard/stats',
  '/api/v1/health',
];
```

**Сообщения для Service Worker:**
```javascript
// Обновление worker
navigator.serviceWorker.controller.postMessage({ 
  type: 'SKIP_WAITING' 
});

// Очистка кэша
navigator.serviceWorker.controller.postMessage({ 
  type: 'CLEAR_CACHE' 
});

// Получение последней синхронизации
const channel = new MessageChannel();
channel.port1.onmessage = (event) => {
  console.log('Last sync:', event.data.lastSync);
};
navigator.serviceWorker.controller.postMessage(
  { type: 'GET_LAST_SYNC' },
  [channel.port2]
);
```

---

## 📱 Manifest.json

**Файл:** `public/manifest.json`

**Ключевые поля:**
```json
{
  "name": "Nanoprobe Sim Lab",
  "short_name": "Nanoprobe Lab",
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#3b82f6",
  "background_color": "#0f172a",
  "orientation": "portrait-primary",
  "icons": [...],
  "shortcuts": [
    {
      "name": "Dashboard",
      "url": "/"
    },
    {
      "name": "SSTV Station", 
      "url": "/sstv"
    }
  ],
  "share_target": {...}
}
```

---

## 🔧 Конфигурация Next.js

**Файл:** `next.config.js`

```javascript
const withPWA = require('@ducanh2912/next-pwa').default({
  dest: 'public',
  cacheOnFrontEndNav: true,
  aggressiveFrontEndNavCaching: true,
  reloadOnOnline: true,
  swcMinify: true,
  disable: process.env.NODE_ENV === 'development',
  workboxOptions: {
    disableDevLogs: true,
    skipWaiting: true,
    clientsClaim: true,
  },
});

module.exports = withPWA(nextConfig);
```

---

## 🎨 Визуальные компоненты

### OfflineBanner
- Красный баннер вверху
- Автоматически скрывается при восстановлении
- Кнопка "Повтор"

### OfflinePage
- Градиентный фон
- Статус подключения
- Последняя синхронизация
- Информация о кэше
- Советы пользователю

### InstallPrompt
- Всплывающее окно снизу
- Кнопки "Установить" и "Позжер"
- Список преимуществ
- Рейтинг
- Информация о платформе

---

## 📊 Lighthouse Score

**Ожидаемые результаты:**

| Категория | Score |
|-----------|-------|
| PWA | 100/100 ✅ |
| Performance | 95/100 |
| Accessibility | 98/100 |
| Best Practices | 100/100 |
| SEO | 100/100 |

**Critical для PWA:**
- ✅ Manifest.json
- ✅ Service Worker
- ✅ HTTPS (в production)
- ✅ Offline страница
- ✅ Install prompt

---

## 🧪 Тестирование

### Проверка offline режима:
1. Откройте приложение
2. Откройте DevTools → Network
3. Выберите "Offline"
4. Обновите страницу
5. Должна отобразиться offline страница

### Проверка установки:
1. Откройте в Chrome/Edge
2. В адресной строке появится иконка установки
3. Или используйте `usePWA().installApp()`

### Проверка обновлений:
1. Измените код приложения
2. Перезагрузите страницу
3. Service Worker обнаружит обновление
4. Нажмите "Обновить" для применения

---

## 🚀 Production Checklist

- [ ] HTTPS настроен
- [ ] manifest.json валиден
- [ ] Service Worker работает
- [ ] Offline страница доступна
- [ ] Иконки всех размеров
- [ ] Push notifications настроены (опционально)
- [ ] Background sync работает

---

## 📱 Push Notifications

**Настройка VAPID ключей:**

1. Сгенерируйте ключи:
```bash
npx web-push generate-vapid-keys
```

2. Добавьте в `.env`:
```
VAPID_PUBLIC_KEY=your-public-key
VAPID_PRIVATE_KEY=your-private-key
```

3. Используйте hook:
```tsx
const { subscribeToPush, requestPermission } = usePushNotifications();

async function enableNotifications() {
  await requestPermission();
  const subscription = await subscribeToPush(
    process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY
  );
  
  // Отправьте subscription на сервер
  await fetch('/api/notifications/subscribe', {
    method: 'POST',
    body: JSON.stringify(subscription),
  });
}
```

---

## 🔗 Полезные ссылки

- [next-pwa документация](https://github.com/shadowwalker/next-pwa)
- [Workbox документация](https://developers.google.com/web/tools/workbox)
- [PWA Checklist](https://web.dev/pwa-checklist/)
- [MDN PWA Guide](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)

---

*Обновлено: 2026-03-15*
