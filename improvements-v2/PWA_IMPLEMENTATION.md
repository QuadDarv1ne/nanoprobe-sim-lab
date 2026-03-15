# PWA Implementation Guide для Nanoprobe Sim Lab

## Обзор

Progressive Web App (PWA) превращает Next.js приложение в устанавливаемое приложение с офлайн-поддержкой, push-уведомлениями и нативным look & feel.

## Архитектура PWA

```
frontend/
├── public/
│   ├── manifest.json           # PWA manifest
│   ├── sw.js                   # Service Worker
│   ├── icons/                  # Иконки для установки
│   │   ├── icon-72x72.png
│   │   ├── icon-96x96.png
│   │   ├── icon-128x128.png
│   │   ├── icon-144x144.png
│   │   ├── icon-152x152.png
│   │   ├── icon-192x192.png
│   │   ├── icon-384x384.png
│   │   └── icon-512x512.png
│   └── favicon.ico
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Добавить manifest link
│   │   └── page.tsx
│   └── hooks/
│       └── usePWA.ts           # PWA hook
└── next.config.js              # PWA config
```

## 1. Установка зависимостей

```bash
cd frontend
npm install @ducanh2912/next-pwa
```

## 2. Настройка next.config.js

```javascript
// frontend/next.config.js
const withPWA = require('@ducanh2912/next-pwa').default({
  dest: 'public',
  disable: process.env.NODE_ENV === 'development',
  register: true,
  skipWaiting: true,
  runtimeCaching: [
    {
      urlPattern: /^https:\/\/api\.(?:nasa\.gov|open-notify\.org)\//,
      handler: 'CacheFirst',
      options: {
        cacheName: 'nasa-api-cache',
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 60 * 60 * 24, // 24 часа
        },
      },
    },
    {
      urlPattern: /^https:\/\/images\.nasa\.gov\//,
      handler: 'CacheFirst',
      options: {
        cacheName: 'nasa-images-cache',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 60 * 24 * 7, // 7 дней
        },
      },
    },
    {
      urlPattern: /\/api\/v1\//,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        networkTimeoutSeconds: 10,
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 5, // 5 минут
        },
      },
    },
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'static-images',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 60 * 24 * 30, // 30 дней
        },
      },
    },
  ],
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['images.nasa.gov', 'api.nasa.gov'],
  },
};

module.exports = withPWA(nextConfig);
```

## 3. PWA Manifest (public/manifest.json)

```json
{
  "name": "Nanoprobe Sim Lab",
  "short_name": "NanoLab",
  "description": "Лаборатория моделирования нанозонда - научно-образовательный проект для моделирования и анализа нанотехнологий",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0f172a",
  "theme_color": "#3b82f6",
  "orientation": "any",
  "scope": "/",
  "lang": "ru",
  "categories": ["education", "science", "productivity"],
  "icons": [
    {
      "src": "/icons/icon-72x72.png",
      "sizes": "72x72",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-96x96.png",
      "sizes": "96x96",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-128x128.png",
      "sizes": "128x128",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-144x144.png",
      "sizes": "144x144",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-152x152.png",
      "sizes": "152x152",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-384x384.png",
      "sizes": "384x384",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "maskable any"
    }
  ],
  "screenshots": [
    {
      "src": "/screenshots/dashboard.png",
      "sizes": "1920x1080",
      "type": "image/png",
      "form_factor": "wide",
      "label": "Dashboard - Main View"
    },
    {
      "src": "/screenshots/sstv.png",
      "sizes": "1920x1080",
      "type": "image/png",
      "form_factor": "wide",
      "label": "SSTV Ground Station"
    }
  ],
  "shortcuts": [
    {
      "name": "SSTV Станция",
      "short_name": "SSTV",
      "description": "Наземная станция SSTV",
      "url": "/sstv",
      "icons": [{ "src": "/icons/sstv-icon.png", "sizes": "192x192" }]
    },
    {
      "name": "Симулятор СЗМ",
      "short_name": "SPM",
      "description": "Симулятор сканирующей зондовой микроскопии",
      "url": "/spm",
      "icons": [{ "src": "/icons/spm-icon.png", "sizes": "192x192" }]
    },
    {
      "name": "Анализ изображений",
      "short_name": "Analyzer",
      "description": "Анализатор AFM-изображений",
      "url": "/analyzer",
      "icons": [{ "src": "/icons/analyzer-icon.png", "sizes": "192x192" }]
    }
  ],
  "related_applications": [],
  "prefer_related_applications": false,
  "handle_links": "preferred",
  "launch_handler": {
    "client_mode": "navigate-existing"
  },
  "edge_side_panel": {
    "preferred_width": 400
  }
}
```

## 4. Обновление layout.tsx

```typescript
// frontend/src/app/layout.tsx
import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from '@/providers/Providers';

const inter = Inter({ subsets: ['latin', 'cyrillic'] });

export const metadata: Metadata = {
  title: {
    default: 'Nanoprobe Sim Lab',
    template: '%s | Nanoprobe Sim Lab',
  },
  description: 'Лаборатория моделирования нанозонда - научно-образовательный проект',
  applicationName: 'Nanoprobe Sim Lab',
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'NanoLab',
  },
  formatDetection: {
    telephone: false,
  },
  openGraph: {
    type: 'website',
    siteName: 'Nanoprobe Sim Lab',
    title: 'Nanoprobe Sim Lab',
    description: 'Лаборатория моделирования нанозонда',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Nanoprobe Sim Lab',
    description: 'Лаборатория моделирования нанозонда',
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/icons/icon-192x192.png" type="image/png" />
        <link rel="apple-touch-icon" href="/icons/icon-192x192.png" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
      </head>
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

## 5. PWA Hook (usePWA.ts)

```typescript
// frontend/src/hooks/usePWA.ts
'use client';

import { useState, useEffect, useCallback } from 'react';

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

interface PWAStatus {
  isInstalled: boolean;
  isOnline: boolean;
  canInstall: boolean;
  needUpdate: boolean;
  installApp: () => Promise<boolean>;
  updateApp: () => void;
}

export function usePWA(): PWAStatus {
  const [isInstalled, setIsInstalled] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const [canInstall, setCanInstall] = useState(false);
  const [needUpdate, setNeedUpdate] = useState(false);
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [registration, setRegistration] = useState<ServiceWorkerRegistration | null>(null);

  // Проверка установки
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Проверка standalone режима
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches
      || (window.navigator as any).standalone === true;
    setIsInstalled(isStandalone);

    // Online/Offline статус
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    setIsOnline(navigator.onLine);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Слушаем событие beforeinstallprompt
  useEffect(() => {
    const handleBeforeInstall = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      setCanInstall(true);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstall);
    return () => window.removeEventListener('beforeinstallprompt', handleBeforeInstall);
  }, []);

  // Регистрация Service Worker и проверка обновлений
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then((reg) => {
        setRegistration(reg);
      });

      navigator.serviceWorker.addEventListener('controllerchange', () => {
        setNeedUpdate(true);
      });
    }
  }, []);

  // Установка приложения
  const installApp = useCallback(async (): Promise<boolean> => {
    if (!deferredPrompt) return false;

    try {
      await deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;

      if (outcome === 'accepted') {
        setCanInstall(false);
        setIsInstalled(true);
        setDeferredPrompt(null);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Install error:', error);
      return false;
    }
  }, [deferredPrompt]);

  // Обновление приложения
  const updateApp = useCallback(() => {
    if (registration?.waiting) {
      registration.waiting.postMessage({ type: 'SKIP_WAITING' });
      window.location.reload();
    }
  }, [registration]);

  return {
    isInstalled,
    isOnline,
    canInstall,
    needUpdate,
    installApp,
    updateApp,
  };
}
```

## 6. Компонент InstallPrompt

```typescript
// frontend/src/components/pwa/InstallPrompt.tsx
'use client';

import { useState } from 'react';
import { usePWA } from '@/hooks/usePWA';
import { Download, X, Wifi, WifiOff, RefreshCw } from 'lucide-react';

export function InstallPrompt() {
  const { canInstall, isOnline, needUpdate, installApp, updateApp } = usePWA();
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  // Push уведомление об обновлении
  if (needUpdate) {
    return (
      <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-blue-600 text-white p-4 rounded-lg shadow-lg z-50 flex items-center gap-3">
        <RefreshCw className="h-5 w-5 flex-shrink-0" />
        <div className="flex-1">
          <p className="font-medium">Доступно обновление</p>
          <p className="text-sm opacity-90">Новая версия готова к установке</p>
        </div>
        <button
          onClick={updateApp}
          className="px-3 py-1 bg-white text-blue-600 rounded-md text-sm font-medium hover:bg-blue-50 transition-colors"
        >
          Обновить
        </button>
      </div>
    );
  }

  // Индикатор офлайн режима
  if (!isOnline) {
    return (
      <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-amber-600 text-white p-4 rounded-lg shadow-lg z-50 flex items-center gap-3">
        <WifiOff className="h-5 w-5 flex-shrink-0" />
        <div className="flex-1">
          <p className="font-medium">Офлайн режим</p>
          <p className="text-sm opacity-90">Некоторые функции могут быть недоступны</p>
        </div>
      </div>
    );
  }

  // Предложение установки
  if (canInstall) {
    return (
      <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-slate-800 text-white p-4 rounded-lg shadow-lg z-50 flex items-center gap-3">
        <Download className="h-5 w-5 flex-shrink-0" />
        <div className="flex-1">
          <p className="font-medium">Установить приложение</p>
          <p className="text-sm opacity-90">Добавьте NanoLab на главный экран</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setDismissed(true)}
            className="p-1 hover:bg-slate-700 rounded transition-colors"
            aria-label="Отклонить"
          >
            <X className="h-4 w-4" />
          </button>
          <button
            onClick={installApp}
            className="px-3 py-1 bg-blue-500 rounded-md text-sm font-medium hover:bg-blue-600 transition-colors"
          >
            Установить
          </button>
        </div>
      </div>
    );
  }

  return null;
}
```

## 7. Push Notifications Integration

```typescript
// frontend/src/lib/pushNotifications.ts

const VAPID_PUBLIC_KEY = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY || '';

export async function requestNotificationPermission(): Promise<boolean> {
  if (!('Notification' in window)) {
    console.warn('This browser does not support notifications');
    return false;
  }

  const permission = await Notification.requestPermission();
  return permission === 'granted';
}

export async function subscribeToPush(): Promise<PushSubscription | null> {
  if (!('serviceWorker' in navigator)) return null;

  try {
    const registration = await navigator.serviceWorker.ready;
    
    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(VAPID_PUBLIC_KEY),
    });

    // Отправляем subscription на сервер
    await fetch('/api/v1/push/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(subscription.toJSON()),
    });

    return subscription;
  } catch (error) {
    console.error('Push subscription error:', error);
    return null;
  }
}

export async function unsubscribeFromPush(): Promise<boolean> {
  if (!('serviceWorker' in navigator)) return false;

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();
    
    if (subscription) {
      await subscription.unsubscribe();
      
      // Уведомляем сервер об отписке
      await fetch('/api/v1/push/unsubscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpoint: subscription.endpoint }),
      });
    }
    
    return true;
  } catch (error) {
    console.error('Push unsubscription error:', error);
    return false;
  }
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');

  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}
```

## 8. Настройки уведомлений (API Backend)

```python
# api/routes/push.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import webpush
from utils.config_manager import ConfigManager

router = APIRouter(prefix="/push", tags=["Push Notifications"])

class PushSubscription(BaseModel):
    endpoint: str
    keys: dict

class PushNotification(BaseModel):
    title: str
    body: str
    icon: Optional[str] = None
    data: Optional[dict] = None

@router.post("/subscribe")
async def subscribe(subscription: PushSubscription, user_id: str = Depends(get_current_user_id)):
    """Сохранить подписку пользователя для push уведомлений"""
    # Сохраняем в БД или Redis
    await redis_client.set(
        f"push_sub:{user_id}",
        subscription.model_dump_json(),
        ex=30 * 24 * 60 * 60  # 30 дней
    )
    return {"status": "subscribed"}

@router.post("/unsubscribe")
async def unsubscribe(user_id: str = Depends(get_current_user_id)):
    """Отписать пользователя от push уведомлений"""
    await redis_client.delete(f"push_sub:{user_id}")
    return {"status": "unsubscribed"}

@router.post("/send")
async def send_notification(
    notification: PushNotification,
    user_id: str = Depends(get_current_user_id)
):
    """Отправить push уведомление пользователю"""
    subscription_data = await redis_client.get(f"push_sub:{user_id}")
    
    if not subscription_data:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    subscription = json.loads(subscription_data)
    
    try:
        webpush.webpush(
            subscription_info={
                "endpoint": subscription["endpoint"],
                "keys": subscription["keys"]
            },
            data=json.dumps({
                "title": notification.title,
                "body": notification.body,
                "icon": notification.icon,
                "data": notification.data or {}
            }),
            vapid_private_key=config.VAPID_PRIVATE_KEY,
            vapid_claims={
                "sub": f"mailto:{config.VAPID_EMAIL}"
            }
        )
        return {"status": "sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 9. Офлайн-страница

```typescript
// frontend/src/app/offline/page.tsx
'use client';

import { WifiOff, RefreshCw } from 'lucide-react';
import Link from 'next/link';

export default function OfflinePage() {
  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="text-center max-w-md">
        <WifiOff className="h-24 w-24 text-slate-400 mx-auto mb-6" />
        <h1 className="text-3xl font-bold text-white mb-4">
          Вы офлайн
        </h1>
        <p className="text-slate-400 mb-8">
          Проверьте подключение к интернету и попробуйте снова.
          Некоторые функции доступны в офлайн-режиме.
        </p>
        <div className="space-y-4">
          <button
            onClick={() => window.location.reload()}
            className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <RefreshCw className="h-5 w-5" />
            Повторить
          </button>
          <Link
            href="/"
            className="block w-full px-6 py-3 bg-slate-700 text-white rounded-lg font-medium hover:bg-slate-600 transition-colors text-center"
          >
            На главную
          </Link>
        </div>
      </div>
    </div>
  );
}
```

## 10. Генерация иконок

Скрипт для автоматической генерации иконок из SVG:

```bash
# frontend/scripts/generate-icons.sh
#!/bin/bash

SOURCE_SVG="public/logo.svg"
OUTPUT_DIR="public/icons"

mkdir -p $OUTPUT_DIR

for size in 72 96 128 144 152 192 384 512; do
  npx svgexport $SOURCE_SVG $OUTPUT_DIR/icon-${size}x${size}.png $size:$size
done

echo "Icons generated successfully!"
```

Или используйте онлайн-генераторы:
- [PWA Asset Generator](https://github.com/nicholasalx/pwa-asset-generator)
- [RealFaviconGenerator](https://realfavicongenerator.net/)

## Тестирование PWA

### Lighthouse Audit
```bash
# Установка Lighthouse
npm install -g lighthouse

# Запуск аудита
lighthouse http://localhost:3000 --view --preset=pwa
```

### Тестирование офлайн-режима
1. Откройте Chrome DevTools (F12)
2. Application > Service Workers
3. Check "Offline"
4. Перезагрузите страницу

### Проверка установки
1. Откройте приложение в Chrome
2. Адресная строка должна показать иконку установки
3. Или меню > "Install app"

## Деплой PWA

### Vercel
PWA автоматически работает при деплое на Vercel. Убедитесь, что все файлы в `public/` включены.

### Docker
```dockerfile
# Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

## Метрики PWA

| Метрика | Цель | Значение |
|---------|------|----------|
| Lighthouse PWA Score | 100 | 🎯 |
| First Load JS | < 100KB | Оптимизировать |
| Time to Interactive | < 3s | Оптимизировать |
| Offline Ready | ✅ | Service Worker |
| Installable | ✅ | Manifest |
