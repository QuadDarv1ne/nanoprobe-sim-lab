/**
 * Service Worker для Nanoprobe Sim Lab PWA
 *
 * Функционал:
 * - Кэширование статики (offline режим)
 * - Кэширование API ответов
 * - Background sync для мутаций
 * - Push уведомления
 */

const CACHE_NAME = 'nanoprobe-lab-v1';
const STATIC_CACHE = 'static-v1';
const API_CACHE = 'api-v1';

// Ресурсы для пре-кэширования
const STATIC_ASSETS = [
  '/',
  '/offline',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
];

// API endpoints для кэширования
const API_ENDPOINTS = [
  '/api/v1/dashboard/stats',
  '/api/v1/health',
];

// ==================== Install ====================

self.addEventListener('install', (event) => {
  console.log('[SW] Installing Service Worker...');

  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Pre-caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[SW] Installation complete, skipping waiting');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[SW] Installation error:', error);
      })
  );
});

// ==================== Activate ====================

self.addEventListener('activate', (event) => {
  console.log('[SW] Activating Service Worker...');

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            // Удаление старых кэшей
            if (cacheName !== STATIC_CACHE && cacheName !== API_CACHE) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[SW] Activation complete, claiming clients');
        return self.clients.claim();
      })
  );
});

// ==================== Fetch ====================

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Игнорируем не-GET запросы
  if (request.method !== 'GET') {
    return;
  }

  // API запросы
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }

  // Статические ресурсы
  if (isStaticAsset(url.pathname)) {
    event.respondWith(handleStaticRequest(request));
    return;
  }

  // HTML страницы
  event.respondWith(handleHtmlRequest(request));
});

// ==================== Handlers ====================

async function handleApiRequest(request) {
  const cache = await caches.open(API_CACHE);

  try {
    // Пробуем получить из кэша
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      // Возвращаем кэш + обновляем в фоне
      fetchAndCache(request, cache);
      return cachedResponse;
    }

    // Запрос к сети
    const networkResponse = await fetch(request);

    // Кэшируем успешные ответы
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;

  } catch (error) {
    console.error('[SW] API request failed:', error);

    // Возвращаем fallback если есть
    const fallback = await cache.match(request);
    if (fallback) {
      return fallback;
    }

    return new Response(
      JSON.stringify({ error: 'offline', message: 'No connection' }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

async function handleStaticRequest(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cachedResponse = await cache.match(request);

  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error('[SW] Static request failed:', error);
    return new Response('Resource not available', { status: 404 });
  }
}

async function handleHtmlRequest(request) {
  const cache = await caches.open(STATIC_CACHE);

  try {
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    const networkResponse = await fetch(request);

    if (networkResponse.ok && networkResponse.headers.get('content-type')?.includes('text/html')) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error('[SW] HTML request failed, returning offline page');

    // Возвращаем offline страницу
    const offlinePage = await cache.match('/offline');
    if (offlinePage) {
      return offlinePage;
    }

    return new Response(
      '<html><body><h1>Offline</h1><p>No connection available</p></body></html>',
      {
        status: 503,
        headers: { 'Content-Type': 'text/html' }
      }
    );
  }
}

async function fetchAndCache(request, cache) {
  try {
    const response = await fetch(request);

    if (response.ok) {
      await cache.put(request, response.clone());
    }
  } catch (error) {
    console.log('[SW] Background fetch failed:', error);
  }
}

function isStaticAsset(pathname) {
  const staticExtensions = [
    '.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg',
    '.ico', '.woff', '.woff2', '.ttf', '.eot'
  ];

  return staticExtensions.some(ext => pathname.endsWith(ext));
}

// ==================== Background Sync ====================

self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync triggered:', event.tag);

  if (event.tag === 'sync-data') {
    event.waitUntil(syncData());
  }
});

async function syncData() {
  // Получаем отложенные запросы из IndexedDB
  // и отправляем их на сервер
  console.log('[SW] Syncing data...');

  // TODO: Реализация синхронизации
  const clients = await self.clients.matchAll();
  clients.forEach(client => {
    client.postMessage({
      type: 'SYNC_COMPLETE',
      status: 'success'
    });
  });
}

// ==================== Push Notifications ====================

self.addEventListener('push', (event) => {
  console.log('[SW] Push received:', event);

  const options = {
    body: event.data?.text() || 'New notification',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View',
        icon: '/icons/expand.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icons/close.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Nanoprobe Sim Lab', options)
  );
});

// ==================== Message Handler ====================

self.addEventListener('message', (event) => {
  console.log('[SW] Message received:', event.data);

  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data && event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => caches.delete(cacheName))
        );
      }).then(() => {
        event.ports[0].postMessage({ cleared: true });
      })
    );
  }

  // Обработка запроса на последнюю синхронизацию
  if (event.data && event.data.type === 'GET_LAST_SYNC') {
    const lastSync = localStorage.getItem('lastSync') || null;
    event.ports[0].postMessage({ lastSync });
  }

  // Обновление последней синхронизации
  if (event.data && event.data.type === 'SET_LAST_SYNC') {
    localStorage.setItem('lastSync', new Date().toISOString());
  }
});

// Отслеживание последней синхронизации при успешных запросах
async function trackSync(request, response) {
  try {
    if (response.ok && request.method === 'GET') {
      const now = new Date().toISOString();
      localStorage.setItem('lastSync', now);

      // Уведомляем клиенты о синхронизации
      const clients = await self.clients.matchAll();
      clients.forEach(client => {
        client.postMessage({
          type: 'SYNC_UPDATE',
          timestamp: now
        });
      });
    }
  } catch (error) {
    console.log('[SW] Sync tracking error:', error);
  }
}

console.log('[SW] Service Worker loaded');
