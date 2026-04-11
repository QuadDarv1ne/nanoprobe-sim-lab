/**
 * PWA Utilities для Nanoprobe Sim Lab
 *
 * Функционал:
 * - Регистрация Service Worker
 * - Проверка обновлений
 * - Offline детекция
 * - Push notifications
 * - Install prompt
 */

'use client';

import { useEffect, useState, useCallback } from 'react';

// ==================== Types ====================

interface PWAState {
  isOnline: boolean;
  isInstalled: boolean;
  updateAvailable: boolean;
  isInstalling: boolean;
  installError: string | null;
}

interface InstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

// ==================== Service Worker Registration ====================

export async function registerServiceWorker() {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });

    console.log('[PWA] Service Worker registered:', registration.scope);

    // Проверка обновлений
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;

      if (newWorker) {
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            // Новое обновление доступно
            window.dispatchEvent(new CustomEvent('pwa-update-available'));
          }
        });
      }
    });

    // Обработка сообщений от SW
    navigator.serviceWorker.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'SYNC_COMPLETE') {
        window.dispatchEvent(new CustomEvent('pwa-sync-complete', { detail: event.data }));
      }
    });

    return registration;
  } catch (error) {
    console.error('[PWA] Service Worker registration failed:', error);
    return null;
  }
}

// ==================== Update Service Worker ====================

export async function updateServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const update = await registration.update();

    console.log('[PWA] Service Worker update check:', update ? 'updated' : 'no update');

    return update;
  } catch (error) {
    console.error('[PWA] Service Worker update failed:', error);
    return false;
  }
}

// ==================== Unregister Service Worker ====================

export async function unregisterServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const success = await registration.unregister();

    console.log('[PWA] Service Worker unregistered:', success);

    // Очистка кэшей
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map(name => caches.delete(name)));

    return success;
  } catch (error) {
    console.error('[PWA] Service Worker unregistration failed:', error);
    return false;
  }
}

// ==================== PWA Installation ====================

export function usePWAInstall() {
  const [deferredPrompt, setDeferredPrompt] = useState<InstallPromptEvent | null>(null);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Проверка установлено ли уже приложение
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsInstalled(true);
    }

    // Обработчик beforeinstallprompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      console.log('[PWA] beforeinstallprompt event fired');
      setDeferredPrompt(e as InstallPromptEvent);
      window.dispatchEvent(new CustomEvent('pwa-install-prompt', { detail: { canInstall: true } }));
    };

    // Обработчик appinstalled
    const handleAppInstalled = () => {
      console.log('[PWA] Application installed');
      setIsInstalled(true);
      setDeferredPrompt(null);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const install = useCallback(async () => {
    if (!deferredPrompt) {
      console.log('[PWA] No install prompt available');
      return { outcome: 'dismissed' as const };
    }

    try {
      await deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;

      console.log('[PWA] Install prompt outcome:', outcome);
      setDeferredPrompt(null);

      return { outcome };
    } catch (error) {
      console.error('[PWA] Install prompt failed:', error);
      return { outcome: 'dismissed' as const };
    }
  }, [deferredPrompt]);

  return {
    isInstalled,
    canInstall: !!deferredPrompt,
    install,
  };
}

// ==================== Online/Offline Detection ====================

export function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      window.dispatchEvent(new CustomEvent('pwa-online'));
    };

    const handleOffline = () => {
      setIsOnline(false);
      window.dispatchEvent(new CustomEvent('pwa-offline'));
    };

    setIsOnline(navigator.onLine);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
}

// ==================== PWA State Hook ====================

export function usePWA() {
  const isOnline = useOnlineStatus();
  const { isInstalled, canInstall, install } = usePWAInstall();
  const [updateAvailable, setUpdateAvailable] = useState(false);

  useEffect(() => {
    const handleUpdateAvailable = () => {
      console.log('[PWA] Update available');
      setUpdateAvailable(true);
    };

    window.addEventListener('pwa-update-available', handleUpdateAvailable);

    return () => {
      window.removeEventListener('pwa-update-available', handleUpdateAvailable);
    };
  }, []);

  const applyUpdate = useCallback(async () => {
    const updated = await updateServiceWorker();
    if (updated) {
      window.location.reload();
    }
  }, []);

  return {
    isOnline,
    isInstalled,
    canInstall,
    install,
    updateAvailable,
    applyUpdate,
  };
}

// ==================== Push Notifications ====================

export async function requestNotificationPermission() {
  if (!('Notification' in window)) {
    console.log('[PWA] Notifications not supported');
    return 'denied';
  }

  try {
    const permission = await Notification.requestPermission();
    console.log('[PWA] Notification permission:', permission);
    return permission;
  } catch (error) {
    console.error('[PWA] Notification permission failed:', error);
    return 'denied';
  }
}

export async function subscribeToPush(vapidPublicKey: string) {
  if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
    throw new Error('Push notifications not supported');
  }

  try {
    const registration = await navigator.serviceWorker.ready;

    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(vapidPublicKey),
    });

    console.log('[PWA] Push subscription:', subscription);
    return subscription;
  } catch (error) {
    console.error('[PWA] Push subscription failed:', error);
    throw error;
  }
}

export async function unsubscribeFromPush() {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();

    if (subscription) {
      const success = await subscription.unsubscribe();
      console.log('[PWA] Push unsubscribed:', success);
      return success;
    }

    return false;
  } catch (error) {
    console.error('[PWA] Push unsubscription failed:', error);
    return false;
  }
}

// ==================== Helpers ====================

function urlBase64ToUint8Array(base64String: string) {
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

// ==================== Offline Page ====================

export function redirectToOffline() {
  if (typeof window !== 'undefined') {
    window.location.href = '/offline';
  }
}
