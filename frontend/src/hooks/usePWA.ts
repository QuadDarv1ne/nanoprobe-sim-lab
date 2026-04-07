/**
 * usePWA Hook
 * 
 * React hook для управления PWA функционалом:
 * - Установка приложения
 * - Проверка статуса online/offline
 * - Обновление service worker
 * - Push notifications
 */

'use client';

import { useState, useEffect, useCallback } from 'react';

// Типы для PWA events
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

interface InstallResult {
  canInstall: boolean;
  installApp: () => Promise<void>;
  isInstalled: boolean;
  platform: string | null;
}

interface UpdateResult {
  updateAvailable: boolean;
  updateServiceWorker: () => Promise<void>;
  waitingWorker: ServiceWorker | null;
}

/**
 * Hook для установки PWA приложения
 */
export function useInstallPWA(): InstallResult {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Проверка установленности
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsInstalled(true);
    }

    // Обработчик beforeinstallprompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    // Обработчик appinstalled
    const handleAppInstalled = () => {
      setIsInstalled(true);
      setDeferredPrompt(null);
    };

    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const canInstall = deferredPrompt !== null && !isInstalled;

  const installApp = useCallback(async () => {
    if (!deferredPrompt) {
      console.warn('Install prompt not available');
      return;
    }

    try {
      await deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      
      if (outcome === 'accepted') {
        console.log('User accepted the install prompt');
      } else {
        console.log('User dismissed the install prompt');
      }
      
      setDeferredPrompt(null);
    } catch (error) {
      console.error('Error during install prompt:', error);
    }
  }, [deferredPrompt]);

  // Определение платформы
  const platform = (() => {
    if (/Android/.test(navigator.userAgent)) return 'Android';
    if (/iPhone|iPad|iPod/.test(navigator.userAgent)) return 'iOS';
    if (/Windows/.test(navigator.userAgent)) return 'Windows';
    if (/Mac/.test(navigator.userAgent)) return 'macOS';
    if (/Linux/.test(navigator.userAgent)) return 'Linux';
    return null;
  })();

  return {
    canInstall,
    installApp,
    isInstalled,
    platform,
  };
}

/**
 * Hook для проверки online/offline статуса
 */
export function useOnlineStatus(): boolean {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const updateOnlineStatus = () => {
      setIsOnline(navigator.onLine);
    };

    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);

    // Начальная проверка
    setIsOnline(navigator.onLine);

    return () => {
      window.removeEventListener('online', updateOnlineStatus);
      window.removeEventListener('offline', updateOnlineStatus);
    };
  }, []);

  return isOnline;
}

/**
 * Hook для обновления Service Worker
 */
export function useServiceWorker(): UpdateResult {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);

  useEffect(() => {
    if ('serviceWorker' in navigator) {
      const handleControllerChange = () => {
        setUpdateAvailable(false);
        setWaitingWorker(null);
      };
      
      navigator.serviceWorker.addEventListener('controllerchange', handleControllerChange);

      let registrationRef: ServiceWorkerRegistration | null = null;

      navigator.serviceWorker.ready.then((registration) => {
        registrationRef = registration;
        
        if (registration.waiting) {
          setWaitingWorker(registration.waiting);
          setUpdateAvailable(true);
        }

        const handleUpdateFound = () => {
          const installingWorker = registration.installing;
          if (!installingWorker) return;

          const handleStateChange = () => {
            if (installingWorker.state === 'installed' && registration.waiting) {
              setWaitingWorker(registration.waiting);
              setUpdateAvailable(true);
            }
          };
          
          installingWorker.addEventListener('statechange', handleStateChange);
        };

        registration.addEventListener('updatefound', handleUpdateFound);
      });

      return () => {
        navigator.serviceWorker.removeEventListener('controllerchange', handleControllerChange);
      };
    }
  }, []);

  const updateServiceWorker = useCallback(async () => {
    if (waitingWorker) {
      waitingWorker.postMessage({ type: 'SKIP_WAITING' });
    } else {
      // Принудительная проверка обновлений
      const registration = await navigator.serviceWorker.ready;
      await registration.update();
    }
  }, [waitingWorker]);

  return {
    updateAvailable,
    updateServiceWorker,
    waitingWorker,
  };
}

/**
 * Hook для Push Notifications
 */
export function usePushNotifications() {
  const [isSupported, setIsSupported] = useState(false);
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [subscription, setSubscription] = useState<PushSubscription | null>(null);

  useEffect(() => {
    // Проверка поддержки
    setIsSupported(
      'serviceWorker' in navigator &&
      'PushManager' in window &&
      'Notification' in window
    );

    // Проверка текущего разрешения
    if ('Notification' in window) {
      setPermission(Notification.permission);
    }
  }, []);

  const requestPermission = useCallback(async () => {
    if (!('Notification' in window)) {
      throw new Error('Push notifications not supported');
    }

    const permission = await Notification.requestPermission();
    setPermission(permission);

    if (permission !== 'granted') {
      throw new Error('Notification permission denied');
    }

    return permission;
  }, []);

  const subscribeToPush = useCallback(async (vapidPublicKey: string) => {
    if (!('serviceWorker' in navigator)) {
      throw new Error('Service Worker not supported');
    }

    try {
      const registration = await navigator.serviceWorker.ready;

      const pushSubscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(vapidPublicKey),
      });

      setSubscription(pushSubscription);
      return pushSubscription;
    } catch (error) {
      console.error('Failed to subscribe to push notifications:', error);
      throw error;
    }
  }, []);

  const unsubscribeFromPush = useCallback(async () => {
    if (subscription) {
      await subscription.unsubscribe();
      setSubscription(null);
    }
  }, [subscription]);

  return {
    isSupported,
    permission,
    subscription,
    requestPermission,
    subscribeToPush,
    unsubscribeFromPush,
  };
}

/**
 * Комбинированный usePWA hook
 */
export function usePWA() {
  const installPWA = useInstallPWA();
  const isOnline = useOnlineStatus();
  const serviceWorker = useServiceWorker();
  const pushNotifications = usePushNotifications();

  return {
    // Install
    ...installPWA,
    
    // Online status
    isOnline,
    
    // Service Worker
    ...serviceWorker,
    
    // Push Notifications
    ...pushNotifications,
  };
}

// Utility функция для конвертации VAPID ключа
function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
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

// Export individual hooks
export {
  useInstallPWA,
  useOnlineStatus,
  useServiceWorker,
  usePushNotifications,
};
