/**
 * PWA Provider компонент
 * 
 * Обеспечивает:
 * - Регистрацию Service Worker
 * - UI для установки приложения
 * - Уведомления об обновлениях
 * - Offline индикатор
 */

'use client';

import { useEffect, useState } from 'react';
import { usePWA, registerServiceWorker, requestNotificationPermission } from '@/lib/pwa';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { WifiOff, Download, RefreshCw, Bell } from 'lucide-react';

export function PWAProvider({ children }: { children: React.ReactNode }) {
  const { isOnline, isInstalled, canInstall, install, updateAvailable, applyUpdate } = usePWA();
  const [showInstallPrompt, setShowInstallPrompt] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);

  useEffect(() => {
    // Регистрация Service Worker
    registerServiceWorker();

    // Проверка уведомлений
    if ('Notification' in window) {
      setNotificationsEnabled(Notification.permission === 'granted');
    }

    // Показ install prompt через 30 секунд
    const timer = setTimeout(() => {
      if (!isInstalled && canInstall) {
        setShowInstallPrompt(true);
      }
    }, 30000);

    return () => clearTimeout(timer);
  }, [isInstalled, canInstall]);

  const handleInstall = async () => {
    const result = await install();
    setShowInstallPrompt(false);
    
    if (result.outcome === 'accepted') {
      console.log('PWA installed successfully');
    }
  };

  const handleEnableNotifications = async () => {
    const permission = await requestNotificationPermission();
    setNotificationsEnabled(permission === 'granted');
  };

  return (
    <>
      {children}
      
      {/* Offline Banner */}
      {!isOnline && (
        <Alert variant="destructive" className="fixed bottom-4 left-4 right-4 z-50">
          <WifiOff className="h-4 w-4" />
          <AlertDescription>
            Вы офлайн. Некоторые функции могут быть недоступны.
          </AlertDescription>
        </Alert>
      )}

      {/* Update Available Banner */}
      {updateAvailable && (
        <Alert className="fixed top-4 left-4 right-4 z-50 bg-blue-600 text-white">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <AlertDescription className="flex items-center justify-between">
            <span>Доступно обновление приложения</span>
            <Button 
              size="sm" 
              variant="secondary"
              onClick={applyUpdate}
            >
              Обновить
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Install Prompt */}
      {showInstallPrompt && !isInstalled && canInstall && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-background rounded-lg p-6 max-w-md w-full shadow-xl">
            <h3 className="text-lg font-semibold mb-2">
              Установить приложение
            </h3>
            <p className="text-muted-foreground mb-4">
              Установите Nanoprobe Sim Lab для быстрого доступа и офлайн режима
            </p>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                onClick={() => setShowInstallPrompt(false)}
              >
                Позже
              </Button>
              <Button onClick={handleInstall}>
                <Download className="h-4 w-4 mr-2" />
                Установить
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Notification Prompt */}
      {!notificationsEnabled && !isInstalled && (
        <div className="fixed bottom-20 right-4 z-40">
          <Button
            variant="outline"
            size="sm"
            onClick={handleEnableNotifications}
          >
            <Bell className="h-4 w-4 mr-2" />
            Включить уведомления
          </Button>
        </div>
      )}
    </>
  );
}
