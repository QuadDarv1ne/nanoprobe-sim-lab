/**
 * OfflineBanner Component
 * 
 * Баннер отображается в верхней части экрана когда нет подключения.
 * Автоматически скрывается при восстановлении соединения.
 */

'use client';

import { useState, useEffect } from 'react';
import { WifiOff, RefreshCw } from 'lucide-react';
import { useOnlineStatus } from '@/hooks/usePWA';

interface OfflineBannerProps {
  onRetry?: () => void;
}

export function OfflineBanner({ onRetry }: OfflineBannerProps) {
  const isOnline = useOnlineStatus();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!isOnline) {
      setIsVisible(true);
    } else {
      // Скрываем с задержкой при восстановлении
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [isOnline]);

  const handleRetry = () => {
    if (onRetry) {
      onRetry();
    } else {
      window.location.reload();
    }
  };

  // Не показываем если онлайн и не виден
  if (isOnline && !isVisible) {
    return null;
  }

  return (
    <div
      className={`fixed top-0 left-0 right-0 z-50 transition-transform duration-300 ${
        isVisible || !isOnline ? 'translate-y-0' : '-translate-y-full'
      }`}
    >
      <div className="bg-red-600 text-white px-4 py-2 shadow-lg">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <WifiOff className="w-5 h-5" />
            <span className="text-sm font-medium">
              {isOnline ? 'Соединение восстановлено' : 'Нет подключения к интернету'}
            </span>
          </div>

          {!isOnline && (
            <button
              onClick={handleRetry}
              className="flex items-center gap-1 px-3 py-1 bg-white/20 hover:bg-white/30 rounded-lg text-sm font-medium transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Повтор
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default OfflineBanner;
