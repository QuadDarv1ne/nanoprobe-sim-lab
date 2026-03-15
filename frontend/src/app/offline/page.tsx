/**
 * Offline Page
 * 
 * Страница отображается когда нет подключения к интернету.
 * Показывает статус подключения и кэшированную информацию.
 */

'use client';

import { useState, useEffect } from 'react';
import { WifiOff, Wifi, RefreshCw, Home, Database } from 'lucide-react';

export default function OfflinePage() {
  const [isOnline, setIsOnline] = useState(false);
  const [lastSync, setLastSync] = useState<string | null>(null);

  useEffect(() => {
    // Проверка текущего статуса
    setIsOnline(navigator.onLine);

    // Обработчики online/offline
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Получение последней синхронизации из localStorage
    const savedSync = localStorage.getItem('lastSync');
    if (savedSync) {
      setLastSync(savedSync);
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRetry = () => {
    window.location.reload();
  };

  const handleGoHome = () => {
    window.location.href = '/';
  };

  const formatLastSync = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString('ru-RU', {
        day: 'numeric',
        month: 'long',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-slate-700">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-slate-700/50 mb-4">
            {isOnline ? (
              <Wifi className="w-10 h-10 text-green-400" />
            ) : (
              <WifiOff className="w-10 h-10 text-red-400" />
            )}
          </div>
          
          <h1 className="text-3xl font-bold text-white mb-2">
            {isOnline ? 'Соединение восстановлено!' : 'Нет подключения'}
          </h1>
          
          <p className="text-slate-400">
            {isOnline
              ? 'Подключение к интернету успешно'
              : 'Проверьте ваше соединение и попробуйте снова'}
          </p>
        </div>

        {/* Status Card */}
        <div className="bg-slate-700/30 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-300">Статус подключения</span>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                isOnline
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-red-500/20 text-red-400'
              }`}
            >
              {isOnline ? 'Online' : 'Offline'}
            </span>
          </div>

          {lastSync && (
            <div className="flex items-center gap-2 text-slate-400 text-sm">
              <Database className="w-4 h-4" />
              <span>Последняя синхронизация:</span>
              <span className="text-slate-200">{formatLastSync(lastSync)}</span>
            </div>
          )}

          {!lastSync && (
            <div className="flex items-center gap-2 text-slate-400 text-sm">
              <Database className="w-4 h-4" />
              <span>Данные не синхронизированы</span>
            </div>
          )}
        </div>

        {/* Cached Data Info */}
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 mb-6">
          <h3 className="text-blue-400 font-medium mb-2">
            📦 Кэшированные данные
          </h3>
          <p className="text-slate-400 text-sm">
            Некоторые данные могут быть доступны офлайн благодаря Service Worker.
            При восстановлении подключения данные будут автоматически синхронизированы.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="space-y-3">
          {!isOnline && (
            <button
              onClick={handleRetry}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
            >
              <RefreshCw className="w-5 h-5" />
              Попробовать снова
            </button>
          )}

          {isOnline && (
            <button
              onClick={handleGoHome}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
            >
              <Home className="w-5 h-5" />
              На главную
            </button>
          )}

          <button
            onClick={() => window.history.back()}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-xl font-medium transition-colors"
          >
            Назад
          </button>
        </div>

        {/* Tips */}
        <div className="mt-6 pt-6 border-t border-slate-700">
          <h4 className="text-slate-400 text-sm mb-3">💡 Советы</h4>
          <ul className="space-y-2 text-slate-500 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-blue-400 mt-1">•</span>
              <span>
                Установите приложение для лучшего offline опыта
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-400 mt-1">•</span>
              <span>
                Данные кэшируются автоматически при посещении
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-400 mt-1">•</span>
              <span>
                Синхронизация произойдёт при восстановлении соединения
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
