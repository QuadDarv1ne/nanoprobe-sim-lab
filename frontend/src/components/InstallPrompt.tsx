/**
 * InstallPrompt Component
 * 
 * Компонент для отображения prompt установки PWA приложения.
 * Автоматически скрывается после установки или отклонения.
 */

'use client';

import { useState, useEffect } from 'react';
import { X, Download, Smartphone, Star } from 'lucide-react';
import { useInstallPWA } from '@/hooks/usePWA';

interface InstallPromptProps {
  onDismiss?: () => void;
  onInstall?: () => void;
}

export function InstallPrompt({ onDismiss, onInstall }: InstallPromptProps) {
  const { canInstall, installApp, isInstalled, platform } = useInstallPWA();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Показываем prompt с задержкой 5 секунд
    if (canInstall && !isInstalled) {
      const timer = setTimeout(() => {
        setIsVisible(true);
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [canInstall, isInstalled]);

  const handleInstall = async () => {
    try {
      await installApp();
      onInstall?.();
    } catch (error) {
      console.error('Install failed:', error);
    } finally {
      setIsVisible(false);
    }
  };

  const handleDismiss = () => {
    setIsVisible(false);
    onDismiss?.();
    
    // Сохраняем что пользователь отклонил установку
    localStorage.setItem('pwaInstallDismissed', 'true');
  };

  // Не показываем если уже установлен или нельзя установить
  if (isInstalled || !canInstall || !isVisible) {
    return null;
  }

  // Не показываем если пользователь уже отклонял
  if (localStorage.getItem('pwaInstallDismissed') === 'true') {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 z-50 animate-slide-up">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header с кнопкой закрытия */}
        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-700">
          <div className="flex items-center gap-2">
            <Smartphone className="w-5 h-5 text-white" />
            <span className="text-white font-semibold">Nanoprobe Sim Lab</span>
          </div>
          <button
            onClick={handleDismiss}
            className="text-white/80 hover:text-white transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="flex items-start gap-3 mb-4">
            <div className="flex-shrink-0 w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center">
              <Download className="w-6 h-6 text-blue-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-white font-semibold mb-1">
                Установите приложение
              </h3>
              <p className="text-slate-400 text-sm">
                Быстрый доступ к SSTV Ground Station и СЗМ симулятору прямо с рабочего стола
              </p>
            </div>
          </div>

          {/* Features */}
          <div className="space-y-2 mb-4">
            <FeatureItem text="🚀 Быстрый запуск с рабочего стола" />
            <FeatureItem text="📡 Offline режим для основных функций" />
            <FeatureItem text="🔔 Push уведомления для событий" />
            <FeatureItem text="💾 Автосохранение данных" />
          </div>

          {/* Rating */}
          <div className="flex items-center gap-2 mb-4 pb-4 border-b border-slate-700">
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
            <span className="text-slate-400 text-sm ml-2">
              Любимое приложение пользователей
            </span>
          </div>

          {/* Buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleInstall}
              className="flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Установить
            </button>
            <button
              onClick={handleDismiss}
              className="px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-xl font-medium transition-colors"
            >
              Позже
            </button>
          </div>

          {/* Platform hint */}
          {platform && (
            <p className="text-slate-500 text-xs mt-3 text-center">
              Оптимизировано для {platform}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// Helper компонент для feature items
function FeatureItem({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-1.5 h-1.5 bg-blue-400 rounded-full flex-shrink-0" />
      <span className="text-slate-300 text-sm">{text}</span>
    </div>
  );
}

export default InstallPrompt;
