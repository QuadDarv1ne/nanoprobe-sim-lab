"use client";

import { Play, Square, RotateCcw, FileText, Download, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/toaster";
import { API_BASE } from "@/lib/config";

const actions = [
  {
    id: 'run_spf_simulator',
    label: 'Запуск СЗМ',
    icon: Play,
    color: 'text-blue-500',
    bg: 'bg-blue-500/10',
  },
  {
    id: 'run_analyzer',
    label: 'Анализ',
    icon: Settings,
    color: 'text-orange-500',
    bg: 'bg-orange-500/10',
  },
  {
    id: 'run_sstv',
    label: 'SSTV',
    icon: Square,
    color: 'text-purple-500',
    bg: 'bg-purple-500/10',
  },
  {
    id: 'generate_report',
    label: 'Отчёт',
    icon: FileText,
    color: 'text-green-500',
    bg: 'bg-green-500/10',
  },
  {
    id: 'export_data',
    label: 'Экспорт',
    icon: Download,
    color: 'text-cyan-500',
    bg: 'bg-cyan-500/10',
  },
  {
    id: 'restart_all',
    label: 'Рестарт',
    icon: RotateCcw,
    color: 'text-red-500',
    bg: 'bg-red-500/10',
  },
];

export function QuickActions() {
  const handleAction = async (actionId: string) => {
    switch (actionId) {
      case 'run_spf_simulator':
        window.location.href = '/simulations';
        break;
      case 'run_analyzer':
        window.location.href = '/analysis';
        break;
      case 'run_sstv':
        window.location.href = '/sstv';
        break;
      case 'generate_report':
        window.location.href = '/reports';
        break;
      case 'export_data':
        await handleExportData();
        break;
      case 'restart_all':
        await handleRestartAll();
        break;
      default:
        toast.error('Неизвестное действие');
    }
  };

  const handleExportData = async () => {
    try {
      toast.info('Экспорт данных...');
      const res = await fetch(`${API_BASE}/api/v1/export-bulk`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nanoprobe_export_${Date.now()}.zip`;
        a.click();
        toast.success('Данные экспортированы');
      } else {
        toast.error('Ошибка экспорта');
      }
    } catch (error) {
      toast.error('Ошибка экспорта данных');
    }
  };

  const handleRestartAll = async () => {
    try {
      toast.info('Перезапуск системы...');
      const res = await fetch(`${API_BASE}/api/v1/system/restart`, {
        method: 'POST',
      });
      if (res.ok) {
        toast.success('Система перезапускается');
      } else {
        toast.error('Ошибка перезапуска');
      }
    } catch (error) {
      toast.error('Ошибка перезапуска системы');
    }
  };

  return (
    <div className="glass rounded-xl p-6 border border-border">
      <h3 className="text-lg font-semibold mb-4">Быстрые действия</h3>

      <div className="grid grid-cols-3 gap-3">
        {actions.map((action) => (
          <button
            key={action.id}
            onClick={() => handleAction(action.id)}
            className="flex flex-col items-center gap-2 p-4 rounded-lg bg-secondary/50 hover:bg-secondary transition-all duration-200 hover:scale-105 hover:shadow-lg"
          >
            <div className={`w-10 h-10 rounded-lg ${action.bg} flex items-center justify-center`}>
              <action.icon className={`h-5 w-5 ${action.color}`} />
            </div>
            <span className="text-xs font-medium text-center">{action.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
