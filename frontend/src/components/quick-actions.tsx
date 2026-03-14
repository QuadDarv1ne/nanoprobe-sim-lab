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
    toast.info(`Выполнение: ${actionId}...`);
    
    try {
      const res = await fetch(`${API_BASE}/api/actions/quick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: actionId }),
      });
      
      const data = await res.json();
      
      if (data.success) {
        toast.success(`Действие выполнено: ${actionId}`);
      } else {
        toast.error(`Ошибка: ${data.error || 'Неизвестная ошибка'}`);
      }
    } catch (error) {
      toast.error('Ошибка соединения с сервером');
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
