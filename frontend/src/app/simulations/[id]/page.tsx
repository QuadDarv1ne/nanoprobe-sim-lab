"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { ArrowLeft, Cpu, Play, Square, Calendar, Clock, FileText, Code } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { toast } from "@/components/ui/toaster";
import { useParams, useRouter } from "next/navigation";

interface Simulation {
  id: number;
  simulation_id: string;
  simulation_type: string;
  status: string;
  start_time?: string;
  end_time?: string;
  duration_seconds?: number;
  parameters?: Record<string, unknown>;
  results_summary?: Record<string, unknown>;
  created_at: string;
}

export default function SimulationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const simId = Number(params.id);

  const [simulation, setSimulation] = useState<Simulation | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isStopping, setIsStopping] = useState(false);

  useEffect(() => {
    fetchSimulation();
  }, [simId]);

  const fetchSimulation = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/simulations/${simId}`);
      if (res.ok) {
        const data = await res.json();
        setSimulation(data);
      } else {
        toast.error('Симуляция не найдена', {
          description: `HTTP ${res.status}`
        });
        setSimulation(null);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Не удалось загрузить симуляцию';
      console.error('Failed to fetch simulation:', error);
      toast.error('Ошибка загрузки', { description: errorMessage });
      setSimulation(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    if (!simulation || simulation.status !== 'running') return;
    setIsStopping(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/simulations/${simId}/stop`, {
        method: 'POST',
      });
      if (res.ok) {
        toast.success('Симуляция остановлена');
        fetchSimulation();
      } else {
        toast.error('Ошибка остановки', { description: `HTTP ${res.status}` });
      }
    } catch (error) {
      toast.error('Ошибка', { description: 'Не удалось остановить симуляцию' });
    } finally {
      setIsStopping(false);
    }
  };

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-muted-foreground">Загрузка симуляции...</p>
        </div>
      </DashboardLayout>
    );
  }

  if (!simulation) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold mb-2">Симуляция не найдена</h2>
          <p className="text-muted-foreground mb-4">Симуляция с ID #{simId} не существует</p>
          <Button onClick={() => router.push('/simulations')}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Вернуться к списку
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  const statusColors: Record<string, string> = {
    completed: 'bg-green-500/10 text-green-500',
    running: 'bg-blue-500/10 text-blue-500',
    failed: 'bg-red-500/10 text-red-500',
    stopped: 'bg-gray-500/10 text-gray-500',
    pending: 'bg-yellow-500/10 text-yellow-500',
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="outline" size="icon" onClick={() => router.push('/simulations')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">Симуляция #{simulation.id}</h1>
              <p className="text-muted-foreground mt-1">{simulation.simulation_id}</p>
            </div>
          </div>
          <div className="flex gap-2">
            {simulation.status === 'running' && (
              <Button variant="destructive" onClick={handleStop} disabled={isStopping}>
                <Square className="h-4 w-4 mr-2" />
                Остановить
              </Button>
            )}
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4">Основная информация</h2>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Cpu className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Тип</p>
                  <p className="font-medium capitalize">{simulation.simulation_type}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className={`px-2 py-1 rounded-full text-sm ${statusColors[simulation.status] || 'bg-gray-500/10 text-gray-500'}`}>
                  {simulation.status}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Calendar className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Создана</p>
                  <p className="font-medium">{format(new Date(simulation.created_at), 'dd.MM.yyyy HH:mm:ss')}</p>
                </div>
              </div>
              {simulation.duration_seconds != null && (
                <div className="flex items-center gap-3">
                  <Clock className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Длительность</p>
                    <p className="font-medium">{simulation.duration_seconds.toFixed(1)}s</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4">Временная шкала</h2>
            <div className="space-y-4">
              {simulation.start_time && (
                <div>
                  <p className="text-sm text-muted-foreground">Начало</p>
                  <p className="font-medium">{format(new Date(simulation.start_time), 'dd.MM.yyyy HH:mm:ss')}</p>
                </div>
              )}
              {simulation.end_time && (
                <div>
                  <p className="text-sm text-muted-foreground">Завершение</p>
                  <p className="font-medium">{format(new Date(simulation.end_time), 'dd.MM.yyyy HH:mm:ss')}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Parameters */}
        {simulation.parameters && Object.keys(simulation.parameters).length > 0 && (
          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Code className="h-5 w-5" />
              Параметры
            </h2>
            <pre className="text-xs bg-secondary/50 rounded p-4 overflow-auto max-h-60">
              {JSON.stringify(simulation.parameters, null, 2)}
            </pre>
          </div>
        )}

        {/* Results */}
        {simulation.results_summary && Object.keys(simulation.results_summary).length > 0 && (
          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Результаты
            </h2>
            <pre className="text-xs bg-secondary/50 rounded p-4 overflow-auto max-h-60">
              {JSON.stringify(simulation.results_summary, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
