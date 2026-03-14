"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Cpu, Play, Square, Trash2, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";

interface Simulation {
  id: number;
  simulation_type: string;
  duration_sec: number;
  status: string;
  created_at: string;
}

export default function SimulationsPage() {
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchSimulations = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/simulations`);
        const data = await res.json();
        setSimulations(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error('Failed to fetch simulations:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSimulations();
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Симуляции</h1>
            <p className="text-muted-foreground mt-1">
              Управление симуляциями сканирования
            </p>
          </div>
          <Button>
            <Play className="h-4 w-4 mr-2" />
            Запустить симуляцию
          </Button>
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-muted-foreground">Загрузка симуляций...</p>
          </div>
        ) : simulations.length === 0 ? (
          <div className="text-center py-12 glass rounded-xl border border-border">
            <Cpu className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Нет симуляций</h3>
            <p className="text-muted-foreground mb-4">
              Запустите симуляцию для начала работы
            </p>
            <Button>
              <Play className="h-4 w-4 mr-2" />
              Запустить симуляцию
            </Button>
          </div>
        ) : (
          <div className="glass rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-secondary/50 border-b border-border">
                <tr>
                  <th className="text-left p-4 font-medium text-muted-foreground">ID</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Тип</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Длительность</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Статус</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дата</th>
                  <th className="text-right p-4 font-medium text-muted-foreground">Действия</th>
                </tr>
              </thead>
              <tbody>
                {simulations.map((sim, index) => (
                  <tr
                    key={sim.id}
                    className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                  >
                    <td className="p-4 font-medium">#{sim.id}</td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded-full bg-purple-500/10 text-purple-500 text-sm">
                        {sim.simulation_type}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">{sim.duration_sec}s</td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-sm ${
                        sim.status === 'completed' 
                          ? 'bg-green-500/10 text-green-500' 
                          : sim.status === 'running'
                          ? 'bg-blue-500/10 text-blue-500'
                          : 'bg-gray-500/10 text-gray-500'
                      }`}>
                        {sim.status}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">
                      {new Date(sim.created_at).toLocaleDateString('ru-RU')}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="outline" size="icon">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon">
                          <Square className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon">
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
