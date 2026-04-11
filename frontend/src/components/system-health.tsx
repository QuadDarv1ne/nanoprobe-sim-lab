"use client";

import type { SystemHealth } from "@/stores/dashboard-store";
import { Progress } from "@/components/ui/progress";
import { Cpu, MemoryStick, HardDrive, CheckCircle, AlertTriangle, AlertCircle } from "lucide-react";

interface SystemHealthProps {
  health: SystemHealth | null;
}

export function SystemHealth({ health }: SystemHealthProps) {
  if (!health) return null;

  const getStatusIcon = () => {
    switch (health.status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'critical':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
    }
  };

  const getStatusColor = (value: number) => {
    if (value < 50) return 'bg-green-500';
    if (value < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="glass rounded-xl p-6 border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Cpu className="h-5 w-5" />
          Состояние системы
        </h3>
        {getStatusIcon()}
      </div>

      <div className="space-y-5">
        {/* CPU */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span>CPU</span>
            </div>
            <span className="font-medium">{health.cpu_percent.toFixed(1)}%</span>
          </div>
          <Progress
            value={health.cpu_percent}
            indicatorClassName={getStatusColor(health.cpu_percent)}
          />
        </div>

        {/* Memory */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <MemoryStick className="h-4 w-4 text-muted-foreground" />
              <span>RAM</span>
            </div>
            <span className="font-medium">{health.memory_percent.toFixed(1)}%</span>
          </div>
          <Progress
            value={health.memory_percent}
            indicatorClassName={getStatusColor(health.memory_percent)}
          />
        </div>

        {/* Disk */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <HardDrive className="h-4 w-4 text-muted-foreground" />
              <span>Диск</span>
            </div>
            <span className="font-medium">{health.disk_percent.toFixed(1)}%</span>
          </div>
          <Progress
            value={health.disk_percent}
            indicatorClassName={getStatusColor(health.disk_percent)}
          />
        </div>
      </div>

      {/* Status Badge */}
      <div className="mt-6 flex items-center justify-center">
        <div className={`px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2 ${
          health.status === 'healthy'
            ? 'bg-green-500/10 text-green-500 border border-green-500/20'
            : health.status === 'warning'
            ? 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20'
            : 'bg-red-500/10 text-red-500 border border-red-500/20'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            health.status === 'healthy' ? 'bg-green-500' : health.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
          } animate-pulse`} />
          {health.status === 'healthy' ? 'Все системы в норме' : health.status === 'warning' ? 'Требуется внимание' : 'Критическое состояние'}
        </div>
      </div>
    </div>
  );
}
