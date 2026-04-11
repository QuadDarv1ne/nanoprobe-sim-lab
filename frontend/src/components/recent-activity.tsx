"use client";

import { useEffect, useState } from "react";
import { FileText, Cpu, Activity } from "lucide-react";
import { API_BASE } from "@/lib/config";
import { formatDistanceToNow } from "date-fns";
import { ru } from "date-fns/locale";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

interface ActivityItem {
  id: number;
  type: 'scan' | 'simulation' | 'analysis';
  title: string;
  timestamp: string;
}

interface ScanData {
  id: number;
  type?: string;
  created_at: string;
}

interface DashboardData {
  scans?: {
    recent?: ScanData[];
  };
}

export function RecentActivity() {
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchRecentActivity = async () => {
      try {
        const data = await apiClient.get<DashboardData>('/api/v1/dashboard/stats/detailed');

        // Формирование списка активности из последних сканирований
        const recentScans = (data.scans?.recent || []).map((scan) => ({
          id: scan.id,
          type: 'scan' as const,
          title: `Сканирование ${scan.type || 'СЗМ'}`,
          timestamp: scan.created_at,
        }));

        setActivity(recentScans.slice(0, 8));
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Не удалось получить последние действия';
        console.error('Failed to fetch recent activity:', error);
        toast.error('Ошибка загрузки активности', {
          description: errorMessage
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecentActivity();
  }, []);

  const getIconForType = (type: string) => {
    switch (type) {
      case 'scan':
        return <FileText className="h-4 w-4 text-blue-500" />;
      case 'simulation':
        return <Cpu className="h-4 w-4 text-purple-500" />;
      case 'analysis':
        return <Activity className="h-4 w-4 text-orange-500" />;
      default:
        return <FileText className="h-4 w-4 text-blue-500" />;
    }
  };

  if (isLoading) {
    return (
      <div className="glass rounded-xl p-6 border border-border h-80 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">Загрузка...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6 border border-border">
      <h3 className="text-lg font-semibold mb-4">Последняя активность</h3>

      <div className="space-y-3 max-h-64 overflow-y-auto">
        {activity.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-2 opacity-20" />
            <p className="text-sm">Нет недавней активности</p>
          </div>
        ) : (
          activity.map((item, index) => (
            <div
              key={item.id}
              className="flex items-center gap-3 p-3 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <div className="w-8 h-8 rounded-full bg-card flex items-center justify-center">
                {getIconForType(item.type)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{item.title}</p>
                <p className="text-xs text-muted-foreground">
                  {formatDistanceToNow(new Date(item.timestamp), {
                    addSuffix: true,
                    locale: ru
                  })}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
