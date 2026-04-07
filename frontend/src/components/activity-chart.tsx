"use client";

import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import type { ChartOptions } from "chart.js";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Activity } from "lucide-react";
import { API_BASE } from "@/lib/config";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface TimelineDataItem {
  date: string;
  scans?: number;
  simulations?: number;
  analysis?: number;
}

interface TimelineResponse {
  timeline?: TimelineDataItem[];
}

export function ActivityChart() {
  const [timelineData, setTimelineData] = useState<TimelineDataItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchActivityData = async () => {
      try {
        const data = await apiClient.get<TimelineResponse>(
          '/api/v1/dashboard/activity/timeline',
          { params: { days: 7 } }
        );
        setTimelineData(data.timeline || []);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Не удалось получить данные графика';
        console.error('Failed to fetch activity data:', error);
        toast.error('Ошибка загрузки данных активности', {
          description: errorMessage
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchActivityData();
  }, []);

  const chartData = {
    labels: timelineData.map(item => {
      const date = new Date(item.date);
      return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' });
    }),
    datasets: [
      {
        label: 'Сканирования',
        data: timelineData.map(item => item.scans),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Симуляции',
        data: timelineData.map(item => item.simulations),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Анализ',
        data: timelineData.map(item => item.analysis),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'rgb(148, 163, 184)',
          usePointStyle: true,
        },
      },
      title: {
        display: true,
        text: 'Активность за 7 дней',
        color: 'rgb(241, 245, 249)',
        font: { size: 14, weight: 'bold' },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(51, 65, 85, 0.5)',
        },
        ticks: {
          color: 'rgb(148, 163, 184)',
        },
      },
      y: {
        grid: {
          color: 'rgba(51, 65, 85, 0.5)',
        },
        ticks: {
          color: 'rgb(148, 163, 184)',
          precision: 0,
        },
      },
    },
  };

  if (isLoading) {
    return (
      <div className="glass rounded-xl p-6 border border-border h-80 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">Загрузка данных...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6 border border-border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Активность
        </h3>
      </div>
      <div className="h-64">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
