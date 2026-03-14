"use client";

import { DashboardStats } from "@/stores/dashboard-store";
import { FileText, Cpu, Activity, GitCompare, FileCheck, Database } from "lucide-react";

interface StatsGridProps {
  stats: DashboardStats | null;
}

const statCards = [
  {
    title: "Сканирования",
    value: (stats: DashboardStats) => stats.total_scans,
    icon: FileText,
    color: "from-blue-500 to-cyan-500",
    bg: "bg-blue-500/10",
  },
  {
    title: "Симуляции",
    value: (stats: DashboardStats) => stats.total_simulations,
    icon: Cpu,
    color: "from-purple-500 to-pink-500",
    bg: "bg-purple-500/10",
  },
  {
    title: "Анализ",
    value: (stats: DashboardStats) => stats.total_analysis,
    icon: Activity,
    color: "from-orange-500 to-red-500",
    bg: "bg-orange-500/10",
  },
  {
    title: "Сравнения",
    value: (stats: DashboardStats) => stats.total_comparisons,
    icon: GitCompare,
    color: "from-green-500 to-emerald-500",
    bg: "bg-green-500/10",
  },
  {
    title: "Отчёты",
    value: (stats: DashboardStats) => stats.total_reports,
    icon: FileCheck,
    color: "from-indigo-500 to-blue-500",
    bg: "bg-indigo-500/10",
  },
  {
    title: "Всего записей",
    value: (stats: DashboardStats) => stats.total_items,
    icon: Database,
    color: "from-teal-500 to-cyan-500",
    bg: "bg-teal-500/10",
  },
];

export function StatsGrid({ stats }: StatsGridProps) {
  if (!stats) return null;

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
      {statCards.map((card, index) => (
        <div
          key={card.title}
          className="card-entrance glass rounded-xl p-4 border border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10 hover:-translate-y-1"
          style={{ animationDelay: `${index * 0.05}s` }}
        >
          <div className="flex items-center justify-between mb-3">
            <div className={`w-10 h-10 rounded-lg ${card.bg} flex items-center justify-center`}>
              <card.icon className={`h-5 w-5 bg-gradient-to-br ${card.color} bg-clip-text text-transparent`} style={{ WebkitTextFillColor: 'transparent' }} />
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-2xl font-bold">{card.value(stats)}</p>
            <p className="text-xs text-muted-foreground">{card.title}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
