"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Microscope, 
  Activity, 
  FileText, 
  Settings, 
  Cpu,
  Zap,
  Database
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Обзор", href: "/", icon: LayoutDashboard },
  { name: "Сканирования", href: "/scans", icon: Microscope },
  { name: "Симуляции", href: "/simulations", icon: Cpu },
  { name: "Анализ", href: "/analysis", icon: Activity },
  { name: "Сравнение", href: "/comparison", icon: Database },
  { name: "Отчёты", href: "/reports", icon: FileText },
  { name: "SSTV", href: "/sstv", icon: Zap },
  { name: "Настройки", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-border">
        <span className="text-2xl mr-3">🔬</span>
        <div>
          <h1 className="font-bold text-lg">Nanoprobe Lab</h1>
          <p className="text-xs text-muted-foreground">Sim Lab v2.0</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200",
                isActive
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
            >
              <item.icon className="h-5 w-5" />
              <span className="font-medium text-sm">{item.name}</span>
              {isActive && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary-foreground animate-pulse" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* System Status */}
      <div className="p-4 border-t border-border">
        <div className="bg-secondary rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-muted-foreground">Система в норме</span>
          </div>
          <div className="text-xs text-muted-foreground">
            API: <span className="text-foreground">localhost:8000</span>
          </div>
        </div>
      </div>
    </div>
  );
}
