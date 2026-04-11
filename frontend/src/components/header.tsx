"use client";

import { Bell, RefreshCw, Moon, Sun, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/providers/theme-provider";
import { useState, useEffect } from "react";
import { useDashboardStore } from "@/stores/dashboard-store";
import { cn } from "@/lib/utils";

export function Header() {
  const { theme, toggleTheme } = useTheme();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [notificationCount, setNotificationCount] = useState(0);
  const { fetchDashboardData, alerts } = useDashboardStore();

  useEffect(() => {
    setNotificationCount(alerts.length);
  }, [alerts]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchDashboardData();
    setIsRefreshing(false);
  };

  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold">Панель управления</h2>
      </div>

      <div className="flex items-center gap-3">
        {/* Refresh Button */}
        <Button
          variant="outline"
          size="icon"
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="relative"
          aria-label="Обновить данные"
        >
          <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} aria-hidden="true" />
        </Button>

        {/* Notifications */}
        <Button
          variant="outline"
          size="icon"
          className="relative"
          aria-label={`Уведомления, ${notificationCount} новых`}
        >
          <Bell className="h-4 w-4" aria-hidden="true" />
          {notificationCount > 0 && (
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-xs flex items-center justify-center" aria-hidden="true">
              {notificationCount}
            </span>
          )}
        </Button>

        {/* Theme Toggle */}
        <Button
          variant="outline"
          size="icon"
          onClick={toggleTheme}
          aria-label={theme === "dark" ? "Переключить на светлую тему" : "Переключить на тёмную тему"}
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" aria-hidden="true" />
          ) : (
            <Moon className="h-4 w-4" aria-hidden="true" />
          )}
        </Button>

        {/* User Menu */}
        <Button variant="ghost" className="gap-2">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center" aria-hidden="true">
            <User className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="text-sm font-medium hidden md:inline-block">Admin</span>
        </Button>
      </div>
    </header>
  );
}
