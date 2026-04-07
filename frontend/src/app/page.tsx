"use client";

import { useEffect } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";
import { StatsGrid } from "@/components/stats-grid";
import { SystemHealth } from "@/components/system-health";
import { ActivityChart } from "@/components/activity-chart";
import { RecentActivity } from "@/components/recent-activity";
import { QuickActions } from "@/components/quick-actions";
import { useDashboardStore } from "@/stores/dashboard-store";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, CheckCircle, AlertTriangle } from "lucide-react";

export default function DashboardPage() {
  const {
    stats,
    systemHealth,
    alerts,
    isLoading,
    fetchDashboardData,
    subscribeToRealtime,
    unsubscribeFromRealtime,
  } = useDashboardStore();

  useEffect(() => {
    const init = async () => {
      await fetchDashboardData();
      subscribeToRealtime();
    };
    init();

    // Cleanup: unsubscribe from WebSocket on unmount
    return () => {
      unsubscribeFromRealtime();
    };
    // Stable references from Zustand store - safe to omit from deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <Alert 
                key={index} 
                variant={alert.level === 'critical' ? 'destructive' : alert.level === 'warning' ? 'warning' : 'info'}
                className="card-entrance"
              >
                {alert.level === 'critical' ? (
                  <AlertCircle className="h-4 w-4" />
                ) : alert.level === 'warning' ? (
                  <AlertTriangle className="h-4 w-4" />
                ) : (
                  <CheckCircle className="h-4 w-4" />
                )}
                <AlertDescription>{alert.message}</AlertDescription>
              </Alert>
            ))}
          </div>
        )}

        {/* Stats Grid */}
        {isLoading ? (
          <StatsGridSkeleton />
        ) : (
          <StatsGrid stats={stats} />
        )}

        {/* System Health & Quick Actions */}
        <div className="grid gap-6 md:grid-cols-2">
          <div className="card-entrance" style={{ animationDelay: '0.1s' }}>
            {isLoading ? (
              <Skeleton className="h-64 w-full rounded-xl" />
            ) : (
              <SystemHealth health={systemHealth} />
            )}
          </div>
          <div className="card-entrance" style={{ animationDelay: '0.15s' }}>
            <QuickActions />
          </div>
        </div>

        {/* Charts & Activity */}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="card-entrance" style={{ animationDelay: '0.2s' }}>
            {isLoading ? (
              <Skeleton className="h-80 w-full rounded-xl" />
            ) : (
              <ActivityChart />
            )}
          </div>
          <div className="card-entrance" style={{ animationDelay: '0.25s' }}>
            {isLoading ? (
              <Skeleton className="h-80 w-full rounded-xl" />
            ) : (
              <RecentActivity />
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

function StatsGridSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <Skeleton key={i} className="h-32 rounded-xl" />
      ))}
    </div>
  );
}
