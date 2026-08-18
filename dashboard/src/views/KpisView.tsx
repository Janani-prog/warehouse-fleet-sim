import { useEffect, useState } from "react";
import { api, type Kpis } from "../lib/api";
import { KpiCard } from "../components/KpiCard";

export function KpisView({ runId }: { runId: string }) {
  const [kpis, setKpis] = useState<Kpis | null>(null);

  useEffect(() => {
    setKpis(null);
    api.kpis(runId).then(setKpis);
  }, [runId]);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Fleet KPIs</h2>
        <p className="text-[13px] text-secondary mt-1">Run summary: {runId}</p>
      </div>

      {!kpis ? (
        <p className="text-secondary text-[13px]">Loading…</p>
      ) : (
        <div className="grid grid-cols-3 gap-gutter">
          <KpiCard label="Throughput" value={`${kpis.throughput_per_100_ticks}`} unit="orders / 100 ticks" />
          <KpiCard label="Avg Wait Time" value={kpis.mean_wait_ticks != null ? `${kpis.mean_wait_ticks}` : "—"} unit="ticks" />
          <KpiCard label="Fleet Size" value={`${kpis.num_robots}`} unit="robots" />
          <KpiCard label="Total Orders" value={`${kpis.total_orders}`} />
          <KpiCard label="Completed Orders" value={`${kpis.completed_orders}`} />
          <KpiCard
            label="Completion Rate"
            value={kpis.completion_rate != null ? `${Math.round(kpis.completion_rate * 100)}%` : "—"}
          />
          <KpiCard label="Active Orders (final tick)" value={`${kpis.active_orders_now}`} />
          <KpiCard label="Near-Misses (final tick)" value={`${kpis.near_miss_count_now}`} />
          <KpiCard label="Run Length" value={`${kpis.num_ticks}`} unit="ticks" />
        </div>
      )}
    </div>
  );
}
