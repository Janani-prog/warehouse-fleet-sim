import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type ForecastRow, type ForecasterThresholds } from "../lib/api";
import { KpiCard } from "../components/KpiCard";

export function AnomalyTimelineView({ runId }: { runId: string }) {
  const [forecast, setForecast] = useState<ForecastRow[]>([]);
  const [thresholds, setThresholds] = useState<ForecasterThresholds | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setLoaded(false);
    Promise.all([api.forecast(runId), api.forecasterThresholds().catch(() => null)]).then(
      ([forecastData, thresholdData]) => {
        setForecast(forecastData);
        setThresholds(thresholdData);
        setLoaded(true);
      }
    );
  }, [runId]);

  const triggerPoints = useMemo(
    () =>
      forecast
        .filter((f) => f.triggered)
        .map((f) => ({ tick: f.tick, value: Math.max(f.congestion_prob ?? 0, f.collision_prob ?? 0) })),
    [forecast]
  );

  const stats = useMemo(() => {
    const ready = forecast.filter((f) => f.congestion_prob != null);
    if (ready.length === 0) return null;
    const avgConfidence =
      ready.reduce((s, f) => s + Math.max(f.congestion_prob ?? 0, f.collision_prob ?? 0), 0) / ready.length;
    const triggerCount = forecast.filter((f) => f.triggered).length;
    return { avgConfidence, triggerCount, readyTicks: ready.length };
  }, [forecast]);

  if (!loaded) {
    return <p className="text-secondary text-[13px]">Loading forecast data…</p>;
  }

  if (forecast.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Anomaly Timeline</h2>
        <p className="text-[13px] text-secondary">
          This run has no forecast data — it was generated before the forecaster was wired in, or without{" "}
          <code className="font-mono text-[12px]">ml.run_forecast_demo</code>. Select the
          congestion_spike_demo run to see live confidence + triggers.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Anomaly Timeline</h2>
        <p className="text-[13px] text-secondary mt-1">Calibrated forecaster confidence vs. time — run: {runId}</p>
      </div>

      {stats && (
        <div className="grid grid-cols-3 gap-gutter">
          <KpiCard label="Avg Confidence" value={stats.avgConfidence.toFixed(2)} />
          <KpiCard label="Triggers" value={`${stats.triggerCount}`} unit={stats.triggerCount > 0 ? "fired" : undefined} accent={stats.triggerCount > 0} />
          <KpiCard label="Classifier-Ready Ticks" value={`${stats.readyTicks}`} unit={`/ ${forecast.length}`} />
        </div>
      )}

      <div className="bg-surface-container-lowest border border-outline-variant rounded p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-[13px] font-semibold text-on-surface">Confidence vs. Tick</h3>
          <div className="flex gap-4 text-[11px] font-mono text-secondary">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-[2px]" style={{ background: "var(--color-primary)" }} /> Congestion
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-[2px]" style={{ background: "var(--color-secondary)" }} /> Collision-risk
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "var(--color-error)" }} /> Triggered
            </span>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={forecast} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="var(--color-outline-variant)" strokeOpacity={0.5} vertical={false} />
            <XAxis dataKey="tick" tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "var(--color-secondary)" }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "var(--color-secondary)" }} />
            <Tooltip
              contentStyle={{ fontSize: 12, fontFamily: "Inter", border: "1px solid var(--color-outline-variant)" }}
              labelFormatter={(t) => `tick ${t}`}
            />
            <Legend wrapperStyle={{ display: "none" }} />
            {thresholds && (
              <ReferenceLine
                y={thresholds.congestion}
                stroke="var(--color-outline)"
                strokeDasharray="6 4"
                label={{ value: `threshold ${thresholds.congestion.toFixed(2)}`, fontSize: 10, fill: "var(--color-secondary)", position: "insideTopRight" }}
              />
            )}
            <Line type="monotone" dataKey="congestion_prob" stroke="var(--color-primary)" dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls />
            <Line type="monotone" dataKey="collision_prob" stroke="var(--color-secondary)" dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
        {triggerPoints.length > 0 && (
          <ResponsiveContainer width="100%" height={40}>
            <ScatterChart margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
              <XAxis dataKey="tick" domain={[0, forecast.length - 1]} hide />
              <YAxis dataKey="value" domain={[0, 1]} hide />
              <Scatter data={triggerPoints} fill="var(--color-error)" />
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
