import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Kpis, type OrderRow, type RobotRow, type Warehouse } from "../lib/api";
import { KpiCard } from "../components/KpiCard";
import { PauseIcon, PlayIcon } from "../components/icons";

const STATE_COLOR: Record<string, string> = {
  idle: "var(--color-secondary)",
  moving: "var(--color-primary)",
  blocked: "var(--color-error)",
};

const TICK_MS = 120;

export function FleetMapView({ runId }: { runId: string }) {
  const [warehouse, setWarehouse] = useState<Warehouse | null>(null);
  const [robots, setRobots] = useState<RobotRow[]>([]);
  const [orders, setOrders] = useState<OrderRow[]>([]);
  const [kpis, setKpis] = useState<Kpis | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(true);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    api.warehouse().then(setWarehouse);
  }, []);

  useEffect(() => {
    setTick(0);
    setPlaying(true);
    Promise.all([api.robots(runId), api.orders(runId), api.kpis(runId)]).then(
      ([robotsData, ordersData, kpisData]) => {
        setRobots(robotsData);
        setOrders(ordersData);
        setKpis(kpisData);
      }
    );
  }, [runId]);

  const maxTick = useMemo(() => robots.reduce((m, r) => Math.max(m, r.tick), 0), [robots]);

  const robotsByTick = useMemo(() => {
    const map = new Map<number, RobotRow[]>();
    for (const r of robots) {
      const list = map.get(r.tick);
      if (list) list.push(r);
      else map.set(r.tick, [r]);
    }
    return map;
  }, [robots]);

  useEffect(() => {
    if (!playing || maxTick === 0) return;
    intervalRef.current = window.setInterval(() => {
      setTick((t) => (t >= maxTick ? 0 : t + 1));
    }, TICK_MS);
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
    };
  }, [playing, maxTick]);

  const currentRobots = robotsByTick.get(tick) ?? [];
  const activeOriginPins = useMemo(
    () =>
      orders.filter(
        (o) => o.arrival_tick <= tick && (o.status === "pending" || o.status === "assigned")
      ),
    [orders, tick]
  );

  if (!warehouse) {
    return <p className="text-secondary text-[13px]">Loading warehouse layout…</p>;
  }

  const { width, height } = warehouse;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Fleet Map</h2>
        <p className="text-[13px] text-secondary mt-1">Replaying run: {runId}</p>
      </div>

      <div className="grid grid-cols-4 gap-gutter">
        <KpiCard label="Throughput" value={kpis ? `${kpis.throughput_per_100_ticks}` : "—"} unit="orders/100 ticks" />
        <KpiCard label="Avg Wait Time" value={kpis?.mean_wait_ticks != null ? `${kpis.mean_wait_ticks}` : "—"} unit="ticks" />
        <KpiCard label="Active Robots" value={`${currentRobots.filter((r) => r.state !== "idle").length}`} unit={`/ ${kpis?.num_robots ?? "—"}`} />
        <KpiCard
          label="Near-Misses"
          value={`${currentRobots.filter((r) => r.state === "blocked").length}`}
          unit="blocked now"
          accent
        />
      </div>

      <div className="relative w-full bg-surface-container-lowest border border-outline-variant overflow-hidden rounded" style={{ aspectRatio: `${width} / ${height}` }}>
        <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
          <defs>
            <pattern id="grid" width="1" height="1" patternUnits="userSpaceOnUse">
              <path d="M 1 0 L 0 0 0 1" fill="none" stroke="var(--color-surface-variant)" strokeWidth="0.03" />
            </pattern>
          </defs>
          <rect x="0" y="0" width={width} height={height} fill="url(#grid)" />

          {warehouse.racks.map(([x, y]) => (
            <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill="var(--color-surface-variant)" stroke="var(--color-outline-variant)" strokeWidth="0.02" />
          ))}

          {activeOriginPins.map((o) => (
            <circle key={o.id} cx={o.origin_x + 0.5} cy={o.origin_y + 0.5} r="0.12" fill="var(--color-outline-variant)" />
          ))}

          {currentRobots.map((r) => (
            <circle
              key={r.robot_id}
              cx={r.x + 0.5}
              cy={r.y + 0.5}
              r="0.32"
              fill={STATE_COLOR[r.state]}
              stroke="var(--color-surface-container-lowest)"
              strokeWidth="0.06"
            />
          ))}
        </svg>

        <div className="absolute bottom-3 left-3 bg-surface/90 border border-outline-variant px-3 py-2 rounded flex gap-4 text-[11px] font-mono text-secondary">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: STATE_COLOR.idle }} /> Idle
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: STATE_COLOR.moving }} /> Moving
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: STATE_COLOR.blocked }} /> Blocked
          </span>
        </div>
      </div>

      <div className="flex items-center gap-4 bg-surface-container-lowest border border-outline-variant rounded px-4 py-3">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="w-8 h-8 flex items-center justify-center rounded bg-primary text-on-primary"
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>
        <input
          type="range"
          min={0}
          max={maxTick}
          value={tick}
          onChange={(e) => {
            setPlaying(false);
            setTick(Number(e.target.value));
          }}
          className="flex-1 accent-primary"
        />
        <span className="font-mono text-[12px] text-secondary w-24 text-right">
          tick {tick} / {maxTick}
        </span>
      </div>
    </div>
  );
}
