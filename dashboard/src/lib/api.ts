const API_BASE = "http://localhost:8000";

export interface RunSummary {
  run_id: string;
  manifest: {
    seed: number;
    num_robots: number;
    order_rate: number | "scripted";
    num_ticks: number;
    warehouse: { width: number; height: number; zone_size: [number, number] };
  };
}

export interface Warehouse {
  width: number;
  height: number;
  zone_size: [number, number];
  racks: [number, number][];
  spawn_points: [number, number][];
  dropoff_points: [number, number][];
  pickup_points: [number, number][];
}

export type RobotState = "idle" | "moving" | "blocked";

export interface RobotRow {
  tick: number;
  robot_id: number;
  x: number;
  y: number;
  state: RobotState;
  order_id: number | null;
}

export interface OrderRow {
  id: number;
  origin_x: number;
  origin_y: number;
  destination_x: number;
  destination_y: number;
  arrival_tick: number;
  status: "pending" | "assigned" | "picked_up" | "completed";
  assigned_robot: number | null;
  assign_tick: number | null;
  pickup_tick: number | null;
  dropoff_tick: number | null;
  wait_ticks: number | null;
}

export interface TickRow {
  tick: number;
  min_pairwise_distance: number | null;
  near_miss_count: number;
  active_orders: number;
}

export interface ForecastRow {
  tick: number;
  congestion_prob: number | null;
  collision_prob: number | null;
  congestion_prob_raw: number | null;
  collision_prob_raw: number | null;
  classifier_triggered: boolean;
  backstop_triggered: boolean;
  triggered: boolean;
}

export interface Kpis {
  throughput_per_100_ticks: number;
  mean_wait_ticks: number | null;
  num_robots: number;
  active_orders_now: number;
  near_miss_count_now: number;
  total_orders: number;
  completed_orders: number;
  completion_rate: number | null;
  num_ticks: number;
}

export interface ForecasterThresholds {
  congestion: number;
  collision: number;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return res.json();
}

export const api = {
  listRuns: () => getJSON<RunSummary[]>("/api/runs"),
  warehouse: () => getJSON<Warehouse>("/api/warehouse"),
  robots: (runId: string) => getJSON<RobotRow[]>(`/api/runs/${runId}/robots`),
  orders: (runId: string) => getJSON<OrderRow[]>(`/api/runs/${runId}/orders`),
  ticks: (runId: string) => getJSON<TickRow[]>(`/api/runs/${runId}/ticks`),
  forecast: (runId: string) => getJSON<ForecastRow[]>(`/api/runs/${runId}/forecast`),
  kpis: (runId: string) => getJSON<Kpis>(`/api/runs/${runId}/kpis`),
  forecasterThresholds: () => getJSON<ForecasterThresholds>("/api/forecaster/thresholds"),
};
