"""Per-tick telemetry: queue depth/density per zone, min pairwise distance,
near-miss events, per-order wait time, robot positions for replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class TelemetryLogger:
    def __init__(self):
        self.tick_rows: list[dict] = []
        self.zone_rows: list[dict] = []
        self.robot_rows: list[dict] = []
        self.event_rows: list[dict] = []
        self.forecast_rows: list[dict] = []

    def log_tick(
        self,
        tick: int,
        min_pairwise_distance: float,
        near_miss_count: int,
        active_orders: int,
    ) -> None:
        self.tick_rows.append(
            {
                "tick": tick,
                "min_pairwise_distance": min_pairwise_distance,
                "near_miss_count": near_miss_count,
                "active_orders": active_orders,
            }
        )

    def log_zone(self, tick: int, zone_id: str, queue_depth: int, robot_density: int) -> None:
        self.zone_rows.append(
            {
                "tick": tick,
                "zone_id": zone_id,
                "queue_depth": queue_depth,
                "robot_density": robot_density,
            }
        )

    def log_robot(self, tick: int, robot_id: int, x: int, y: int, state: str, order_id: int | None) -> None:
        self.robot_rows.append(
            {
                "tick": tick,
                "robot_id": robot_id,
                "x": x,
                "y": y,
                "state": state,
                "order_id": order_id,
            }
        )

    def log_event(self, tick: int, event_type: str, **fields) -> None:
        self.event_rows.append({"tick": tick, "type": event_type, **fields})

    def log_forecast(
        self,
        tick: int,
        congestion_prob: float | None,
        collision_prob: float | None,
        classifier_triggered: bool,
        backstop_triggered: bool,
        congestion_prob_raw: float | None = None,
        collision_prob_raw: float | None = None,
    ) -> None:
        self.forecast_rows.append(
            {
                "tick": tick,
                "congestion_prob": congestion_prob,
                "collision_prob": collision_prob,
                "congestion_prob_raw": congestion_prob_raw,
                "collision_prob_raw": collision_prob_raw,
                "classifier_triggered": classifier_triggered,
                "backstop_triggered": backstop_triggered,
                "triggered": classifier_triggered or backstop_triggered,
            }
        )

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "ticks": pd.DataFrame(self.tick_rows),
            "zones": pd.DataFrame(self.zone_rows),
            "robots": pd.DataFrame(self.robot_rows),
            "events": pd.DataFrame(self.event_rows),
            "forecast": pd.DataFrame(self.forecast_rows),
        }

    def save(self, out_dir: str | Path, orders: list, manifest: dict) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = self.to_frames()
        for name, df in frames.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)

        order_rows = [
            {
                "id": o.id,
                "origin_x": o.origin[0],
                "origin_y": o.origin[1],
                "destination_x": o.destination[0],
                "destination_y": o.destination[1],
                "arrival_tick": o.arrival_tick,
                "status": o.status.value,
                "assigned_robot": o.assigned_robot,
                "assign_tick": o.assign_tick,
                "pickup_tick": o.pickup_tick,
                "dropoff_tick": o.dropoff_tick,
                "wait_ticks": o.wait_ticks,
            }
            for o in orders
        ]
        pd.DataFrame(order_rows).to_parquet(out_dir / "orders.parquet", index=False)

        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
