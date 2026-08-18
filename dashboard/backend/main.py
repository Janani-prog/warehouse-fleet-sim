"""Dashboard backend: serves a completed simulator run's telemetry as JSON
for the frontend to render (static replay, per M5's scope - no live
streaming). Reads directly from the Parquet files sim.telemetry.TelemetryLogger
writes; the warehouse layout itself isn't persisted per-run since it's a
fixed, deterministic floor plan (sim.grid.generate_default_layout), so it's
just regenerated here rather than duplicated to disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sim.grid import generate_default_layout

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "runs"
FORECASTER_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "forecaster"

app = FastAPI(title="Fleet Ops Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _run_dir(run_id: str) -> Path:
    run_dir = DATA_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return run_dir


def _read_parquet_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    # NaN isn't valid JSON; FastAPI's default encoder would emit it anyway
    # (invalid per spec) - convert to None explicitly instead.
    df = df.astype(object).where(pd.notnull(df), None)
    return json.loads(df.to_json(orient="records"))


@app.get("/api/runs")
def list_runs() -> list[dict]:
    if not DATA_DIR.exists():
        return []
    runs = []
    for run_dir in sorted(DATA_DIR.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        runs.append({"run_id": run_dir.name, "manifest": manifest})
    return runs


@app.get("/api/runs/{run_id}/manifest")
def get_manifest(run_id: str) -> dict:
    run_dir = _run_dir(run_id)
    with open(run_dir / "manifest.json") as f:
        return json.load(f)


@app.get("/api/warehouse")
def get_warehouse() -> dict:
    """The floor plan is fixed across every run, so this isn't run-scoped."""
    wh = generate_default_layout()
    return {
        "width": wh.width,
        "height": wh.height,
        "zone_size": list(wh.zone_size),
        "racks": [list(c) for c in sorted(wh.racks)],
        "spawn_points": [list(c) for c in wh.spawn_points],
        "dropoff_points": [list(c) for c in wh.dropoff_points],
        "pickup_points": [list(c) for c in wh.pickup_points],
    }


@app.get("/api/runs/{run_id}/robots")
def get_robots(run_id: str) -> list[dict]:
    return _read_parquet_records(_run_dir(run_id) / "robots.parquet")


@app.get("/api/runs/{run_id}/zones")
def get_zones(run_id: str) -> list[dict]:
    return _read_parquet_records(_run_dir(run_id) / "zones.parquet")


@app.get("/api/runs/{run_id}/orders")
def get_orders(run_id: str) -> list[dict]:
    return _read_parquet_records(_run_dir(run_id) / "orders.parquet")


@app.get("/api/runs/{run_id}/ticks")
def get_ticks(run_id: str) -> list[dict]:
    records = _read_parquet_records(_run_dir(run_id) / "ticks.parquet")
    for row in records:
        if row.get("min_pairwise_distance") is not None and math.isnan(row["min_pairwise_distance"]):
            row["min_pairwise_distance"] = None
    return records


@app.get("/api/runs/{run_id}/forecast")
def get_forecast(run_id: str) -> list[dict]:
    return _read_parquet_records(_run_dir(run_id) / "forecast.parquet")


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str) -> list[dict]:
    return _read_parquet_records(_run_dir(run_id) / "events.parquet")


@app.get("/api/forecaster/thresholds")
def get_forecaster_thresholds() -> dict:
    """The actual trained calibrated-confidence operating thresholds (see
    ml/train_forecaster.py) - the Anomaly Timeline draws its threshold line
    from this, not a guessed constant."""
    path = FORECASTER_MODEL_DIR / "thresholds.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="forecaster model not trained yet")
    with open(path) as f:
        return json.load(f)


@app.get("/api/runs/{run_id}/kpis")
def get_kpis(run_id: str) -> dict:
    run_dir = _run_dir(run_id)
    orders = pd.read_parquet(run_dir / "orders.parquet") if (run_dir / "orders.parquet").exists() else pd.DataFrame()
    ticks = pd.read_parquet(run_dir / "ticks.parquet") if (run_dir / "ticks.parquet").exists() else pd.DataFrame()
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)

    completed = orders[orders["status"] == "completed"] if not orders.empty else orders
    num_ticks = manifest.get("num_ticks", 0)
    throughput_per_100_ticks = (len(completed) / num_ticks * 100) if num_ticks else 0.0

    return {
        "throughput_per_100_ticks": round(throughput_per_100_ticks, 2),
        "mean_wait_ticks": (
            round(float(completed["wait_ticks"].mean()), 1) if not completed.empty else None
        ),
        "num_robots": manifest.get("num_robots"),
        "active_orders_now": (
            int(ticks.iloc[-1]["active_orders"]) if not ticks.empty else 0
        ),
        "near_miss_count_now": (
            int(ticks.iloc[-1]["near_miss_count"]) if not ticks.empty else 0
        ),
        "total_orders": len(orders),
        "completed_orders": len(completed),
        "completion_rate": round(len(completed) / len(orders), 3) if len(orders) else None,
        "num_ticks": num_ticks,
    }
