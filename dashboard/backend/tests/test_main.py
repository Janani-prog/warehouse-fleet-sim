import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dashboard.backend.main as main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    return TestClient(main.app)


def _write_run(run_dir, manifest, orders=None, ticks=None, robots=None):
    run_dir.mkdir(parents=True)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    if orders is not None:
        pd.DataFrame(orders).to_parquet(run_dir / "orders.parquet", index=False)
    if ticks is not None:
        pd.DataFrame(ticks).to_parquet(run_dir / "ticks.parquet", index=False)
    if robots is not None:
        pd.DataFrame(robots).to_parquet(run_dir / "robots.parquet", index=False)


def test_list_runs_empty(client):
    assert client.get("/api/runs").json() == []


def test_list_runs_returns_manifest(client, tmp_path):
    _write_run(tmp_path / "run_a", {"seed": 0, "num_robots": 4})
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["run_id"] == "run_a"
    assert data[0]["manifest"]["num_robots"] == 4


def test_get_manifest_404_for_missing_run(client):
    resp = client.get("/api/runs/does-not-exist/manifest")
    assert resp.status_code == 404


def test_warehouse_endpoint_shape(client):
    resp = client.get("/api/warehouse")
    assert resp.status_code == 200
    data = resp.json()
    assert data["width"] > 0 and data["height"] > 0
    assert len(data["racks"]) > 0
    assert all(len(cell) == 2 for cell in data["racks"][:5])


def test_robots_endpoint_reads_parquet(client, tmp_path):
    _write_run(
        tmp_path / "run_b",
        {"num_robots": 1},
        robots=[{"tick": 0, "robot_id": 0, "x": 1, "y": 2, "state": "idle", "order_id": None}],
    )
    resp = client.get("/api/runs/run_b/robots")
    assert resp.status_code == 200
    assert resp.json() == [{"tick": 0, "robot_id": 0, "x": 1, "y": 2, "state": "idle", "order_id": None}]


def test_robots_endpoint_missing_file_returns_empty_list(client, tmp_path):
    _write_run(tmp_path / "run_c", {"num_robots": 1})
    resp = client.get("/api/runs/run_c/robots")
    assert resp.status_code == 200
    assert resp.json() == []


def test_forecaster_thresholds_404_when_model_untrained(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "FORECASTER_MODEL_DIR", tmp_path / "no_such_model")
    resp = client.get("/api/forecaster/thresholds")
    assert resp.status_code == 404


def test_forecaster_thresholds_reads_json(client, tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with open(model_dir / "thresholds.json", "w") as f:
        json.dump({"congestion": 0.5, "collision": 0.6}, f)
    monkeypatch.setattr(main, "FORECASTER_MODEL_DIR", model_dir)
    resp = client.get("/api/forecaster/thresholds")
    assert resp.status_code == 200
    assert resp.json() == {"congestion": 0.5, "collision": 0.6}


def test_kpis_computes_completion_rate_and_wait(client, tmp_path):
    _write_run(
        tmp_path / "run_d",
        {"num_robots": 2, "num_ticks": 100},
        orders=[
            {"id": 0, "status": "completed", "wait_ticks": 10},
            {"id": 1, "status": "completed", "wait_ticks": 20},
            {"id": 2, "status": "pending", "wait_ticks": None},
        ],
        ticks=[
            {"tick": 0, "active_orders": 3, "near_miss_count": 0, "min_pairwise_distance": 5.0},
            {"tick": 1, "active_orders": 2, "near_miss_count": 1, "min_pairwise_distance": 3.0},
        ],
    )
    resp = client.get("/api/runs/run_d/kpis")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_orders"] == 3
    assert data["completed_orders"] == 2
    assert data["completion_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert data["mean_wait_ticks"] == 15.0
    assert data["active_orders_now"] == 2
    assert data["near_miss_count_now"] == 1
