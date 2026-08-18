# Warehouse Fleet Simulation & Autonomous Correction Loop

Simulation-only academic capstone: anomaly detection and collision-safe navigation
in ML-driven autonomous warehouse robots. No hardware, no paid services, no
external hosting.

See `CLAUDE.md` for full project memory (locked design decisions, session-by-session
status log) and `docs/` for the PRD, technical architecture, frontend spec, security
notes, and phased backlog.

## Status: Part 1 complete (Review 1 target)

The system currently **observes and predicts; it does not yet act autonomously**.
That's Part 2 (RAG-grounded LLM agent + whitelisted action executor + causal
evaluation of whether the corrective loop actually helps) — not started yet.

What's built so far is a complete, self-contained pipeline in its own right:

- **Simulator** (`sim/`) — a deterministic, seeded, tick-based multi-robot warehouse:
  fixed 24×16 grid layout, Poisson order arrivals, full per-tick telemetry (positions,
  near-misses, queue depth, wait times) logged to Parquet for replay.
- **Path planning** (`ml/astar.py`, `ml/learned_planner.py`) — classical A* vs. an
  imitation-learned local-observation MLP policy. Both hit 100% success rate on a
  150-trip held-out benchmark; the learned planner's paths average 7.1% longer than
  A*'s optimal, and (reported honestly, not reframed) A* is also faster in wall-clock
  at this grid's small size. See `data/results/planning_benchmark.csv` after
  regenerating (see below).
- **Traffic / collision avoidance** (`ml/traffic.py`, `ml/learned_traffic.py`) — a
  classical priority-based conflict resolver (the discrete analogue of ORCA/velocity
  obstacles, since the sim is grid-based) vs. a learned binary proceed/yield
  classifier that's always routed back through the same classical safety net before
  it can touch simulator state. Paired benchmark over 20 held-out seeds
  (Wilcoxon signed-rank): completion rate is statistically indistinguishable
  (p=0.21), near-misses are **22.5% lower** under the learned policy (p=0.0027),
  at the cost of a **+3.1 tick** longer mean detour per order (p=0.0023) — a genuine
  tradeoff, not a one-sided win. See `data/results/traffic_benchmark.csv`.
- **Anomaly forecaster** (`ml/forecaster_model.py`, `ml/calibration.py`) — a
  temperature-scaled LSTM classifier predicting P(congestion) and P(collision-risk)
  in the next 10 ticks from a rolling 30-tick telemetry window, backed by an
  independent rule-based backstop. On a scripted congestion-spike scenario, the
  calibrated confidence crosses its trained threshold **16 ticks before** the actual
  congestion event — see the Anomaly Timeline in the dashboard, run
  `congestion_spike_demo`.
- **Dashboard** (`dashboard/`) — FastAPI backend serving each run's telemetry as JSON,
  React/TypeScript frontend (styled from the project's Stitch design export) with
  three working views: **Fleet Map** (animated replay with play/pause + scrubber),
  **KPIs** (throughput, completion rate, wait times), and **Anomaly Timeline**
  (calibrated confidence vs. tick, with the real trained threshold and trigger
  points). Agent Action Log and Causal Evaluation Report are visible in the nav but
  marked "Part 2" and intentionally not implemented yet — there's no data to show
  in them until the agent exists.

Every classical algorithm has unit tests and is also the ground-truth generator for
its learned counterpart; every benchmark reproduces from a single seeded command
(see below).

## What's next (Part 2)

RAG pipeline over a synthetic SOP corpus, a prompt-based LLM agent (via Ollama) that
proposes one of a hard-coded whitelist of actions, an action executor that validates
before ever touching simulator state, and a causal evaluation harness (paired
same-seed trials, loop ON vs. OFF, with a statistical test) to determine whether the
closed loop actually helps rather than just "looking like it does." Full detail in
`docs/BACKLOG.md`.

## Setup

**Python (simulator / ML / eval):**

```
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
pytest
```

**Dashboard:**

```
cd dashboard
npm install
npm run dev              # frontend, http://localhost:5173
```

In a second terminal, from the repo root:

```
.venv/Scripts/activate
uvicorn dashboard.backend.main:app --reload --port 8000
```

The dashboard's run picker reads whatever is under `data/runs/`; see below for how
to (re)generate a run.

## Reproducing everything from a clean clone

`data/` (datasets, trained models, run logs, benchmark results) is gitignored — it's
all regenerable from seeded scripts. Run in this order:

```
# Path planning
python -m ml.generate_planning_dataset
python -m ml.train_planner
python -m ml.benchmark_planning

# Traffic / collision avoidance
python -m ml.generate_traffic_dataset
python -m ml.train_traffic
python -m ml.benchmark_traffic

# Anomaly forecaster
python -m ml.generate_forecast_dataset
python -m ml.train_forecaster

# A default demo run (steady load) and the scripted congestion-spike demo
python -m sim.run --seed 0 --ticks 500 --robots 8 --order-rate 0.15 --out data/runs/demo
python -m ml.run_forecast_demo
```

Then start the dashboard (above) and pick either `demo` (steady load) or
`congestion_spike_demo` (has real forecaster confidence data — this is the one that
shows the Anomaly Timeline's early-warning behavior) from the run selector.
