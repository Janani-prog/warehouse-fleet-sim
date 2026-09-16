# Warehouse Fleet Simulation & Autonomous Correction Loop

Simulation-only academic capstone: anomaly detection and collision-safe navigation
in ML-driven autonomous warehouse robots. No hardware, no paid services, no
external hosting — everything runs free, locally, on CPU (plus a local Ollama
LLM for Part 2).

See `CLAUDE.md` for full project memory (locked design decisions, session-by-session
status log, and every documented deviation from the original plan) and `docs/` for
the PRD, technical architecture, frontend spec, security notes, and phased backlog.

## Status: Part 1 + Part 2 complete (Part 1 tagged `review-1`, Part 2 tagged `review-2`)

The system **observes, predicts, decides, and acts** — end to end, with a hard
safety boundary between "the LLM proposes" and "the simulator's state actually
changes." M11 (Hungarian vs. learned task scheduling) was not attempted: it's
explicitly scoped as P1 in `docs/BACKLOG.md` ("the first thing cut under time
pressure"), and this session hit real time/memory pressure finishing Part 2 — see
CLAUDE.md's session notes for what that cut and why.

## Part 1 — observe and predict

- **Simulator** (`sim/`) — a deterministic, seeded, tick-based multi-robot warehouse:
  fixed 24×16 grid layout, Poisson order arrivals, full per-tick telemetry (positions,
  near-misses, queue depth, wait times) logged to Parquet for replay.
- **Path planning** (`ml/astar.py`, `ml/learned_planner.py`) — classical A* vs. an
  imitation-learned local-observation MLP policy. Both hit 100% success rate on a
  150-trip held-out benchmark; the learned planner's paths average **7.1% longer**
  than A*'s optimal, and (reported honestly, not reframed) A* is also **faster in
  wall-clock** at this grid's small size (~0.24ms vs ~3.4ms per query — the MLP's
  forward-pass overhead exceeds heapq A*'s cost here). `data/results/planning_benchmark.csv`.
- **Traffic / collision avoidance** (`ml/traffic.py`, `ml/learned_traffic.py`) — a
  classical priority-based conflict resolver (the discrete analogue of ORCA/velocity
  obstacles, since the sim is grid-based) vs. a learned binary proceed/yield
  classifier that's always routed back through the same classical safety net before
  it can touch simulator state. Paired benchmark over 20 held-out seeds
  (Wilcoxon signed-rank): completion rate is statistically indistinguishable
  (p=0.212), near-misses are **22.5% lower** under the learned policy (p=0.0027),
  at the cost of a **+3.1 tick** longer mean detour per order (p=0.0023) — a genuine
  tradeoff, not a one-sided win. `data/results/traffic_benchmark.csv`.
- **Anomaly forecaster** (`ml/forecaster_model.py`, `ml/calibration.py`) — a
  temperature-scaled LSTM classifier predicting P(congestion) and P(collision-risk)
  in the next 10 ticks from a rolling 30-tick telemetry window, backed by an
  independent rule-based backstop. Held-out test set: congestion precision 0.856 /
  recall 0.996 / F1 0.921; collision-risk precision 0.914 / recall 0.669 / F1 0.773
  (harder target — near-misses are noisier/less autocorrelated than order backlog).
  On the scripted congestion-spike demo, calibrated confidence crosses its trained
  threshold **16 ticks before** the actual congestion event (tick 235 vs. 251) —
  genuine early warning, not reactive detection.

## Part 2 — decide and act (RAG-grounded LLM agent, whitelisted executor, causal eval)

- **RAG retrieval** (`rag/`) — 18 hand-authored SOP documents (2 anomaly types ×
  4 severity bands, plus zone-specific variants and cross-cutting procedures like
  false-positive handling and backstop-vs-classifier precedence), retrieved via
  hybrid BM25 (`rank_bm25`) + dense embedding search. Dense embeddings go through
  local Ollama (`nomic-embed-text`) rather than `sentence-transformers`, since
  Ollama is already required for the agent — see CLAUDE.md's M7 note. Fusion is
  weighted Reciprocal Rank Fusion (dense weighted over BM25 — found and fixed a
  real bug where equal weighting let a glossary-style SOP outrank the correct doc
  on live queries). Retrieval eval on 18 hand-labeled queries: **top-1 83.3%, top-3
  100%** — `data/results/retrieval_benchmark.csv`.
- **LLM agent + action executor** (`agent/`) — a prompt-based agent via Ollama
  (`llama3.1:8b`, Locked Decision #3: fine-tuning only if prompt-only proves
  inadequate) proposes exactly one of four whitelisted actions
  (`REASSIGN_TASK`, `REPLAN_ROUTE`, `THROTTLE_ZONE_TRAFFIC`, `NO_ACTION`) as
  structured JSON, grounded in the retrieved SOP and a short list of pre-validated
  candidate targets (so the model never invents a zone/robot/order id — it only
  confirms or rejects an offered one). `agent/executor.py` independently
  re-validates the action and target against the current whitelist and simulator
  state before ever calling into the simulator; malformed/adversarial/unparseable
  output always falls back to `NO_ACTION`, logged, never crashes (proven with an
  adversarial test using a real-but-unoffered target id). Agent accuracy on a
  16-case labeled (anomaly, SOP, correct action) eval set: **16/16 (100%)** —
  well above the threshold for pursuing LoRA fine-tuning, so per Locked Decision #3
  it was not attempted. `data/results/agent_benchmark.csv`.
- **Causal evaluation** (`eval/`) — paired-trial harness: same seed, closed loop ON
  vs. OFF, everything else identical, Wilcoxon signed-rank + Cohen's d + bootstrap
  95% CI per metric (throughput, mean wait, near-miss total, congestion duration).
  **Result, reported honestly**: the full run was cut short at **N=14** (not the
  recommended N≥30) after repeated out-of-memory kills on the dev machine running
  this session (Ollama's model plus many concurrent apps competing for RAM — a real
  environment constraint, documented in CLAUDE.md, not a bug in the harness, which
  now checkpoints incrementally so a kill loses at most one in-flight pair). At
  N=14, no metric reaches statistical significance, and `congestion_ticks` is
  structurally zero across all 28 trials — the compressed scenario used to make
  N pairs tractable under CPU-only LLM inference never actually pushes the fleet's
  backlog past the congestion threshold, a real finding about that scenario's
  sizing, not a null result to read too much into. `near_miss_total` and
  `mean_wait` do vary meaningfully per seed, so those two are genuine
  underpowered-null findings. **A fresh session with a quieter machine re-running
  `python -m eval.run_causal_eval --n-pairs 30` would be the single highest-value
  next step** if this project continues — the harness itself is correct and tested,
  it just didn't get to finish. `data/results/causal_eval/causal_eval_report.json`.
- **Dashboard, Part 2 extension** (`dashboard/`) — two new views: **Agent Action
  Log** (every triggered tick's retrieve → decide → act cycle, expandable to show
  the LLM's full reasoning and raw JSON output) and **Causal Evaluation Report**
  (per-metric cards with mean delta, 95% CI, p-value, plain-language
  interpretation). Verified via direct backend API checks and clean
  `tsc`/`vite build` (no browser-automation tool was available in the execution
  environment for a full visual pass — a documented gap, not a silent skip).

## Part 1 dashboard views (unchanged from Review 1)

**Fleet Map** (animated replay, play/pause + scrubber, real per-tick near-miss
count), **KPIs** (throughput, completion rate, wait times), **Anomaly Timeline**
(calibrated confidence vs. tick, with the real trained threshold and trigger
points).

Every classical algorithm has unit tests and is also the ground-truth generator for
its learned counterpart; every benchmark reproduces from a single seeded command
(see below). 151 tests pass across `sim`, `ml`, `dashboard`, `rag`, `agent`, `eval`.

## Setup

**Python (simulator / ML / RAG / agent / eval):**

```
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
pytest
```

**Ollama (required for Part 2 — RAG dense embeddings + the LLM agent):**

Install Ollama, then pull the two models this project uses:

```
ollama pull nomic-embed-text
ollama pull llama3.1:8b
ollama serve
```

Part 1 (simulator, planning, traffic, forecaster) and Part 1's dashboard views
work without Ollama running at all. `rag/`, `agent/`, and `eval/`'s live-Ollama
tests skip cleanly (not fail) if `ollama serve` isn't reachable.

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

# RAG retrieval eval (needs `ollama serve` running)
python -m rag.benchmark_retrieval

# Agent accuracy eval (needs `ollama serve` running, llama3.1:8b pulled)
python -m agent.benchmark_agent

# Full closed-loop demo run — logged trigger->retrieve->decide->act cycles
# (slow: one LLM call per triggered tick; start with fewer --ticks if your
# machine is memory-constrained, see CLAUDE.md's session notes)
python -m agent.run_agent_demo --ticks 500 --out data/runs/agent_closed_loop_demo

# Causal evaluation: paired trials, loop ON vs OFF (slow for the same reason;
# checkpoints incrementally to data/results/causal_eval/causal_eval_trials.csv,
# so an interrupted run can be resumed by just re-running the same command)
python -m eval.run_causal_eval --n-pairs 30 --out data/results/causal_eval
```

Then start the dashboard (above) and pick a run from the selector:
- `demo` or `congestion_spike_demo` for the three Part-1 views (the latter has real
  forecaster confidence data — this is the one that shows the Anomaly Timeline's
  early-warning behavior)
- `agent_closed_loop_demo` for the Agent Action Log view
- the Causal Evaluation Report view reads `data/results/causal_eval/` directly,
  not a specific run

## Known limitations, stated plainly

- M9's causal evaluation ran at N=14 pairs, not the recommended N≥30 — see above.
- M11 (Hungarian vs. learned scheduling) was not attempted (P1, explicitly the
  first thing cut under time pressure per the backlog's scope tiers).
- M10's new dashboard views were verified via API + type-check + build, not a live
  browser pass (no browser-automation tool was available in this session).
- This dev machine has a real, repeated memory constraint running Ollama-backed
  scripts alongside many other open applications — anyone reproducing the Part 2
  benchmarks on a similarly constrained machine should expect the same, and should
  know `eval/causal_harness.py`'s checkpointing exists for exactly this reason.
