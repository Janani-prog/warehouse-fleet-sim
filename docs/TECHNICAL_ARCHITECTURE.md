# Technical Architecture — Warehouse Fleet Simulation & Autonomous Correction Loop

## 1. System Overview

```
                     ┌─────────────────────────┐
                     │      Simulator Core       │
                     │  (grid world, robots,     │
                     │   orders, physics/tick)   │
                     └──────────┬─────────────┘
              ┌──────────────┼──────────────┐
              ▼                ▼                ▼
      ┌──────────┐   ┌──────────────┐   ┌──────────────┐
      │ Scheduling │   │ Path Planning│   │Traffic/Collision│
      │ Hungarian  │   │  A* vs       │   │  ORCA vs      │
      │ vs Learned │   │  Learned     │   │  Learned      │
      └──────────┘   └──────────────┘   └──────────────┘
              └──────────────┼──────────────┘
                             ▼
                   ┌───────────────────┐
                   │  Fleet Telemetry    │  (queue depth, density,
                   │      Stream         │   near-miss counts, wait times)
                   └─────────┬──────────┘
                             ▼
                   ┌───────────────────┐
                   │  LSTM Anomaly       │  binary/multi-class
                   │  Classifier +       │  classification,
                   │  Calibration +      │  calibrated confidence
                   │  Rule Backstop      │
                   └─────────┬──────────┘
                     (only if confidence ≥ threshold OR rule fires)
                             ▼
                   ┌───────────────────┐
                   │  Hybrid RAG          │  retrieves matching
                   │  (BM25 + embedding)  │  synthetic SOP
                   └─────────┬──────────┘
                             ▼
                   ┌───────────────────┐
                   │  LLM Agent           │  prompt-based (Ollama),
                   │  (whitelisted        │  optional LoRA fine-tune
                   │   action selector)   │
                   └─────────┬──────────┘
                             ▼
                   ┌───────────────────┐
                   │  Action Executor     │  validates action against
                   │  (hard-coded         │  whitelist enum before
                   │   whitelist)         │  touching simulator state
                   └─────────┬──────────┘
                             ▼
                     back into Simulator Core

                   ┌───────────────────┐
                   │ Causal Eval Harness │  paired trials (seed-matched),
                   │                      │  loop ON vs OFF, stats test
                   └───────────────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │     Dashboard        │  read-only visualization
                   └───────────────────┘
```

## 2. Components

### 2.1 Simulator Core
- Discrete-time tick-based simulation, grid-based warehouse (simpler, faster, still supports A*/ORCA-style local avoidance on a fine grid or continuous local layer).
- Entities: robots (position, velocity, task, battery optional), shelves/racks, orders (origin→destination, arrival time), obstacles.
- Deterministic given a random seed — required for paired-trial causal evaluation.
- Emits a structured telemetry stream every tick (JSON or Arrow/Parquet rows): queue depth per zone, robot density per zone, pairwise min-distance, near-miss events, per-order wait time.
- Tech: Python, NumPy; no physics engine needed at grid granularity.

### 2.2 Task Scheduling (P1)
- **Classical**: Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) — minimize total travel cost assigning orders to idle robots.
- **Learned**: lightweight learned scheduler — start with a greedy learned heuristic (small MLP scoring robot-order pairs, trained via supervised imitation of Hungarian solutions, or simple RL if time allows) rather than jumping straight to full RL, since imitation learning is far more stable to train on CPU.
- Comparison metric: total assignment cost, mean order wait time, compute time.

### 2.3 Path Planning (P0)
- **Classical**: A* on the grid with standard heuristics.
- **Learned**: value-iteration-trained or imitation-learned planner (train on A* solutions across many random start/goal pairs — supervised learning of a policy network is far more tractable on CPU than RL from scratch).
- Comparison metric: path length vs. optimal, planning latency, success rate.

### 2.4 Traffic / Collision Avoidance (P0)
- **Classical**: ORCA (Optimal Reciprocal Collision Avoidance) — use an existing lightweight Python ORCA implementation or a simplified velocity-obstacle variant.
- **Learned**: small policy network trained via imitation of ORCA trajectories, or lightweight multi-agent RL (e.g. independent Q-learning) on a reduced local observation space — keep the observation/action space small (local neighbors only) to stay CPU-tractable.
- Comparison metric: collision/near-miss count, throughput, mean detour length.

### 2.5 Anomaly Forecaster
- **Reframed as classification**: given a rolling window of fleet telemetry (e.g. last 30 ticks), predict P(congestion event in next N ticks) and P(collision-risk event in next N ticks) as separate binary heads, or one multi-class head.
- Model: small LSTM (1–2 layers, ≤64 hidden units) — trains in minutes on CPU given the small feature/window size.
- **Calibration**: apply temperature scaling or simple isotonic regression on a held-out validation split so the output probability is a genuinely calibrated confidence, not a raw logit.
- **Confidence gate**: agent loop only triggers when calibrated confidence ≥ a chosen operating threshold (tuned via precision/recall tradeoff on validation data — report this explicitly, don't hide it).
- **Rule-based backstop**: simple threshold rule (e.g., queue depth > X or local robot density > Y) runs independently; either the classifier (above threshold) or the rule firing can trigger the pipeline. This is the redundancy that protects the loop from a single unreliable model.
- Training data: generated entirely from the simulator by running many episodes and labeling windows that precede a labeled anomaly event.

### 2.6 Hybrid RAG Pipeline
- Corpus: a synthetic set of ~15–30 "company SOP" documents you author, each describing a response procedure for a specific anomaly type/severity (e.g. "Zone congestion, moderate severity → throttle inbound traffic to zone for 60s").
- Retrieval: hybrid = BM25 (keyword, via `rank_bm25`) + dense embedding similarity (a small free sentence-embedding model via `sentence-transformers`, e.g. `all-MiniLM-L6-v2`, run locally on CPU), combined via reciprocal rank fusion or simple weighted score.
- Output: top-1 or top-k SOP passage, passed into the agent prompt as grounding context.

### 2.7 LLM Agent
- **Baseline (P0)**: prompt-based agent. Local model served via Ollama (e.g. `llama3.1:8b-instruct-q4` or `qwen2.5:7b-instruct-q4`, quantized to fit laptop RAM). Prompt includes: anomaly type + confidence, retrieved SOP text, fixed whitelist of actions with descriptions, and structured-output instructions (JSON with an `action` field constrained to the whitelist enum + a `reason` field).
- **Stretch (P1)**: LoRA fine-tune of a small base model on Colab's free GPU tier using a dataset of (anomaly + SOP + correct action) examples generated from your own simulator runs / synthetic corpus, then quantize (e.g. GGUF via `llama.cpp` conversion) and serve locally via Ollama. Compare fine-tuned vs. prompt-only accuracy as an ablation.
- **Whitelisted actions** (fixed enum, extend only with care): `REASSIGN_TASK`, `REPLAN_ROUTE`, `THROTTLE_ZONE_TRAFFIC`, `NO_ACTION`.

### 2.8 Action Executor
- Never executes raw LLM text. Parses the LLM's structured JSON output, validates `action` is a member of the whitelist enum, validates any parameters (zone ID, robot ID) exist in current simulator state, and only then calls the corresponding simulator API. Invalid/unparseable output → logged, treated as `NO_ACTION`, never crashes the sim.

### 2.9 Causal Evaluation Harness
- Paired-trial design: for each of N random seeds, run the full episode twice — once with the predict→retrieve→act loop enabled, once with it disabled (classical baselines only) — everything else identical.
- Metrics per trial: throughput, mean order wait time, collision/near-miss count, congestion duration.
- Statistical test: paired t-test or Wilcoxon signed-rank test (nonparametric, safer default) on the metric deltas across seed pairs; report effect size and confidence interval, not just p-value.
- Recommend N ≥ 30 paired trials for reasonable power; trials are cheap since the simulator is fast and free.

### 2.10 Dashboard
- Read-only visualization: live fleet map, current KPIs, anomaly/confidence timeline, agent action log with retrieved SOP + reasoning shown per action, and a causal-eval results view (paired deltas, test statistic, effect size).
- See FRONTEND_SPEC.md for design detail.

## 3. Tech Stack (all free)
- Language: Python 3.11 (simulator, ML, RAG, agent glue), TypeScript/React for dashboard.
- ML: PyTorch (CPU) for LSTM + learned scheduler/planner/policy; scikit-learn for baselines/calibration.
- Classical algorithms: `scipy` (Hungarian), custom A*, a Python ORCA implementation (e.g. adapt from an existing open-source lightweight implementation).
- RAG: `rank_bm25`, `sentence-transformers` (CPU), local vector store (`faiss-cpu` or simple in-memory cosine search — faiss is optional overkill at this corpus size).
- LLM serving: Ollama, local quantized open model.
- Fine-tuning (if pursued): Colab free tier, `peft`/LoRA, `transformers`, export to GGUF for Ollama.
- Dashboard: React + Vite, simple charting (Recharts), FastAPI backend serving telemetry/log data from local files/SQLite.
- Storage: SQLite or flat Parquet/JSON files for telemetry, trial logs, and agent decision logs — no external DB needed.
- Orchestration: everything runs locally via scripts/CLI; no containers required, though a `docker-compose.yml` is fine as an optional convenience.

## 4. Data Flow Summary
Simulator tick → telemetry stream → rolling window → classifier + rule backstop → (if triggered) RAG retrieval → agent prompt → structured action → validated executor → simulator state update → next tick. Every stage logs to disk for the dashboard and for causal-eval trial replay/audit.
