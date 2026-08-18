# Phased Backlog

No calendar deadlines — phases are milestone-gated. Each milestone must leave the system in a working, demoable state before moving to the next. P0 = must-have, P1 = include if P0 solid, P2 = stretch/cut freely.

The project is split into two review-gated parts. **Build Part 1 first, in full, before touching any Part 2 ticket.** Part 1 is deliberately scoped to be a complete, coherent, demoable system on its own — not a random half of the pipeline — so it stands on its own for the first review even though it's not the full closed-loop system yet.

---

# PART 1 — Review 1 Target

Scope: a working multi-robot simulator with classical-vs-learned comparisons for path planning and traffic/collision avoidance, a calibrated anomaly forecaster running live against the simulator, and a dashboard that visualizes all of it. This deliberately excludes RAG, the LLM agent, fine-tuning, and causal evaluation — those are Part 2, and are the riskier/more novel pieces that benefit from having Part 1's foundation already solid and tested.

### Review 1 Demo Checklist (what must be true when this is shown)
- [ ] A live/replayable simulation run with a visible fleet of robots moving, completing orders, avoiding collisions
- [ ] Side-by-side comparison numbers for classical vs. learned path planning (and traffic avoidance if ready)
- [ ] The anomaly forecaster visibly flagging risk before it happens on at least one scripted scenario (e.g. an engineered congestion spike), with its calibrated confidence shown
- [ ] Dashboard showing: fleet map, KPI strip, anomaly timeline — polished enough to present, not a debug view
- [ ] A short README explaining what's built, what's next (Part 2), framed as "system currently observes and predicts; does not yet act autonomously — that's Part 2"

## M0 — Project Skeleton [P0]
- [ ] Repo structure: `sim/`, `ml/`, `rag/`, `agent/`, `eval/`, `dashboard/`, `data/`, `docs/`
- [ ] Python env + `requirements.txt`; Node env for dashboard
- [ ] CLAUDE.md committed (see separate file)
- [ ] Basic CI-free local test runner (`pytest`) wired, even with zero tests yet
- **Done when**: `pip install -r requirements.txt` and `npm install` both succeed clean.

## M1 — Simulator Core [P0]
- [ ] Grid warehouse layout representation (racks, zones, spawn/dock points)
- [ ] Robot entity + tick-based movement
- [ ] Order generator (arrival process, origin/destination)
- [ ] Deterministic seeding (same seed → identical run)
- [ ] Telemetry logger (per-tick JSON/Parquet rows: queue depth, density, min pairwise distance, wait times)
- [ ] Unit tests: seed determinism, no entities spawn out-of-bounds
- **Done when**: a scripted run with dummy random-walk robots produces a replayable telemetry log.

## M2 — Path Planning [P0]
- [ ] A* implementation + unit tests (known-optimal-path test cases)
- [ ] Generate A*-solved (start, goal, path) dataset via simulator
- [ ] Learned planner: imitation-learned policy network trained on A* dataset
- [ ] Benchmark script: path length ratio, latency, success rate, classical vs learned
- **Done when**: benchmark report (table or chart) comparing A* vs learned planner exists and runs reproducibly.

## M3 — Traffic / Collision Avoidance [P0]
- [ ] ORCA (or simplified velocity-obstacle) implementation + unit tests (two-robot head-on case, etc.)
- [ ] Learned local avoidance policy (imitation from ORCA trajectories or small independent-Q RL)
- [ ] Benchmark script: near-miss/collision count, throughput, mean detour — classical vs learned
- **Done when**: multi-robot scenario runs collision-free (or with logged near-misses) under both approaches, with a comparison report.

## M4 — Anomaly Forecaster [P0]
- [ ] Label anomaly events in simulator (congestion, collision-risk) from telemetry rules
- [ ] Generate windowed training dataset from many simulated episodes
- [ ] Train LSTM classifier (binary/multi-class), CPU-only
- [ ] Calibration step (temperature scaling or isotonic regression) on held-out split
- [ ] Rule-based backstop implementation (independent of the LSTM)
- [ ] Report: precision/recall at chosen operating threshold, calibration curve
- **Done when**: classifier + backstop both produce trigger signals on a live simulator run, with confidence values visible in logs.

## M5 — Dashboard, Part 1 scope [P0]
- [ ] FastAPI backend serving telemetry/log data from local storage
- [ ] Fleet Map, KPI strip, Anomaly Timeline views only (see FRONTEND_SPEC.md) — Agent Action Log and Causal Eval Report views are Part 2, once there's data to show in them
- [ ] Wire to a completed simulator run (static replay is fine for the demo — live streaming is a nice-to-have, not required)
- **Done when**: dashboard loads a real run's data and all three Part-1 views render correctly and look presentable.

## M6 — Part 1 Wrap-up [P0]
- [ ] Run through the Review 1 Demo Checklist above end to end
- [ ] Write the "what's built / what's next" README section
- [ ] Tag/branch the repo state at this point (e.g. `git tag review-1`) so Part 2 work is clearly separated and Part 1 stays reproducible for the record
- **Done when**: a fresh clone can install, run the demo scenario, and show the dashboard, with no Part 2 code required.

---

# PART 2 — Review 2 Target

Scope: everything that closes the loop — retrieval-grounded decisions, the LLM agent, whitelisted autonomous action, causal evaluation of whether the loop actually helps, and (if pursued) fine-tuning. Start this only after Part 1 is tagged and demoed.

## M7 — RAG Pipeline [P0]
- [ ] Author synthetic SOP corpus (15–30 docs covering anomaly types × severities)
- [ ] BM25 index + dense embedding index (local sentence-transformers model)
- [ ] Hybrid retrieval scoring (RRF or weighted fusion)
- [ ] Retrieval eval: hand-label 10–20 test queries with expected SOP, measure top-1/top-3 accuracy
- **Done when**: given an anomaly type, retrieval returns the correct SOP ≥ target accuracy on the labeled test set.

## M8 — LLM Agent + Action Executor [P0]
- [ ] Define whitelist action enum + executor validation logic
- [ ] Prompt-based agent via Ollama (structured JSON output, few-shot examples)
- [ ] Agent accuracy eval on labeled (anomaly, SOP, correct action) test set
- [ ] Full closed loop wired: forecaster/backstop trigger → RAG → agent → executor → simulator
- [ ] **[P1 stretch]** LoRA fine-tuning on Colab free GPU using generated (anomaly, SOP, action) training examples; quantize to GGUF; serve via Ollama; compare accuracy vs. prompt-only
- **Done when**: a live simulator run demonstrates at least one full trigger→retrieve→decide→act cycle end-to-end, logged.

## M9 — Causal Evaluation Harness [P0]
- [ ] Paired-trial runner: same seed, loop ON vs OFF, N ≥ 30 pairs
- [ ] Metrics collection per trial (throughput, wait time, near-miss count, congestion duration)
- [ ] Statistical test (Wilcoxon signed-rank default; paired t-test if normality holds) + effect size + CI
- [ ] Results report (table/plots) with plain-language interpretation
- **Done when**: running the harness produces a reproducible statistical report from a single command.

## M10 — Dashboard, Part 2 extension [P0]
- [ ] Extend the Part 1 dashboard (don't rebuild): add Agent Action Log and Causal Eval Report views (see FRONTEND_SPEC.md)
- [ ] Wire to closed-loop run data and causal-eval harness output
- **Done when**: dashboard loads a full Part-2 run and all 5 views (3 from Part 1 + these 2) render correctly.

## M11 — Scheduling (Hungarian vs Learned) [P1]
- [ ] Hungarian algorithm assignment + unit tests
- [ ] Imitation-learned scheduler trained on Hungarian solutions
- [ ] Benchmark: assignment cost, wait time, compute time
- **Done when**: comparison report exists; integrate into simulator's order-assignment step if time allows.

## M12 — Writeup / Polish [P0]
- [ ] Consolidate all benchmark reports (planning, traffic, forecasting, RAG, agent, causal eval) into final report
- [ ] Record a short demo run (video or live) showing dashboard + one full corrective cycle
- [ ] Clean up README, ensure fresh clone + install + run works end to end
