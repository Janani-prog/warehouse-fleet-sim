# CLAUDE.md — Project Memory

Read this file first in any new session before doing anything else. It is the source of truth for scope, decisions already made, and where the project currently stands. Update the "Current Status" section at the end of every work session.

## Project
Anomaly Detection & Collision-Safe Navigation in ML-Driven Autonomous Warehouse Robots — a simulation-only academic capstone. No hardware, no paid services, no external hosting.

## Companion Docs (read these too)
- `PRD.md` — goals, scope tiers, success metrics
- `TECHNICAL_ARCHITECTURE.md` — full system design, components, data flow
- `FRONTEND_SPEC.md` — dashboard design spec + Stitch prompt
- `SECURITY.md` — safety/security scope
- `BACKLOG.md` — phased milestone tickets, split into two review-gated parts. **Build Part 1 (M0–M6) in full first** — it's a complete, demoable system on its own (simulator + classical-vs-learned path planning & traffic avoidance + calibrated forecaster + dashboard). Part 2 (M7–M12) adds RAG, the LLM agent, causal evaluation, and closes the loop. Do not start Part 2 tickets before Part 1's "Review 1 Demo Checklist" passes and the repo is tagged (`git tag review-1`).

## Locked Design Decisions (do not silently revisit these)
1. **Forecasting is classification, not regression.** Predict P(anomaly in next N ticks) as binary/multi-class, not a continuous multi-step time series. This was a deliberate re-scope after an earlier attempt found LSTM regression precision too low to trust downstream.
2. **Agent action triggering is confidence-gated + backstopped.** The LLM agent only acts when calibrated LSTM confidence clears a threshold OR a simple rule-based backstop fires. Never wire raw/uncalibrated model output directly into the action loop.
3. **LLM agent is prompt-based by default (via Ollama), fine-tuning is optional stretch.** Only invest Colab GPU time in LoRA fine-tuning if the prompt-only baseline's accuracy on the labeled action-selection test set is inadequate. If fine-tuning is done, keep it as an ablation comparison, not a replacement — report both.
4. **The LLM never executes actions directly.** All output is structured JSON, parsed and validated against a hard-coded whitelist enum (`REASSIGN_TASK`, `REPLAN_ROUTE`, `THROTTLE_ZONE_TRAFFIC`, `NO_ACTION`) by the Action Executor before touching simulator state. Malformed output → `NO_ACTION`, logged, never crashes.
5. **Scope tiers**: Path Planning and Traffic/Collision Avoidance are P0 (core to the project title). Scheduling (Hungarian vs learned) is P1 — full comparison, but the first thing cut under time pressure. LoRA fine-tuning is also P1.
6. **Evaluation is causal, not observational.** Every claimed improvement must come from paired trials (same seed, intervention ON vs OFF) with a statistical test — never "before/after" framing without pairing.
7. **Everything must run free.** Local CPU for sim/ML/RAG/dashboard. Colab free-tier GPU only if fine-tuning is attempted, then quantize and serve locally via Ollama. No paid API calls anywhere in the core pipeline.

## Repo Structure
```
sim/        - simulator core, telemetry logging
ml/         - scheduling, path planning, traffic policies (classical + learned), LSTM forecaster
rag/        - SOP corpus, retrieval pipeline
agent/      - prompt templates, Ollama client, action executor, (optional) LoRA fine-tune scripts
eval/       - causal evaluation harness, benchmark scripts
dashboard/  - React + FastAPI dashboard
data/       - generated datasets, trial logs (gitignored if large)
docs/       - PRD, architecture, frontend spec, security, backlog (this set of files)
```

## Permission to Deviate From Spec
These docs are the plan, not a cage. If, once something is actually built and you can see real behavior/data/results, you believe a different approach would genuinely be better than what's specified here (in the PRD, architecture doc, or backlog) — you're authorized to say so and propose it. This is expected and welcome, not a violation of the plan.

The bar and process for doing this:
- This applies to **implementation choices below the "Locked Design Decisions" level** (e.g. "actually a gradient-boosted tree beats the LSTM classifier on this data" or "the imitation-learned planner needs a different feature set than planned") — propose and proceed with those directly, just document what changed and why.
- For anything that would **change a Locked Decision itself** (the classification-not-regression framing, confidence-gating, the whitelist-only execution model, prompt-first LLM approach, causal-not-observational evaluation, or the $0 constraint) — flag it explicitly, explain the evidence/reasoning, and get a go-ahead before changing course. These exist because they were the product of a specific earlier failure mode (the forecaster-reliability issue), so they need a real conversation, not a silent swap.
- Either way: **never change course silently.** Every deviation, big or small, gets a short note — either inline in the relevant milestone's ticket in `BACKLOG.md`, or in the "Current Status" log below — stating what was specified, what was done instead, and why. The goal is that a fresh session (or you, months later) can reconstruct the reasoning, not just the current state.
- "I can do better" should be backed by something concrete you actually observed while building (a benchmark result, a failure mode, a constraint that turned out to bind harder than expected) — not a preference expressed before anything's been tried.

## Version Control
- **Prerequisite (you do this once, not Claude Code)**: create an empty GitHub repo and either set it as the local `origin` remote, or have the `gh` CLI authenticated (`gh auth login`) so Claude Code can create/push it. Claude Code should not attempt to invent credentials or silently create accounts — if push fails due to auth, it should stop and tell you rather than working around it.
- **Commit per logical unit of work**, not per milestone-and-a-half — e.g. "A* implementation + tests", then separately "learned planner + benchmark script", not one giant commit for all of M2. Commit messages: short imperative summary line, optional body explaining *why* for anything non-obvious (especially any deviation logged per the Permission to Deviate section).
- **Push regularly**, at minimum at the end of every milestone, ideally after each meaningful commit, so the GitHub history is a real record of progress, not a single end-of-session dump.
- **Tag part boundaries**: `git tag review-1` at the end of M6 (Part 1 wrap-up), `git tag review-2` at the end of M12. These are the two points a reviewer might actually check out.
- **Branching**: keep it simple — commit to `main` directly for this solo/academic project unless a milestone involves a risky rewrite you want to be able to abandon cleanly, in which case use a short-lived feature branch and merge back once the milestone's "Done when" condition is met.
- **.gitignore**: exclude `data/` outputs that are large/regeneratable (trial logs, generated datasets, model checkpoints) unless small enough to be useful as committed fixtures; exclude any `.env`, `node_modules/`, Python venvs, and Colab-downloaded model weights.
- **Never commit secrets** — there shouldn't be any in this project's core pipeline (see SECURITY.md §4), but if any local `.env` is ever introduced, it stays gitignored.
- After pushing, note the commit/tag in the "Current Status" section below so a fresh session knows exactly what's already on GitHub.

## Version Control
- Use git from the start (M0). Commit at meaningful checkpoints — at minimum once per completed ticket, not just once per milestone — with clear, descriptive messages (what changed and why, not just "update").
- Author identity: use whatever `git config user.name` / `user.email` is already set locally (i.e. the repo owner's own identity). Do not add, change, or override author/committer identity for any commit.
- **No AI-tool attribution anywhere in the repo.** No `Co-authored-by` trailers, no "Generated with Claude" / "Built with Claude Code" notices, no mentions of Claude or AI assistance in commit messages, code comments, README, or docs. Commits and the repo as a whole should read as ordinary authored work.
- Push to the GitHub remote regularly — at least after every completed milestone, ideally after every completed ticket. If no remote is configured yet, ask for the GitHub repo URL (or create one via `gh repo create` if the `gh` CLI is authenticated) before the first push.
- Tag part boundaries: `git tag review-1` at the end of M6 (Part 1 wrap-up), `git tag review-2` at the end of M12 (final writeup/polish).
- Use a `.gitignore` covering the usual (`__pycache__/`, `node_modules/`, `.env`, large generated `data/` artifacts, model checkpoints unless intentionally versioned).
- Normal branch hygiene is fine but not required — direct commits to `main` per milestone are acceptable for a solo project; use feature branches only if it genuinely helps you avoid a broken `main` mid-milestone.

## Working Agreement for Claude Code Sessions
- Work through `BACKLOG.md` milestones in order (M0 → M10). Don't start a later milestone's tickets before the current one's "Done when" condition is met.
- After finishing a milestone, write a short entry in "Current Status" below (what's done, what's next, any open issue) before ending the session — this is what makes a fresh chat able to resume without re-deriving context.
- If a design decision from the "Locked Design Decisions" list needs to change, say so explicitly and update this file — don't just quietly deviate.
- Prefer simple, testable implementations over clever ones — every classical algorithm (A*, ORCA, Hungarian) needs unit tests before its learned counterpart is built, since the classical version is both the baseline and the ground-truth generator for imitation learning.
- Every run that produces a benchmark or eval result should be reproducible from a single script/command with a fixed seed.

## Current Status
_(update this section each session)_

- **Last updated**: 2026-08-18
- **Current part**: Part 1 (Review 1 target)
- **Current milestone**: M2 — complete
- **Completed**:
  - Repo skeleton (`sim/`, `ml/`, `rag/`, `agent/`, `eval/`, `dashboard/`, `data/`, `docs/`) created; source docs (PRD, TECHNICAL_ARCHITECTURE, FRONTEND_SPEC, SECURITY, BACKLOG) moved into `docs/` under canonical filenames.
  - The Stitch design export (5 dashboard screens: fleet map, anomaly timeline, agent action log, causal eval report, loading state — HTML/CSS mockups + screenshots + a full design-token spec) extracted to `docs/design/stitch_warehouse_fleet_operations_console/`. Use this as the concrete visual reference for M5 so the dashboard looks like a finished internal tool, not a placeholder.
  - Python env: `.venv` + `requirements.txt` (numpy, scipy, pandas, pyarrow, torch CPU, scikit-learn, matplotlib, fastapi, uvicorn, pydantic, pytest), installs clean via `--extra-index-url https://download.pytorch.org/whl/cpu` (keeps torch to the ~205MB CPU wheel instead of the much larger CUDA build).
  - **Deviation (below Locked Decisions, not requiring sign-off)**: RAG/agent-only deps (`sentence-transformers`, `rank_bm25`, `faiss-cpu`, `peft`, `transformers`, an Ollama client) are deliberately *not* in `requirements.txt` yet — they're unused until M7/M8 (Part 2), so adding them now would just slow every `pip install` for no benefit. Add them when M7 starts.
  - `pyproject.toml` configures pytest (`pythonpath = ["."]`, testpaths under `sim/ml/eval` `tests/`); one placeholder test passes, confirming the harness is wired.
  - Dashboard scaffolded with Vite + React + TypeScript + Tailwind v4 (via the official `@tailwindcss/vite` plugin, not the old `tailwind init`/PostCSS flow, since v4 dropped that). Vite's demo boilerplate (counter, logos, marketing copy) stripped down to an empty placeholder page — real views land in M5 against the Stitch mockups.
  - Git initialized, default branch renamed `master` → `main`. GitHub repo created via authenticated `gh` CLI: **https://github.com/Janani-prog/warehouse-fleet-sim** (public), set as `origin`. Three logical commits (docs/design, Python skeleton, dashboard scaffold) pushed to `main`.
  - Verified both from clean: `pip install -r requirements.txt` succeeds, `npm install` succeeds, `npx vite build` succeeds, `pytest` passes.

- **M1 — Simulator Core**:
  - `sim/grid.py`: fixed, deterministic 24×16 warehouse layout — 3 horizontal rack bands with vertical cross-aisle gaps, 16 zones (6×4 cells each) for aggregate metrics, 5 dock points on the left edge doubling as robot spawn points and order drop-offs, pickup points on the aisle cells bordering each rack band.
  - `sim/entities.py`: `Robot` (position, state, current order, path stub for M2) and `Order` (origin/destination, lifecycle status, arrival/assign/pickup/dropoff ticks, `wait_ticks`).
  - `sim/order_generator.py`: Poisson-arrival order generator driven by the world's single shared `np.random.Generator`, seeded once — this is what makes runs deterministic end to end.
  - `sim/policies.py`: **M1 placeholder movement only** — greedy random walk biased 80% toward the current target cell, no real pathing or collision avoidance. This is intentionally dumb; it's replaced by A*/learned planning in M2 and ORCA/learned avoidance in M3. Nearest-idle-robot order assignment (Manhattan distance) is likewise a placeholder for Hungarian/learned scheduling (M11, P1).
  - `sim/telemetry.py`: per-tick Parquet tables — `ticks` (min pairwise distance, near-miss count, active orders), `zones` (queue depth + robot density per zone), `robots` (full per-tick position/state for replay), `events` (order arrived/assigned/picked-up/completed), plus `orders.parquet` (full lifecycle) and `manifest.json` (seed/config). This schema is designed to directly serve both M4 (windowed features per tick) and M5 (fleet-map replay) without rework.
  - `sim/run.py`: CLI (`python -m sim.run --seed --ticks --robots --order-rate --out`) — done-when condition verified: a 500-tick, 6-robot, seed-0 run produces a full replayable telemetry log in `data/runs/demo/` (gitignored, regeneratable).
  - Tests (`sim/tests/`): 9 passing — zone/layout sanity, spawn/dock/pickup points free, no-rack-neighbors; same-seed runs produce byte-identical robot trajectories and order histories, different seeds diverge; no robot or order position ever lands out of bounds or inside a rack across a run.
  - **Known tuning issue, not a bug**: with the M1 placeholder random-walk policy, default demo params (6 robots, order_rate=0.4) let the pending-order backlog grow largely unbounded over 500 ticks (mean wait ~184 ticks, only ~20% of orders completed) — the random walk is just too inefficient to keep up. Expected and fine for M1 (whose job is only to prove determinism/bounds/telemetry/replay work), but the default scenario params will need retuning once A* (M2) and real avoidance (M3) land, so the Review-1 demo run shows a fleet that's actually keeping up with load rather than perpetually backlogged.
- **Next up**: M2 — Path Planning. A* implementation + unit tests, A*-solved dataset generation via the simulator, imitation-learned planner trained on that dataset, benchmark script (path length ratio, latency, success rate) comparing classical vs. learned. Swap `sim/policies.py`'s random walk for real A* pathing once it exists.

- **M2 — Path Planning**:
  - `ml/astar.py`: standard 4-connectivity, unit-cost A* with Manhattan heuristic. 6 tests in `ml/tests/test_astar.py` including known-optimal-length cases (open grid, forced detour through a single gap) and real-layout pickup→dropoff routing.
  - `ml/generate_planning_dataset.py`: samples realistic trips (pickup↔dropoff and general free-cell pairs) and solves each with A*, saved to Parquet. Train set (seed 0, n=500) and a separately-seeded held-out test set (seed 999, n=150) in `data/datasets/` (gitignored, regenerate via the script).
  - Learned planner: local-observation imitation policy, not a global planner. `ml/planner_features.py` encodes state as a flattened 5×5 occupancy window around the current cell + normalized goal delta (27-dim) — deliberately small/local per the architecture doc's CPU-tractability guidance. `ml/planner_model.py` is a 2-hidden-layer MLP (64 units) predicting one of 4 moves. `ml/train_planner.py` trains it via supervised cross-entropy against A*'s chosen action at each step of every training path (30 epochs, reached ~97% held-out per-step action accuracy). `ml/learned_planner.py` does the autoregressive rollout (repeated single-step queries, since it has no global search) with a revisit-limited fallback so it can't infinite-loop between two cells.
  - `ml/benchmark_planning.py`: classical vs. learned on the 150 held-out trips, reproducible from a fixed seed. Result: **both 100% success rate**; learned paths average **7.1% longer** than A*'s optimal (well within the PRD's "matches within X%" bar) but A* is faster in wall-clock (~0.24ms vs ~3.4ms per query — the MLP forward-pass overhead exceeds heapq A*'s cost at this grid's tiny size, so the "learned wins on compute time" alternative-axis framing from the PRD does *not* hold here and I'm reporting that honestly rather than reframing the metric). Table: `data/results/planning_benchmark.csv`; chart: `data/results/planning_benchmark.png`.
  - **Wired A* into the live simulator** (`sim/world.py`): robots now cache a full A* path when their target changes (order assigned → pickup; picked up → dropoff) and walk it one cell/tick, replanning only on target change — not the learned planner, which stays a benchmark-only artifact for now (`sim.policies.PlannerFn` is a swappable interface if a learned-driven sim run is wanted later). Idle robots hold position instead of wandering (was M1's random-walk default) — deliberate, for a calmer/more legible fleet-map demo.
  - **Bug found and fixed during integration**: robots whose cached A* paths crossed could wait on each other's occupied cell forever — a genuine permanent deadlock, reproduced with as few as 2 robots, not just a throughput/tuning issue. Fixed with a `stuck_ticks` counter per robot (`sim/entities.py`) and a minimal liveness guarantee in `sim/world.py`: after `STUCK_THRESHOLD=3` blocked ticks, a robot force-side-steps onto any free unclaimed neighbor cell and its path is dropped for a fresh replan next tick (logged as a `deadlock_broken` telemetry event for auditability). This is explicitly a stopgap, not real avoidance — M3's ORCA/learned traffic policy is the principled fix. Regression tests in `sim/tests/test_deadlock.py`.
  - **Default demo params retuned** (`World.__init__`, `sim/run.py` CLI): 6 robots/order_rate=0.4 (M1's placeholder values) left the queue growing unboundedly even with real A* pathing, because fleet throughput capacity was below arrival rate — this surfaced the deadlock bug too, since it's far more visible under dense traffic. Swept a few (robots, order_rate) combinations post-fix; **8 robots / order_rate=0.15** keeps the backlog stable (~92-96% order completion, active-order count settling rather than climbing) and is now the default. Documented here rather than in the backlog since it's a parameter choice, not a scope change.
  - All 26 tests pass (`sim/`: 17, `ml/`: 9). `data/datasets/`, `data/models/`, `data/results/`, `data/runs/` are all gitignored/regeneratable — rerun `ml/generate_planning_dataset.py` → `ml/train_planner.py` → `ml/benchmark_planning.py` → `sim/run.py` in that order from a clean clone to reproduce everything.
- **Next up**: M3 — Traffic/Collision Avoidance. ORCA (or simplified velocity-obstacle) implementation + unit tests, learned local avoidance policy (imitation from ORCA or independent-Q RL), benchmark (near-miss/collision count, throughput, mean detour) classical vs. learned. This directly replaces the M2 stopgap deadlock-breaker with principled multi-robot coordination — expect `deadlock_broken` event counts to become a "before" baseline the M3 benchmark can meaningfully compare against.
- **Open issues / blockers**: —
