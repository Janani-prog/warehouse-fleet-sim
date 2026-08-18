# PRD — Anomaly Detection & Collision-Safe Navigation in ML-Driven Autonomous Warehouse Robots

Status: v2 (re-scoped)
Owner: [your name]
Build agent: Claude Code
Budget: $0 — free compute/hosting only

## 1. Problem Statement
Simulated multi-robot warehouses need three coordination decisions solved continuously: which robot does which task, how each robot gets there, and how robots avoid colliding en route. Each decision has a classical solution and an increasingly common learned alternative. This project builds both for all three problems, layers a predictive anomaly-detection system on top (LSTM classifier), grounds any automated response in retrieved company procedure (RAG), and lets a small LLM agent pick a corrective action from a fixed, whitelisted set. The system's actual value is measured with paired-trial causal evaluation, not asserted.

## 2. Why This Version Is Different
The original design fed raw LSTM forecast output directly into the agent's decision loop. Forecast precision came out low in earlier attempts, and the agent inherited that unreliability with no safeguard. This version treats forecaster-agent coupling as a first-class design problem:
- Forecasting is reframed as **classification** ("will risk exceed threshold in the next N steps?") rather than continuous multi-step regression — a better-behaved, more tractable target for a small model trained on simulator data.
- The agent only acts when forecast confidence is **calibrated and above threshold**; below threshold, the system logs but does not act.
- A **rule-based backstop** (queue length / robot density thresholds) runs in parallel as a redundant trigger, so the loop degrades gracefully if the LSTM underperforms.
- The evaluation explicitly measures **net benefit under imperfect precision** — this is a feature of the study, not a caveat buried in limitations.

## 3. Goals
- G1: Build a working multi-robot warehouse simulator (grid or continuous space) supporting order arrival, robot fleet, and obstacle/collision physics.
- G2: Implement classical and learned solutions for task scheduling, path planning, and traffic/collision avoidance; benchmark head-to-head.
- G3: Build a fleet-anomaly classifier (LSTM) that predicts congestion/collision risk ahead of time, calibrated for confidence.
- G4: Build a hybrid RAG pipeline over a synthetic SOP corpus that retrieves the correct procedure for a detected anomaly type.
- G5: Build an LLM agent (prompt-based baseline, LoRA fine-tune as stretch) that selects one action from a whitelist given the anomaly + retrieved SOP, and executes it in the simulator.
- G6: Evaluate the full closed loop with paired-trial causal methodology (same seed, with/without intervention, paired statistical test).
- G7: Present all of the above through a clean, minimal dashboard.

## 4. Non-Goals
- No real hardware, no ROS, no physical robots.
- No paid APIs, cloud GPUs, or hosting of any kind.
- No general-purpose LLM chat interface — the agent's action space is fixed and whitelisted, never free-form.
- No multi-warehouse / multi-facility scope — single simulated warehouse floor.
- No real company data — SOP corpus is synthetic, authored for this project.

## 5. Users / Audience
- Primary: capstone evaluators/committee assessing rigor, novelty, and correctness of methodology.
- Secondary: you, as the operator running experiments and reading the dashboard.

## 6. Scope Tiers
**P0 (must-have, core to the title):**
- Simulator core
- Path planning: A* vs. learned planner
- Traffic/collision avoidance: ORCA vs. learned policy
- LSTM anomaly classifier + calibration + rule backstop
- RAG pipeline + synthetic SOP corpus
- Prompt-based LLM agent + whitelisted action execution
- Causal (paired-trial) evaluation harness
- Dashboard (read-only, visualization of the above)

**P1 (include if P0 is solid):**
- Task scheduling: Hungarian algorithm vs. learned scheduler
- LoRA fine-tuning of the agent model + ablation vs. prompt-only

**P2 (stretch, cut freely):**
- Multiple warehouse layouts / stress scenarios
- Adaptive confidence threshold tuning UI
- Exporting results as a formatted report

## 7. Success Metrics
- **Correctness**: classical baselines (A*, ORCA, Hungarian) match published behavior on toy scenarios (unit-testable).
- **ML viability**: learned alternatives reach within an agreed tolerance of classical baseline performance on at least one axis (e.g., learned planner matches A* path length within X%, or beats it on compute time).
- **Forecast quality**: classifier precision/recall reported at the chosen calibrated operating point, not just raw AUC.
- **Agent reliability**: agent picks a valid, SOP-grounded action ≥ target accuracy on a held-out labeled anomaly set (target defined once corpus exists, e.g. ≥85%).
- **Causal impact**: paired-trial test shows statistically significant (p < 0.05) improvement in at least one fleet KPI (throughput, mean wait time, collision-near-miss count) when the closed loop is active vs. inactive, holding scenario seed constant.

## 8. Key Risks
| Risk | Mitigation |
|---|---|
| Forecast precision too low to trust | Classification reframing + confidence gating + rule backstop (see §2) |
| LLM picks unsafe/invalid action | Hard whitelist enforced in code, never trust raw LLM text — parse into a validated enum |
| Learned planner/policy fails to converge on laptop CPU | Keep state/action space small; cap training to short episodes; classical baseline always available as fallback |
| Scope creep across 7 subsystems | Tiered P0/P1/P2 scope, milestone-gated backlog (see BACKLOG.md) |
| Causal eval underpowered (too few trials) | Fully scripted, cheap-to-rerun simulator means trials cost only CPU time — run enough seeds (≥30 paired trials recommended) |

## 9. Constraints
- 100% free: simulator, forecasting, RAG, dashboard on CPU; LLM fine-tuning (if attempted) on free-tier Colab GPU, served locally via Ollama after quantization.
- Single-machine, no external services requiring an account with billing.
