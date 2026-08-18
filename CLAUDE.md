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

- **Last updated**: [date]
- **Current part**: Part 1 (Review 1 target)
- **Current milestone**: M0 — not yet started
- **Completed**: —
- **Next up**: repo skeleton + environment setup, then work M0 → M6 in order. Stop and demo once M6's Review 1 Demo Checklist passes — do not start Part 2 (M7+) until then.
- **Open issues / blockers**: —
