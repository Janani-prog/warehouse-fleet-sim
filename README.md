# Warehouse Fleet Simulation & Autonomous Correction Loop

Simulation-only academic capstone: anomaly detection and collision-safe navigation
in ML-driven autonomous warehouse robots. No hardware, no paid services, no
external hosting.

See `CLAUDE.md` for project memory and current status, and `docs/` for the full
PRD, technical architecture, frontend spec, security notes, and phased backlog.

## Status

Part 1 (simulator, classical-vs-learned planning/traffic, calibrated anomaly
forecaster, dashboard) is in progress. See `CLAUDE.md` → Current Status for
exactly where things stand.

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
npm run dev
```
