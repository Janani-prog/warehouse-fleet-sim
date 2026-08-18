# Security & Safety Considerations

This is a local, offline, simulation-only academic project with no real hardware, no real user data, and no external network exposure by default. Security work here is scoped narrowly but should still be treated seriously, since the closed-loop agent-executes-actions pattern is a real-world-relevant design worth doing correctly.

## 1. LLM Action Safety (primary concern)
- **The LLM never executes anything directly.** It only ever produces structured text (JSON with an `action` field). All execution goes through the Action Executor (see TECHNICAL_ARCHITECTURE.md §2.8), which:
  - Parses output defensively — malformed/non-JSON output is treated as `NO_ACTION`, logged, and never crashes the loop.
  - Validates `action` is a member of the hard-coded whitelist enum (`REASSIGN_TASK`, `REPLAN_ROUTE`, `THROTTLE_ZONE_TRAFFIC`, `NO_ACTION`) — any value outside this set is rejected.
  - Validates any referenced parameters (robot ID, zone ID) actually exist in current simulator state before acting — prevents acting on hallucinated entities.
  - Never allows the agent to modify its own whitelist, prompt, or the executor logic. The agent's "creativity" is scoped to which whitelisted action to pick, not what actions exist.
- No arbitrary code execution from LLM output at any point (no `eval`, no dynamic import of agent-authored code).
- Prompt injection surface: retrieved SOP text is inserted into the agent prompt. Since the SOP corpus is authored by you (not user-submitted or scraped from an untrusted source), injection risk is low — but the executor's whitelist enforcement is the real safeguard regardless of what the prompt contains, by design.

## 2. Forecast → Action Coupling Safety
- Confidence-gated triggering (§2.5 of architecture doc) exists partly for reliability and partly for safety: the system should fail toward inaction (safe default) rather than acting on low-confidence signals.
- All triggers (classifier and rule backstop) and all resulting actions are logged with full context, so any unexpected behavior is auditable after the fact.

## 3. Data
- All telemetry, SOP corpus, and training data are synthetic/simulator-generated. No real company data, no PII, no real facility layouts.
- If any real-world SOP language is used as inspiration, rewrite it rather than copying verbatim, and don't attribute it to a real company by name.

## 4. Local Environment
- No secrets/API keys required anywhere in the core pipeline (everything free and local). If you add optional cloud fallbacks later, keep any keys in a local `.env` excluded from version control (`.gitignore`) — never commit them.
- Ollama and any local model server should bind to localhost only, not `0.0.0.0`, unless you deliberately need LAN access for a demo.
- Dependency hygiene: pin versions in `requirements.txt`/`package.json`, periodically check for known CVEs in dependencies (`pip-audit`, `npm audit`) — low priority for an academic project but easy to do and worth a line in the writeup.

## 5. Evaluation Integrity
- Trial logs for the causal evaluation should be treated as immutable once written (append-only, timestamped files) — don't allow ad hoc post-hoc editing of trial results, to keep the evaluation credible for the committee.
- Random seeds used per trial should be logged and reproducible, so results can be independently re-run.

## 6. Explicit Non-Threats (for scoping honesty in the writeup)
- No multi-tenant access control needed (single local user).
- No network-facing attack surface unless you choose to deploy the dashboard beyond localhost.
- No adversarial-robustness testing of the ML models is in scope (could be a "future work" line, not a requirement).
