# Frontend Spec — Fleet Ops Dashboard

## 1. Purpose
A read-only operations dashboard for the simulated warehouse fleet, used to (a) demonstrate the system working during evaluation/defense, and (b) let you inspect what the agent did and why. It is a supporting artifact, not the core deliverable — it must look like a real internal tool, not an AI-generated demo page.

## 2. Design Principles
- **Clean, professional, minimalistic, clutter-free.** Dense information, sparse decoration. Think internal ops tooling (Linear, Vercel dashboard, Datadog) — not a marketing landing page.
- No gradients-as-decoration, no glassmorphism, no emoji in UI chrome, no stock illustration, no "AI sparkle" iconography. These are the tells that make a dashboard look AI-generated.
- One accent color used sparingly (status/alerts only); neutral grayscale for everything else.
- Real typographic hierarchy (2–3 weights, consistent scale) instead of everything bold or everything the same size.
- Generous whitespace over borders/boxes to separate sections — avoid boxing every single element in its own card with a shadow.
- Numbers and state changes should be legible at a glance — this is a monitoring tool, optimize for scanning, not for scrolling prose.

## 3. Key Views
1. **Fleet Map** — top-down grid view of the warehouse, robot positions (colored by state: idle/moving/blocked), live order pins, zone boundaries. Minimal chrome, map is the focus.
2. **KPI Strip** — throughput, mean wait time, active robots, current near-miss count. Small multiples, not big vanity numbers.
3. **Anomaly Timeline** — time series of calibrated confidence score with a visible threshold line; markers where the rule backstop or classifier triggered; markers where an action was taken.
4. **Agent Action Log** — reverse-chronological list: timestamp, anomaly type + confidence, retrieved SOP snippet, action taken, and outcome. This is the most important view for demonstrating the "why" of the system — make it readable, not just a raw log dump.
5. **Causal Evaluation Report** — paired-trial results: per-metric deltas across seeds (small box/strip plot), test statistic, p-value, effect size, plain-language one-line interpretation.

## 4. Layout
- Persistent left sidebar (5 nav items above, icon + label, no more).
- Top bar: scenario/seed selector, loop ON/OFF toggle (for demoing paired trials live), run status indicator.
- Main content area: single view at a time, generous margins, max content width ~1200px, don't stretch dense tables/maps edge-to-edge on large screens.

## 5. Component Notes
- Fleet map: SVG or canvas grid render, not a 3D engine — keep it crisp and fast.
- Charts: Recharts, minimal gridlines, no drop shadows, muted color palette with one accent for alerts/threshold lines.
- Tables (action log): fixed-width columns, monospace for IDs/timestamps, normal weight for prose (SOP snippets/reasoning).
- Empty/loading states: plain, no illustrations — a short neutral sentence is enough.

## 6. Tech
- React + Vite + TypeScript, Tailwind for styling (utility-first keeps output disciplined and avoids ad hoc CSS drift), Recharts for charts, FastAPI backend serving JSON from local telemetry/log storage.

---

## 7. Prompt for Google Stitch

Paste this into Stitch as-is to generate the initial dashboard design:

```
Design a clean, professional, minimalist internal operations dashboard for monitoring a simulated autonomous warehouse robot fleet. This is an internal engineering/ops tool, not a consumer or marketing product — it should look like Linear, Vercel's dashboard, or Datadog, not like a generic AI-generated demo page.

Layout: persistent left sidebar with 5 nav items (Fleet Map, KPIs, Anomaly Timeline, Agent Action Log, Causal Evaluation Report), each with a simple line icon and label. Top bar with a scenario/seed selector dropdown, an ON/OFF toggle labeled "Correction Loop", and a small status indicator dot (running/idle). Main content area, max width around 1200px, generous margins, not stretched edge-to-edge.

Visual style: neutral grayscale palette (near-white background, dark charcoal text, light gray borders/dividers) with exactly one accent color used only for alerts, thresholds, and the active nav item — nothing else uses color. No gradients, no glassmorphism, no drop shadows on cards, no emoji, no decorative illustrations or 3D icons. Typography: one sans-serif family, 2–3 weights maximum, clear size hierarchy between page titles, section headers, and body/data text. Prefer whitespace and alignment over boxes and borders to separate sections.

Design 5 screens:
1. Fleet Map — a top-down grid layout of a warehouse floor with small colored dots representing robots (color = state: idle, moving, blocked) moving between rack rows, plus a few pin markers for active orders. A slim KPI strip above the map showing throughput, mean wait time, active robots, and current near-miss count as small, quiet numbers (not oversized hero stats).
2. Anomaly Timeline — a horizontal time-series line chart of a confidence score from 0 to 1, with a dashed horizontal threshold line, and small markers on the line where the system triggered an action.
3. Agent Action Log — a clean reverse-chronological table/list: timestamp, anomaly type, confidence percentage, a short retrieved procedure snippet, the action taken (as a small pill/tag), and outcome. Monospace for timestamps/IDs, normal weight for text content.
4. Causal Evaluation Report — a results page showing 3-4 small box/strip plots comparing paired trial outcomes (loop ON vs OFF) for different metrics, with a statistics summary (test statistic, p-value, effect size) in a simple key-value list below, and one plain-language sentence interpreting the result.
5. Empty/loading state — plain, centered, single neutral sentence, no illustration.

Overall impression: dense, legible, quiet, built for scanning by an engineer — not decorative.
```
