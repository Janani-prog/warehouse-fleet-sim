import type { RunSummary } from "../lib/api";
import { DotIcon } from "./icons";

interface Props {
  runs: RunSummary[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  title: string;
}

export function TopBar({ runs, selectedRunId, onSelectRun, title }: Props) {
  return (
    <header className="fixed top-0 right-0 left-sidebar-width z-10 bg-surface border-b border-outline-variant flex justify-between items-center h-row-height-md px-container-padding">
      <div className="flex items-center gap-4">
        <h2 className="text-[15px] font-semibold text-on-surface">{title}</h2>
        <select
          className="h-7 px-2 bg-surface-container-lowest border border-outline-variant rounded text-[12px] font-mono text-on-surface"
          value={selectedRunId ?? ""}
          onChange={(e) => onSelectRun(e.target.value)}
        >
          {runs.length === 0 && <option value="">no runs found</option>}
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} (seed {r.manifest.seed})
            </option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-4">
        <span
          title="Correction loop lands in Part 2 — the agent, RAG retrieval, and action executor aren't built yet."
          className="flex items-center gap-1.5 text-[12px] font-medium text-outline cursor-not-allowed select-none"
        >
          Correction Loop: Off (Part 2)
        </span>
        <div className="h-4 w-px bg-outline-variant" />
        <span className="flex items-center gap-1.5 text-[12px] text-secondary">
          <DotIcon className="text-primary" />
          Replay ready
        </span>
      </div>
    </header>
  );
}
