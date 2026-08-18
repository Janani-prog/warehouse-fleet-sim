import { useEffect, useState } from "react";
import { Sidebar, type ViewId } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { api, type RunSummary } from "./lib/api";
import { FleetMapView } from "./views/FleetMapView";
import { KpisView } from "./views/KpisView";
import { AnomalyTimelineView } from "./views/AnomalyTimelineView";
import { ComingSoonView } from "./views/ComingSoonView";

const VIEW_TITLES: Record<ViewId, string> = {
  fleet_map: "Fleet Map",
  kpis: "KPIs",
  anomaly_timeline: "Anomaly Timeline",
  agent_log: "Agent Action Log",
  causal_eval: "Causal Evaluation Report",
};

function App() {
  const [view, setView] = useState<ViewId>("fleet_map");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((data) => {
        setRuns(data);
        if (data.length > 0) setSelectedRunId(data[0].run_id);
      })
      .catch(() =>
        setError(
          "Could not reach the dashboard backend at localhost:8000 — start it with `uvicorn dashboard.backend.main:app --reload`."
        )
      );
  }, []);

  return (
    <div className="min-h-screen bg-surface-bright">
      <Sidebar active={view} onSelect={setView} />
      <TopBar runs={runs} selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} title={VIEW_TITLES[view]} />
      <main className="ml-sidebar-width pt-row-height-md min-h-screen p-container-padding">
        <div className="max-w-[1200px] mx-auto w-full">
          {error ? (
            <p className="text-[13px] text-error">{error}</p>
          ) : !selectedRunId ? (
            <p className="text-[13px] text-secondary">No runs found under data/runs/. Run a scenario first.</p>
          ) : view === "fleet_map" ? (
            <FleetMapView runId={selectedRunId} />
          ) : view === "kpis" ? (
            <KpisView runId={selectedRunId} />
          ) : view === "anomaly_timeline" ? (
            <AnomalyTimelineView runId={selectedRunId} />
          ) : (
            <ComingSoonView title={VIEW_TITLES[view]} />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
