import { Fragment, useEffect, useMemo, useState } from "react";
import { api, type AgentDecisionRow, type WhitelistedAction } from "../lib/api";
import { KpiCard } from "../components/KpiCard";

const ACTION_COLORS: Record<WhitelistedAction, string> = {
  THROTTLE_ZONE_TRAFFIC: "var(--color-primary)",
  REPLAN_ROUTE: "var(--color-secondary)",
  REASSIGN_TASK: "var(--color-secondary)",
  NO_ACTION: "var(--color-outline)",
};

function ActionBadge({ action }: { action: WhitelistedAction }) {
  return (
    <span
      className="font-mono text-[11px] uppercase tracking-wide px-2 py-0.5 rounded border"
      style={{ color: ACTION_COLORS[action], borderColor: ACTION_COLORS[action] }}
    >
      {action}
    </span>
  );
}

export function AgentActionLogView({ runId }: { runId: string }) {
  const [decisions, setDecisions] = useState<AgentDecisionRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [expandedTick, setExpandedTick] = useState<number | null>(null);

  useEffect(() => {
    setLoaded(false);
    api.agentDecisions(runId).then((data) => {
      setDecisions(data);
      setLoaded(true);
    });
  }, [runId]);

  const stats = useMemo(() => {
    if (decisions.length === 0) return null;
    const executed = decisions.filter((d) => d.executed).length;
    const byAction = decisions.reduce<Record<string, number>>((acc, d) => {
      acc[d.action] = (acc[d.action] ?? 0) + 1;
      return acc;
    }, {});
    return { total: decisions.length, executed, byAction };
  }, [decisions]);

  if (!loaded) {
    return <p className="text-secondary text-[13px]">Loading agent decisions…</p>;
  }

  if (decisions.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Agent Action Log</h2>
        <p className="text-[13px] text-secondary">
          This run has no closed-loop decisions logged — it predates M8, or was run without{" "}
          <code className="font-mono text-[12px]">agent.run_agent_demo</code>. Select a run produced by that
          script (e.g. <code className="font-mono text-[12px]">agent_closed_loop_demo</code>) to see the
          trigger → retrieve → decide → act cycle.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Agent Action Log</h2>
        <p className="text-[13px] text-secondary mt-1">
          Every forecaster/backstop-triggered tick's retrieve → decide → act cycle — run: {runId}
        </p>
      </div>

      {stats && (
        <div className="grid grid-cols-3 gap-gutter">
          <KpiCard label="Triggered Cycles" value={`${stats.total}`} />
          <KpiCard label="Executed" value={`${stats.executed}`} unit={`/ ${stats.total}`} />
          <KpiCard
            label="Throttle Actions"
            value={`${stats.byAction["THROTTLE_ZONE_TRAFFIC"] ?? 0}`}
            accent={(stats.byAction["THROTTLE_ZONE_TRAFFIC"] ?? 0) > 0}
          />
        </div>
      )}

      <div className="bg-surface-container-lowest border border-outline-variant rounded overflow-x-auto">
        <table className="w-full text-left text-[12px]">
          <thead>
            <tr className="border-b border-outline-variant text-secondary font-mono text-[11px] uppercase tracking-wide">
              <th className="px-3 py-2 font-medium">Tick</th>
              <th className="px-3 py-2 font-medium">Anomaly</th>
              <th className="px-3 py-2 font-medium">Severity</th>
              <th className="px-3 py-2 font-medium">Confidence</th>
              <th className="px-3 py-2 font-medium">Retrieved SOP</th>
              <th className="px-3 py-2 font-medium">Action</th>
              <th className="px-3 py-2 font-medium">Target</th>
              <th className="px-3 py-2 font-medium">Executed</th>
            </tr>
          </thead>
          <tbody>
            {decisions.map((d) => (
              <Fragment key={d.tick}>
                <tr
                  className="border-b border-outline-variant/50 hover:bg-surface-container-low cursor-pointer"
                  onClick={() => setExpandedTick(expandedTick === d.tick ? null : d.tick)}
                >
                  <td className="px-3 py-2 font-mono text-on-surface">{d.tick}</td>
                  <td className="px-3 py-2 text-on-surface">{d.anomaly_type.replace("_", " ")}</td>
                  <td className="px-3 py-2 text-secondary">{d.severity}</td>
                  <td className="px-3 py-2 font-mono text-secondary">
                    {d.confidence != null ? d.confidence.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-2 font-mono text-secondary">{d.sop_doc_id ?? "—"}</td>
                  <td className="px-3 py-2">
                    <ActionBadge action={d.action} />
                  </td>
                  <td className="px-3 py-2 font-mono text-secondary">
                    {d.target_id ? `${d.target_type}:${d.target_id}` : "—"}
                  </td>
                  <td className="px-3 py-2">
                    <span className={d.executed ? "text-primary" : "text-error"}>{d.executed ? "yes" : "no"}</span>
                  </td>
                </tr>
                {expandedTick === d.tick && (
                  <tr className="border-b border-outline-variant/50 bg-surface-container-low">
                    <td colSpan={8} className="px-3 py-3">
                      <p className="text-[12px] text-on-surface mb-1">
                        <span className="text-secondary">Reason: </span>
                        {d.reason ?? "—"}
                      </p>
                      {d.raw_llm_output && (
                        <pre className="font-mono text-[11px] text-secondary whitespace-pre-wrap break-all">
                          {d.raw_llm_output}
                        </pre>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
