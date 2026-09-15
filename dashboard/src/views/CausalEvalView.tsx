import { useEffect, useState } from "react";
import { api, type CausalEvalReport } from "../lib/api";
import { KpiCard } from "../components/KpiCard";

const METRIC_LABELS: Record<string, string> = {
  throughput: "Throughput (orders completed)",
  mean_wait: "Mean Order Wait (ticks)",
  near_miss_total: "Near-Miss Count",
  congestion_ticks: "Congestion Duration (ticks)",
};

function MetricCard({ name, result }: { name: string; result: CausalEvalReport["metrics"][string] }) {
  const significant = result["significant_at_0.05"];
  const improved = significant && (result.lower_is_better ? result.mean_delta < 0 : result.mean_delta > 0);
  const accentColor = !significant
    ? "var(--color-secondary)"
    : improved
      ? "var(--color-primary)"
      : "var(--color-error)";

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[13px] font-semibold text-on-surface">{METRIC_LABELS[name] ?? name}</h3>
        <span
          className="font-mono text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border"
          style={{ color: accentColor, borderColor: accentColor }}
        >
          {!significant ? "not significant" : improved ? "improved" : "worsened"}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <div className="font-mono text-[11px] text-secondary uppercase tracking-widest">Mean Δ (on − off)</div>
          <div className="text-[16px] font-semibold text-on-surface font-mono">{result.mean_delta.toFixed(2)}</div>
        </div>
        <div>
          <div className="font-mono text-[11px] text-secondary uppercase tracking-widest">95% CI</div>
          <div className="text-[13px] text-on-surface font-mono">
            [{result.ci_95_low.toFixed(2)}, {result.ci_95_high.toFixed(2)}]
          </div>
        </div>
        <div>
          <div className="font-mono text-[11px] text-secondary uppercase tracking-widest">p-value</div>
          <div className="text-[13px] text-on-surface font-mono">{result.p_value.toFixed(4)}</div>
        </div>
      </div>
      <p className="text-[12px] text-secondary leading-5">{result.interpretation}</p>
    </div>
  );
}

export function CausalEvalView() {
  const [report, setReport] = useState<CausalEvalReport | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [notRun, setNotRun] = useState(false);

  useEffect(() => {
    api
      .causalEval()
      .then((data) => {
        setReport(data);
        setLoaded(true);
      })
      .catch(() => {
        setNotRun(true);
        setLoaded(true);
      });
  }, []);

  if (!loaded) {
    return <p className="text-secondary text-[13px]">Loading causal evaluation report…</p>;
  }

  if (notRun || !report) {
    return (
      <div className="flex flex-col gap-2">
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Causal Evaluation Report</h2>
        <p className="text-[13px] text-secondary">
          No causal evaluation has been run yet. Run{" "}
          <code className="font-mono text-[12px]">python -m eval.run_causal_eval</code> to generate paired-trial
          results (loop ON vs. OFF, same seeds) for this view.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-[20px] font-semibold text-on-surface leading-7">Causal Evaluation Report</h2>
        <p className="text-[13px] text-secondary mt-1">
          Paired trials (same seed, closed loop ON vs. OFF) — Wilcoxon signed-rank test per metric
        </p>
      </div>

      <div className="grid grid-cols-3 gap-gutter">
        <KpiCard label="Paired Trials" value={`${report.n_pairs}`} />
        {report.scenario && (
          <>
            <KpiCard label="Ticks per Trial" value={`${report.scenario.ticks}`} />
            <KpiCard label="Fleet Size" value={`${report.scenario.robots}`} unit="robots" />
          </>
        )}
      </div>

      <div className="grid grid-cols-2 gap-gutter">
        {Object.entries(report.metrics).map(([name, result]) => (
          <MetricCard key={name} name={name} result={result} />
        ))}
      </div>
    </div>
  );
}
