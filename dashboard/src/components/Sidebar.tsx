import type { ReactNode } from "react";
import { LogIcon, MapIcon, ReportIcon, TimelineIcon } from "./icons";

export type ViewId = "fleet_map" | "kpis" | "anomaly_timeline" | "agent_log" | "causal_eval";

interface NavItem {
  id: ViewId;
  label: string;
  icon: ReactNode;
  comingSoon?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { id: "fleet_map", label: "Fleet Map", icon: <MapIcon /> },
  { id: "kpis", label: "KPIs", icon: <DashboardIcon /> },
  { id: "anomaly_timeline", label: "Anomaly Timeline", icon: <TimelineIcon /> },
  { id: "agent_log", label: "Agent Action Log", icon: <LogIcon />, comingSoon: true },
  { id: "causal_eval", label: "Causal Evaluation Report", icon: <ReportIcon />, comingSoon: true },
];

function DashboardIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <rect x="3" y="3" width="7" height="9" rx="1" />
      <rect x="14" y="3" width="7" height="5" rx="1" />
      <rect x="14" y="12" width="7" height="9" rx="1" />
      <rect x="3" y="16" width="7" height="5" rx="1" />
    </svg>
  );
}

export function Sidebar({ active, onSelect }: { active: ViewId; onSelect: (id: ViewId) => void }) {
  return (
    <nav className="fixed left-0 top-0 h-full w-sidebar-width bg-surface border-r border-outline-variant flex flex-col z-20">
      <div className="p-gutter border-b border-outline-variant">
        <h1 className="text-[16px] font-semibold text-primary leading-6">Fleet Monitor</h1>
        <p className="font-mono text-[12px] text-secondary mt-0.5">Part 1 — Observe &amp; Predict</p>
      </div>
      <ul className="flex-1 py-gutter flex flex-col gap-1 px-1">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          return (
            <li key={item.id}>
              <button
                type="button"
                disabled={item.comingSoon}
                onClick={() => onSelect(item.id)}
                className={[
                  "w-full flex items-center gap-3 px-3 py-2 rounded text-[13px] transition-colors duration-150 text-left",
                  isActive
                    ? "bg-surface-container-highest text-primary font-semibold"
                    : item.comingSoon
                      ? "text-outline cursor-not-allowed"
                      : "text-secondary hover:bg-surface-container-low",
                ].join(" ")}
              >
                {item.icon}
                <span className="flex-1">{item.label}</span>
                {item.comingSoon && (
                  <span className="font-mono text-[10px] text-outline uppercase tracking-wide">Part 2</span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
      <div className="p-gutter border-t border-outline-variant">
        <p className="font-mono text-[11px] text-outline">warehouse-fleet-sim</p>
      </div>
    </nav>
  );
}
