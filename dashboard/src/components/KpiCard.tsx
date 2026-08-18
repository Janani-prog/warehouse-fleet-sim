interface Props {
  label: string;
  value: string;
  unit?: string;
  accent?: boolean;
}

export function KpiCard({ label, value, unit, accent }: Props) {
  return (
    <div className="bg-surface-container-lowest border border-outline-variant p-4 flex flex-col gap-1 rounded">
      <span className="font-mono text-[11px] text-secondary uppercase tracking-widest">{label}</span>
      <div className="flex items-baseline gap-1.5">
        <span className="text-[20px] font-semibold text-on-surface leading-7">{value}</span>
        {unit && (
          <span className={`font-mono text-[12px] ${accent ? "text-error" : "text-secondary"}`}>{unit}</span>
        )}
      </div>
    </div>
  );
}
