// Minimal inline SVG icon set - 16px line icons, matching FRONTEND_SPEC's
// "simple line icon" sidebar guidance without depending on an external
// icon font/CDN (keeps the dashboard fully self-contained and offline-safe).

type IconProps = { className?: string };

const base = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function MapIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21" />
      <line x1="9" y1="3" x2="9" y2="18" />
      <line x1="15" y1="6" x2="15" y2="21" />
    </svg>
  );
}

export function TimelineIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <polyline points="3 17 9 9 13 13 21 5" />
      <circle cx="9" cy="9" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="13" cy="13" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="21" cy="5" r="1.2" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function LogIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4 4h13l3 3v13H4z" />
      <line x1="8" y1="10" x2="16" y2="10" />
      <line x1="8" y1="14" x2="16" y2="14" />
      <line x1="8" y1="18" x2="12" y2="18" />
    </svg>
  );
}

export function ReportIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <rect x="4" y="3" width="16" height="18" rx="1" />
      <line x1="8" y1="8" x2="16" y2="8" />
      <path d="M8 14l2.5 2.5L16 12" />
    </svg>
  );
}

export function DotIcon({ className }: IconProps) {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" className={className}>
      <circle cx="5" cy="5" r="5" fill="currentColor" />
    </svg>
  );
}

export function PlayIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <polygon points="6 4 20 12 6 20" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function PauseIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <rect x="6" y="4" width="4" height="16" fill="currentColor" stroke="none" />
      <rect x="14" y="4" width="4" height="16" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function BoltIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <polygon points="13 2 4 14 11 14 9 22 20 9 13 9" />
    </svg>
  );
}
