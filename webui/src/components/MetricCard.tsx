import type { MetricPoint } from "../types";

interface Props {
  label: string;
  metricKey: string;
  data: MetricPoint[];
  format?: "number" | "percent" | "int" | "duration";
  color?: string;
  icon?: React.ReactNode;
}

function fmt(v: number | undefined, format: string): string {
  if (v === undefined || v !== v) return "--";
  if (format === "int") return Math.round(v).toLocaleString();
  if (format === "percent") return `${(v * 100).toFixed(1)}%`;
  if (format === "duration") return `${v.toFixed(1)}s`;
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  if (Math.abs(v) >= 1000) return v.toFixed(1);
  return v.toPrecision(4);
}

function Sparkline({
  data,
  color,
}: {
  data: number[];
  color: string;
}) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 80;
  const h = 24;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  });
  return (
    <svg width={w} height={h} className="mt-1">
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function MetricCard({
  label,
  metricKey,
  data,
  format = "number",
  color = "#6366f1",
  icon,
}: Props) {
  const values = data
    .map((p) => p[metricKey])
    .filter((v): v is number => v !== undefined && v === v);
  const current = values.length > 0 ? values[values.length - 1] : undefined;
  const prev = values.length > 1 ? values[values.length - 2] : undefined;
  const delta =
    current !== undefined && prev !== undefined && prev !== 0
      ? ((current - prev) / Math.abs(prev)) * 100
      : undefined;

  const sparkData = values.slice(-30);

  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 hover:border-slate-700/60 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-400 font-medium truncate">
          {icon && <span className="mr-1.5 inline-block align-middle">{icon}</span>}
          {label}
        </span>
        {delta !== undefined && (
          <span
            className={`text-xs font-mono ${delta >= 0 ? "text-emerald-400" : "text-rose-400"}`}
          >
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(1)}%
          </span>
        )}
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-semibold text-slate-100 font-mono">
          {fmt(current, format)}
        </span>
        <Sparkline data={sparkData} color={color} />
      </div>
    </div>
  );
}
