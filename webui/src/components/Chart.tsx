import { useState, useMemo, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts";
import { Image, FileSpreadsheet } from "lucide-react";
import type { MetricPoint } from "../types";
import { downloadSvgAsPng, downloadCsv } from "../utils/chartExport";

const COLORS = [
  "#6366f1",
  "#10b981",
  "#f59e0b",
  "#f43f5e",
  "#8b5cf6",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
];

interface ChartPanelProps {
  title: string;
  data: MetricPoint[];
  keys: string[];
  colors?: string[];
  yAxisLabel?: string;
  height?: number;
  showArea?: boolean;
  showMinMax?: boolean;
}

function formatValue(v: number): string {
  if (v === undefined || v === null) return "--";
  if (v !== v) return "NaN";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  return v.toPrecision(4);
}

type YAxisMode = "auto" | "full";

function computeAutoFitDomain(
  data: MetricPoint[],
  keys: string[],
  padding = 0.1,
): [number, number] | undefined {
  const vals: number[] = [];
  for (const p of data) {
    for (const k of keys) {
      const v = p[k];
      if (v !== undefined && v === v) vals.push(v);
    }
  }
  if (vals.length === 0) return undefined;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const range = hi - lo || Math.abs(hi) * 0.1 || 1;
  let domainLo = lo - range * padding;
  let domainHi = hi + range * padding;
  if (lo >= 0) domainLo = Math.max(0, domainLo);
  if (hi <= 0) domainHi = Math.min(0, domainHi);
  return [domainLo, domainHi];
}

function ema(values: (number | undefined)[], weight: number): (number | undefined)[] {
  const result: (number | undefined)[] = [];
  let prev: number | undefined;
  for (const v of values) {
    if (v === undefined || v !== v) {
      result.push(prev);
      continue;
    }
    if (prev === undefined) {
      prev = v;
    } else {
      prev = prev * weight + v * (1 - weight);
    }
    result.push(prev);
  }
  return result;
}

function addSmoothedKeys(
  data: MetricPoint[],
  keys: string[],
  weight: number,
): MetricPoint[] {
  const smoothed: Record<string, (number | undefined)[]> = {};
  for (const k of keys) {
    smoothed[k] = ema(
      data.map((p) => p[k]),
      weight,
    );
  }
  return data.map((p, i) => {
    const out: MetricPoint = { ...p };
    for (const k of keys) {
      const sv = smoothed[k][i];
      if (sv !== undefined) out[`__smooth_${k}`] = sv;
    }
    return out;
  });
}

function SmoothingSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] text-slate-500">Smooth</span>
      <input
        type="range"
        min={0}
        max={0.99}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-16 h-1 accent-indigo-500 cursor-pointer"
      />
      <span className="text-[10px] text-slate-400 font-mono w-7 text-right">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function YAxisModeToggle({
  mode,
  onChange,
}: {
  mode: YAxisMode;
  onChange: (m: YAxisMode) => void;
}) {
  return (
    <div className="inline-flex rounded-md bg-slate-800/80 border border-slate-700/50 text-[10px]">
      {(["auto", "full"] as const).map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={`px-2 py-0.5 transition-colors ${
            mode === m
              ? "bg-indigo-500/30 text-indigo-300"
              : "text-slate-500 hover:text-slate-300"
          } ${m === "auto" ? "rounded-l-md" : "rounded-r-md"}`}
        >
          {m === "auto" ? "Auto" : "Full"}
        </button>
      ))}
    </div>
  );
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const items = payload.filter(
    (p: any) => !String(p.dataKey).startsWith("__smooth_"),
  );
  return (
    <div className="rounded-lg bg-slate-800 border border-slate-700 px-3 py-2 shadow-xl text-xs">
      <p className="text-slate-400 mb-1">Step {label}</p>
      {items.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.color }} className="font-mono">
          {p.dataKey}: {formatValue(p.value)}
        </p>
      ))}
    </div>
  );
}

function ExportButtons({
  chartRef,
  title,
  data,
  keys,
}: {
  chartRef: React.RefObject<HTMLDivElement | null>;
  title: string;
  data: MetricPoint[];
  keys: string[];
}) {
  const safeName = title.replace(/[/\\:() ]/g, "_").replace(/_+/g, "_");
  const handlePng = useCallback(() => {
    if (!chartRef.current) return;
    downloadSvgAsPng(chartRef.current, `${safeName}.png`);
  }, [chartRef, safeName]);
  const handleCsv = useCallback(() => {
    downloadCsv(data, keys, `${safeName}.csv`);
  }, [data, keys, safeName]);
  return (
    <div className="flex items-center gap-0.5">
      <button
        onClick={handlePng}
        className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-slate-500 hover:text-indigo-400 hover:bg-slate-800/60 transition-colors"
        title="PNG"
      >
        <Image size={11} />
      </button>
      <button
        onClick={handleCsv}
        className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-slate-500 hover:text-emerald-400 hover:bg-slate-800/60 transition-colors"
        title="CSV"
      >
        <FileSpreadsheet size={11} />
      </button>
    </div>
  );
}

export default function ChartPanel({
  title,
  data,
  keys,
  colors,
  yAxisLabel,
  height = 260,
  showArea = false,
}: ChartPanelProps) {
  const [yMode, setYMode] = useState<YAxisMode>("auto");
  const [smoothing, setSmoothing] = useState(0.6);
  const chartRef = useRef<HTMLDivElement>(null);
  const available = keys.filter((k) => data.some((p) => p[k] !== undefined));

  const filtered = useMemo(
    () => data.filter((p) => available.some((k) => p[k] !== undefined)),
    [data, available],
  );

  const smoothedData = useMemo(
    () => (smoothing > 0 ? addSmoothedKeys(filtered, available, smoothing) : filtered),
    [filtered, available, smoothing],
  );

  const domainKeys = useMemo(
    () =>
      smoothing > 0
        ? available.flatMap((k) => [k, `__smooth_${k}`])
        : available,
    [available, smoothing],
  );

  const domain = useMemo(
    () => (yMode === "auto" ? computeAutoFitDomain(smoothedData, domainKeys) : undefined),
    [smoothedData, domainKeys, yMode],
  );

  if (available.length === 0) {
    return (
      <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">{title}</h3>
        <div className="flex items-center justify-center text-slate-500 text-sm" style={{ height }}>
          No data yet
        </div>
      </div>
    );
  }

  const useComposed = showArea || smoothing > 0;
  const ChartComponent = useComposed ? ComposedChart : LineChart;

  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-slate-300">{title}</h3>
        <div className="flex items-center gap-3">
          <ExportButtons chartRef={chartRef} title={title} data={filtered} keys={available} />
          <SmoothingSlider value={smoothing} onChange={setSmoothing} />
          <YAxisModeToggle mode={yMode} onChange={setYMode} />
        </div>
      </div>
      <div ref={chartRef}>
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent data={smoothedData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis
            dataKey="step"
            stroke="var(--chart-axis)"
            tick={{ fill: "var(--chart-tick)", fontSize: 11 }}
            tickLine={false}
          />
          <YAxis
            stroke="var(--chart-axis)"
            tick={{ fill: "var(--chart-tick)", fontSize: 11 }}
            tickLine={false}
            tickFormatter={formatValue}
            domain={domain}
            label={
              yAxisLabel
                ? {
                    value: yAxisLabel,
                    angle: -90,
                    position: "insideLeft",
                    fill: "var(--chart-tick)",
                    fontSize: 11,
                  }
                : undefined
            }
          />
          <Tooltip content={<CustomTooltip />} />
          {available.length > 1 && (
            <Legend
              wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
              iconType="plainline"
            />
          )}
          {available.map((key, i) => {
            const color = (colors || COLORS)[i % COLORS.length];
            const hasSmooth = smoothing > 0;
            if (showArea && !hasSmooth) {
              return (
                <Area
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={color}
                  fill={color}
                  fillOpacity={0.1}
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                />
              );
            }
            return (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                strokeWidth={hasSmooth ? 1 : 1.5}
                dot={false}
                connectNulls
                opacity={hasSmooth ? 0.25 : 1}
              />
            );
          })}
          {smoothing > 0 &&
            available.map((key, i) => {
              const color = (colors || COLORS)[i % COLORS.length];
              return (
                <Line
                  key={`__smooth_${key}`}
                  type="monotone"
                  dataKey={`__smooth_${key}`}
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                  legendType="none"
                />
              );
            })}
        </ChartComponent>
      </ResponsiveContainer>
      </div>
    </div>
  );
}

interface MultiStatChartProps {
  title: string;
  data: MetricPoint[];
  meanKey: string;
  maxKey: string;
  minKey: string;
  color?: string;
  height?: number;
}

export function MinMaxChart({
  title,
  data,
  meanKey,
  maxKey,
  minKey,
  color = "#6366f1",
  height = 260,
}: MultiStatChartProps) {
  const [yMode, setYMode] = useState<YAxisMode>("auto");
  const [smoothing, setSmoothing] = useState(0.6);
  const chartRef = useRef<HTMLDivElement>(null);
  const available = [meanKey, maxKey, minKey].filter((k) =>
    data.some((p) => p[k] !== undefined),
  );

  const filtered = useMemo(
    () => data.filter((p) => available.some((k) => p[k] !== undefined)),
    [data, available],
  );

  const smoothedData = useMemo(
    () => (smoothing > 0 ? addSmoothedKeys(filtered, [meanKey], smoothing) : filtered),
    [filtered, meanKey, smoothing],
  );

  const domainKeys = useMemo(
    () => (smoothing > 0 ? [meanKey, `__smooth_${meanKey}`] : [meanKey]),
    [meanKey, smoothing],
  );

  const domain = useMemo(
    () => (yMode === "auto" ? computeAutoFitDomain(smoothedData, domainKeys) : undefined),
    [smoothedData, domainKeys, yMode],
  );

  const latestStats = useMemo(() => {
    for (let i = filtered.length - 1; i >= 0; i--) {
      const p = filtered[i];
      if (p[meanKey] !== undefined) {
        return { mean: p[meanKey], min: p[minKey], max: p[maxKey] };
      }
    }
    return null;
  }, [filtered, meanKey, minKey, maxKey]);

  if (available.length === 0) {
    return (
      <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">{title}</h3>
        <div className="flex items-center justify-center text-slate-500 text-sm" style={{ height }}>
          No data yet
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-medium text-slate-300">{title}</h3>
        <div className="flex items-center gap-3">
          <ExportButtons chartRef={chartRef} title={title} data={filtered} keys={available} />
          <SmoothingSlider value={smoothing} onChange={setSmoothing} />
          <YAxisModeToggle mode={yMode} onChange={setYMode} />
        </div>
      </div>
      {latestStats && (
        <div className="flex items-center gap-4 mb-2 text-[11px] font-mono">
          <span className="text-slate-400">
            min: <span className="text-slate-300">{formatValue(latestStats.min)}</span>
          </span>
          <span style={{ color }}>
            mean: <span className="font-semibold">{formatValue(latestStats.mean)}</span>
          </span>
          <span className="text-slate-400">
            max: <span className="text-slate-300">{formatValue(latestStats.max)}</span>
          </span>
        </div>
      )}
      <div ref={chartRef}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={smoothedData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis
            dataKey="step"
            stroke="var(--chart-axis)"
            tick={{ fill: "var(--chart-tick)", fontSize: 11 }}
            tickLine={false}
          />
          <YAxis
            stroke="var(--chart-axis)"
            tick={{ fill: "var(--chart-tick)", fontSize: 11 }}
            tickLine={false}
            tickFormatter={formatValue}
            domain={domain}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey={meanKey}
            stroke={color}
            strokeWidth={smoothing > 0 ? 1 : 2}
            dot={false}
            connectNulls
            opacity={smoothing > 0 ? 0.25 : 1}
          />
          {smoothing > 0 && (
            <Line
              type="monotone"
              dataKey={`__smooth_${meanKey}`}
              stroke={color}
              strokeWidth={2}
              dot={false}
              connectNulls
              legendType="none"
            />
          )}
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
            iconType="plainline"
          />
        </ComposedChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
