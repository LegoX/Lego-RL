import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Loader2, X, Search, Image, FileSpreadsheet } from "lucide-react";
import { useT } from "../i18n";
import type { RunInfo, MetricPoint } from "../types";
import { fetchMetrics } from "../api/client";
import { downloadSvgAsPng, downloadCsv } from "../utils/chartExport";

interface Props {
  runs: RunInfo[];
}

const RUN_COLORS = [
  "#c96a45", "#4a9440", "#e0a01a", "#d03b3b",
  "#7a6ddb", "#199e70", "#b8506e", "#8aa02a",
  "#ef9d6a", "#9085e9", "#3fb98c", "#fab219",
];

const DEFAULT_METRICS = [
  "critic/score/mean",
  "actor/pg_loss",
  "actor/entropy",
  "critic/rewards/mean",
  "response_length/mean",
  "perf/throughput",
  "perf/time_per_step",
  "actor/grad_norm",
];

function formatValue(v: number): string {
  if (v !== v) return "NaN";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  return v.toPrecision(4);
}

function shortName(id: string): string {
  const parts = id.split("-");
  if (parts.length >= 4) {
    const dateIdx = parts.findIndex((p) => /^\d{8}$/.test(p));
    if (dateIdx >= 0) {
      return parts.slice(0, Math.min(dateIdx, 3)).join("-") +
        "-" + parts.slice(dateIdx).join("-");
    }
  }
  return id.length > 30 ? id.slice(0, 15) + "…" + id.slice(-12) : id;
}

interface RunData {
  runId: string;
  metrics: MetricPoint[];
  keys: string[];
  loading: boolean;
}

function CompareChart({
  metricKey,
  runDataList,
  selectedRuns,
  height = 280,
}: {
  metricKey: string;
  runDataList: RunData[];
  selectedRuns: string[];
  height?: number;
}) {
  const { t } = useT();
  const chartRef = useRef<HTMLDivElement>(null);

  const merged = useMemo(() => {
    const active = runDataList.filter(
      (rd) => selectedRuns.includes(rd.runId) && !rd.loading
    );
    if (active.length === 0) return [];

    const stepMap = new Map<number, Record<string, number>>();
    for (const rd of active) {
      const label = shortName(rd.runId);
      for (const p of rd.metrics) {
        if (p[metricKey] === undefined) continue;
        if (!stepMap.has(p.step)) stepMap.set(p.step, { step: p.step });
        stepMap.get(p.step)![label] = p[metricKey];
      }
    }

    return Array.from(stepMap.values()).sort((a, b) => a.step - b.step);
  }, [runDataList, selectedRuns, metricKey]);

  const lineLabels = useMemo(
    () => selectedRuns.map((id) => shortName(id)),
    [selectedRuns]
  );

  const handlePng = useCallback(() => {
    if (!chartRef.current) return;
    const safeName = metricKey.replace(/[/\\:]/g, "_");
    downloadSvgAsPng(chartRef.current, `compare-${safeName}.png`);
  }, [metricKey]);

  const handleCsv = useCallback(() => {
    const safeName = metricKey.replace(/[/\\:]/g, "_");
    downloadCsv(merged, lineLabels, `compare-${safeName}.csv`);
  }, [merged, lineLabels, metricKey]);

  if (merged.length === 0) return null;

  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-slate-300 font-mono">
          {metricKey}
        </h3>
        <div className="flex items-center gap-1">
          <button
            onClick={handlePng}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-slate-500 hover:text-indigo-400 hover:bg-slate-800/60 transition-colors"
            title={t("compare.downloadPNG")}
          >
            <Image size={11} />
            {t("compare.downloadPNG")}
          </button>
          <button
            onClick={handleCsv}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-slate-500 hover:text-emerald-400 hover:bg-slate-800/60 transition-colors"
            title={t("compare.downloadCSV")}
          >
            <FileSpreadsheet size={11} />
            {t("compare.downloadCSV")}
          </button>
        </div>
      </div>
      <div ref={chartRef}>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={merged}>
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
            />
            <Tooltip
              contentStyle={{
                background: "#302a24",
                border: "1px solid #3d352d",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(v: number) => formatValue(v)}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: "#a9a297" }}
              iconType="plainline"
            />
            {lineLabels.map((label, i) => (
              <Line
                key={label}
                type="monotone"
                dataKey={label}
                stroke={RUN_COLORS[i % RUN_COLORS.length]}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function ComparePanel({ runs }: Props) {
  const { t } = useT();
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [runDataMap, setRunDataMap] = useState<Map<string, RunData>>(new Map());
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(DEFAULT_METRICS);
  const [metricSearch, setMetricSearch] = useState("");
  const [runSearch, setRunSearch] = useState("");

  const filteredRuns = useMemo(() => {
    const q = runSearch.trim().toLowerCase();
    const base = q ? runs.filter((r) => r.id.toLowerCase().includes(q)) : runs;
    // keep selected runs out of the scrollable list (they're pinned above)
    return base.filter((r) => !selectedRuns.includes(r.id));
  }, [runs, runSearch, selectedRuns]);

  const allKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const rd of runDataMap.values()) {
      for (const k of rd.keys) keys.add(k);
    }
    return Array.from(keys).sort();
  }, [runDataMap]);

  const filteredKeys = useMemo(() => {
    if (!metricSearch) return allKeys;
    const q = metricSearch.toLowerCase();
    return allKeys.filter((k) => k.toLowerCase().includes(q));
  }, [allKeys, metricSearch]);

  const toggleRun = useCallback((runId: string) => {
    setSelectedRuns((prev) =>
      prev.includes(runId) ? prev.filter((r) => r !== runId) : [...prev, runId]
    );
  }, []);

  const toggleMetric = useCallback((key: string) => {
    setSelectedMetrics((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  }, []);

  // Fetch metrics for newly selected runs
  useEffect(() => {
    for (const runId of selectedRuns) {
      if (runDataMap.has(runId)) continue;
      setRunDataMap((prev) => {
        const next = new Map(prev);
        next.set(runId, { runId, metrics: [], keys: [], loading: true });
        return next;
      });
      fetchMetrics(runId)
        .then((data) => {
          setRunDataMap((prev) => {
            const next = new Map(prev);
            next.set(runId, {
              runId,
              metrics: data.metrics,
              keys: data.available_keys,
              loading: false,
            });
            return next;
          });
        })
        .catch(() => {
          setRunDataMap((prev) => {
            const next = new Map(prev);
            next.set(runId, { runId, metrics: [], keys: [], loading: false });
            return next;
          });
        });
    }
  }, [selectedRuns]);

  const runDataList = useMemo(
    () =>
      selectedRuns
        .map((id) => runDataMap.get(id))
        .filter((rd): rd is RunData => rd != null),
    [selectedRuns, runDataMap]
  );

  const anyLoading = runDataList.some((rd) => rd.loading);

  const activeMetrics = useMemo(
    () =>
      selectedMetrics.filter((key) =>
        runDataList.some(
          (rd) => !rd.loading && rd.metrics.some((p) => p[key] !== undefined)
        )
      ),
    [selectedMetrics, runDataList]
  );

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-100">
          {t("compare.title")}
        </h2>
        <p className="text-xs text-slate-500">{t("compare.desc")}</p>
      </div>

      {/* Run selector */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3 mb-4">
        <div className="flex items-center justify-between gap-2 mb-2">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium whitespace-nowrap">
            {t("compare.selectRuns")} · {selectedRuns.length} selected
          </span>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search
                size={12}
                className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500"
              />
              <input
                type="text"
                value={runSearch}
                onChange={(e) => setRunSearch(e.target.value)}
                placeholder="Search runs…"
                className="bg-slate-800 text-slate-300 text-[11px] border border-slate-700 rounded-md pl-6 pr-2 py-1 w-52 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
            {selectedRuns.length > 0 && (
              <button
                onClick={() => setSelectedRuns([])}
                className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors whitespace-nowrap"
              >
                {t("compare.clear")}
              </button>
            )}
          </div>
        </div>

        {/* Selected runs pinned at top */}
        {selectedRuns.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pb-2 mb-2 border-b border-slate-800/60">
            {selectedRuns.map((id, i) => (
              <button
                key={id}
                onClick={() => toggleRun(id)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono border border-indigo-500/40 bg-indigo-500/10 text-slate-200"
              >
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: RUN_COLORS[i % RUN_COLORS.length] }}
                />
                {shortName(id)}
                <X size={10} className="text-slate-500 ml-0.5" />
              </button>
            ))}
          </div>
        )}

        {/* Scrollable filtered list of unselected runs */}
        <div className="flex flex-wrap gap-1.5 max-h-44 overflow-y-auto">
          {filteredRuns.map((run) => (
            <button
              key={run.id}
              onClick={() => toggleRun(run.id)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono transition-colors border border-slate-700/40 bg-slate-800/40 text-slate-500 hover:text-slate-300 hover:border-slate-600"
            >
              <span
                className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                  run.state === "running"
                    ? "bg-emerald-400"
                    : run.state === "finished"
                      ? "bg-slate-500"
                      : "bg-amber-500"
                }`}
              />
              {shortName(run.id)}
            </button>
          ))}
          {filteredRuns.length === 0 && (
            <span className="text-[11px] text-slate-600 py-1">
              {runSearch ? "no runs match" : "all runs selected"}
            </span>
          )}
        </div>
        <div className="text-[10px] text-slate-600 mt-1.5">
          showing {filteredRuns.length} of {runs.length} runs
        </div>
      </div>

      {/* Metric selector */}
      {selectedRuns.length >= 2 && allKeys.length > 0 && (
        <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3 mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
              {t("compare.selectMetrics")} ({selectedMetrics.length})
            </span>
            <div className="relative">
              <Search
                size={12}
                className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500"
              />
              <input
                type="text"
                value={metricSearch}
                onChange={(e) => setMetricSearch(e.target.value)}
                placeholder="Filter..."
                className="bg-slate-800 text-slate-300 text-[10px] border border-slate-700 rounded-md pl-6 pr-2 py-1 w-40 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
            {filteredKeys.map((key) => {
              const active = selectedMetrics.includes(key);
              return (
                <button
                  key={key}
                  onClick={() => toggleMetric(key)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors border ${
                    active
                      ? "border-indigo-500/40 bg-indigo-500/15 text-indigo-300"
                      : "border-slate-700/30 text-slate-500 hover:text-slate-300 hover:border-slate-600"
                  }`}
                >
                  {key}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Loading indicator */}
      {anyLoading && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-4 mb-4 flex items-center gap-3">
          <Loader2 size={16} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">{t("compare.loading")}</span>
        </div>
      )}

      {/* Placeholder */}
      {selectedRuns.length < 2 && (
        <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-8 text-center text-sm text-slate-500">
          {t("compare.noRuns")}
        </div>
      )}

      {/* Charts */}
      {selectedRuns.length >= 2 && !anyLoading && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {activeMetrics.map((key) => (
            <CompareChart
              key={key}
              metricKey={key}
              runDataList={runDataList}
              selectedRuns={selectedRuns}
            />
          ))}
          {activeMetrics.length === 0 && (
            <div className="col-span-2 rounded-xl bg-slate-900/80 border border-slate-800/60 p-8 text-center text-sm text-slate-500">
              No matching metrics found across selected runs.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
