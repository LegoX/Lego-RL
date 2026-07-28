import { useState, useMemo } from "react";
import { Search, Plus, X, TrendingUp } from "lucide-react";
import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

const CHART_COLORS = [
  "#6366f1",
  "#10b981",
  "#f59e0b",
  "#f43f5e",
  "#8b5cf6",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
];

interface Props {
  data: MetricPoint[];
  availableKeys: string[];
}

interface CustomChart {
  id: number;
  title: string;
  keys: string[];
}

export default function ExplorerPanel({ data, availableKeys }: Props) {
  const [charts, setCharts] = useState<CustomChart[]>([]);
  const [search, setSearch] = useState("");
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);

  const grouped = useMemo(() => {
    const groups: Record<string, string[]> = {};
    for (const key of availableKeys) {
      const parts = key.split("/");
      const group = parts.length > 1 ? parts[0] : "other";
      if (!groups[group]) groups[group] = [];
      groups[group].push(key);
    }
    return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b));
  }, [availableKeys]);

  const filteredGroups = useMemo(() => {
    if (!search) return grouped;
    const q = search.toLowerCase();
    return grouped
      .map(([group, keys]) => [group, keys.filter((k) => k.toLowerCase().includes(q))] as [string, string[]])
      .filter(([, keys]) => keys.length > 0);
  }, [grouped, search]);

  const addChart = (keys: string[]) => {
    const id = Date.now();
    const title = keys.length === 1
      ? keys[0].split("/").pop() || keys[0]
      : `Custom Chart ${charts.length + 1}`;
    setCharts((prev) => [...prev, { id, title, keys }]);
  };

  const removeChart = (id: number) => {
    setCharts((prev) => prev.filter((c) => c.id !== id));
  };

  const [pendingKeys, setPendingKeys] = useState<string[]>([]);

  const togglePending = (key: string) => {
    setPendingKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const commitPending = () => {
    if (pendingKeys.length > 0) {
      addChart(pendingKeys);
      setPendingKeys([]);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            Metric Explorer
          </h2>
          <p className="text-xs text-slate-500">
            {availableKeys.length} metrics available &middot; select metrics to
            create custom charts
          </p>
        </div>
        {pendingKeys.length > 0 && (
          <button
            onClick={commitPending}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500 text-white text-xs hover:bg-indigo-400 transition-colors"
          >
            <Plus size={12} />
            Plot {pendingKeys.length} metric{pendingKeys.length > 1 ? "s" : ""}
          </button>
        )}
      </div>

      {/* Metric browser */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 mb-6">
        <div className="relative mb-3">
          <Search
            size={14}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
          />
          <input
            type="text"
            placeholder="Search metrics..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-800 text-slate-300 text-xs border border-slate-700 rounded-lg pl-8 pr-3 py-2 focus:outline-none focus:border-indigo-500"
          />
        </div>

        {pendingKeys.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {pendingKeys.map((key) => (
              <span
                key={key}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 text-[10px] border border-indigo-500/30"
              >
                {key.split("/").pop()}
                <button onClick={() => togglePending(key)}>
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="max-h-64 overflow-y-auto space-y-1">
          {filteredGroups.map(([group, keys]) => (
            <div key={group}>
              <button
                onClick={() =>
                  setExpandedGroup(expandedGroup === group ? null : group)
                }
                className="w-full text-left px-2 py-1.5 rounded text-xs font-medium text-slate-300 hover:bg-slate-800/60 flex items-center justify-between"
              >
                <span>
                  {group}{" "}
                  <span className="text-slate-500 font-normal">
                    ({keys.length})
                  </span>
                </span>
                <span className="text-slate-600 text-[10px]">
                  {expandedGroup === group ? "collapse" : "expand"}
                </span>
              </button>
              {expandedGroup === group && (
                <div className="ml-3 space-y-0.5 mb-1">
                  {keys.map((key) => {
                    const lastVal = data
                      .slice()
                      .reverse()
                      .find((p) => p[key] !== undefined);
                    const selected = pendingKeys.includes(key);
                    return (
                      <button
                        key={key}
                        onClick={() => togglePending(key)}
                        className={`w-full text-left px-2 py-1 rounded text-xs flex items-center justify-between gap-2 transition-colors ${
                          selected
                            ? "bg-indigo-500/15 text-indigo-300 border border-indigo-500/20"
                            : "text-slate-400 hover:bg-slate-800/40 hover:text-slate-300 border border-transparent"
                        }`}
                      >
                        <span className="font-mono truncate">{key}</span>
                        <span className="flex items-center gap-2 flex-shrink-0">
                          {lastVal && (
                            <span className="text-slate-500 font-mono text-[10px]">
                              {typeof lastVal[key] === "number"
                                ? lastVal[key].toPrecision(4)
                                : "--"}
                            </span>
                          )}
                          <TrendingUp
                            size={11}
                            className="text-slate-600 cursor-pointer hover:text-indigo-400"
                            onClick={(e) => {
                              e.stopPropagation();
                              addChart([key]);
                            }}
                          />
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
          {filteredGroups.length === 0 && (
            <p className="text-xs text-slate-500 text-center py-4">
              {availableKeys.length === 0
                ? "No metrics available yet"
                : "No metrics match your search"}
            </p>
          )}
        </div>
      </div>

      {/* Custom charts */}
      {charts.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {charts.map((chart) => (
            <div key={chart.id} className="relative group">
              <button
                onClick={() => removeChart(chart.id)}
                className="absolute top-3 right-3 z-10 p-1 rounded bg-slate-800/80 text-slate-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X size={12} />
              </button>
              <ChartPanel
                title={chart.title}
                data={data}
                keys={chart.keys}
                colors={chart.keys.map(
                  (_, i) => CHART_COLORS[i % CHART_COLORS.length],
                )}
                showArea={chart.keys.length === 1}
              />
            </div>
          ))}
        </div>
      )}

      {charts.length === 0 && (
        <div className="rounded-xl border border-dashed border-slate-700 p-8 text-center">
          <TrendingUp size={24} className="mx-auto text-slate-600 mb-2" />
          <p className="text-sm text-slate-500">
            Select metrics above and click &quot;Plot&quot; to create custom
            charts, or click the{" "}
            <TrendingUp size={11} className="inline text-slate-500" /> icon
            next to any metric for a quick single-metric chart.
          </p>
        </div>
      )}
    </div>
  );
}
