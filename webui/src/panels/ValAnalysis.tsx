import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Loader2, RefreshCw, AlertTriangle } from "lucide-react";

interface Props {
  runId: string | null;
  onOpenTrajectory?: (step: string, task: string) => void;
}

interface ValEvent {
  idx: number;
  step: string;
  // Trainer global_step this val belongs to, resolved server-side from the log.
  // The `step` field above is the ROLLOUT counter baked into the dump dir name
  // (rollout 2808 == train step 40), which is meaningless to read off a chart.
  train_step: number | null;
  num_tasks: number;
  solved: number;
  mean_solve_rate: number | null;
  repo_rates: Record<string, number | null>;
  repo_counts: Record<string, number>;
  repo_solved: Record<string, number>;
}
interface ValTask {
  task: string;
  repo: string;
  first: number | null;
  last: number | null;
  delta: number | null;
  solved_events: number;
  appeared_events: number;
}
interface ValData {
  repos: string[];
  events: ValEvent[];
  tasks: ValTask[];
  num_events: number;
  unique_tasks: number;
  dropped_stragglers: number;
  events_histogram: number[];
  first_step: string;
  last_step: string;
  num_exp_dirs: number;
}

function shortRepo(r: string): string {
  const i = r.indexOf("__");
  return i >= 0 ? r.slice(i + 2) : r;
}
function hslRamp(v: number | null): string {
  if (v === null) return "#1e293b";
  return `hsl(${v * 130}, 72%, 50%)`; // 0 red -> green
}
// Label a val event by the trainer step it belongs to; fall back to the raw
// rollout counter from the dir name when the pairing couldn't be resolved.
function stepLabel(ev: ValEvent): string {
  return ev.train_step != null ? String(ev.train_step) : ev.step.replace("step_", "");
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-xl font-semibold ${accent ?? "text-slate-100"}`}>{value}</div>
    </div>
  );
}

// SVG radar comparing first-val vs best-val per-repo solve rate
function Radar({
  repos,
  first,
  best,
}: {
  repos: string[];
  first: Record<string, number | null>;
  best: Record<string, number | null>;
}) {
  const C = 170;
  const R = 120;
  const n = repos.length;
  const pt = (i: number, val: number) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
    return [C + R * val * Math.cos(a), C + R * val * Math.sin(a)];
  };
  const poly = (rates: Record<string, number | null>) =>
    repos.map((r, i) => pt(i, rates[r] ?? 0).join(",")).join(" ");

  return (
    <svg viewBox="0 0 340 340" className="w-full max-w-[420px]">
      {[0.25, 0.5, 0.75, 1].map((g) => (
        <polygon key={g} points={repos.map((_r, i) => pt(i, g).join(",")).join(" ")} fill="none" stroke="#334155" strokeWidth={0.5} />
      ))}
      {repos.map((r, i) => {
        const [x, y] = pt(i, 1);
        const [lx, ly] = pt(i, 1.16);
        const anchor = Math.abs(lx - C) < 8 ? "middle" : lx > C ? "start" : "end";
        const fv = first[r];
        const bv = best[r];
        const pct = (v: number | null | undefined) =>
          v == null ? "—" : `${Math.round(v * 100)}%`;
        return (
          <g key={r}>
            <line x1={C} y1={C} x2={x} y2={y} stroke="#334155" strokeWidth={0.5} />
            <text x={lx} y={ly - 4} fontSize={8.5} fill="#94a3b8" textAnchor={anchor} dominantBaseline="middle">
              {shortRepo(r)}
            </text>
            {/* numeric readout at each vertex: first → best solve rate */}
            <text x={lx} y={ly + 6} fontSize={8} textAnchor={anchor} dominantBaseline="middle">
              <tspan fill="#94a3b8">{pct(fv)}</tspan>
              <tspan fill="#64748b">→</tspan>
              <tspan fill="#10b981" fontWeight="600">{pct(bv)}</tspan>
            </text>
          </g>
        );
      })}
      <polygon points={poly(first)} fill="rgba(148,163,184,0.12)" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 3" />
      <polygon points={poly(best)} fill="rgba(16,185,129,0.18)" stroke="#10b981" strokeWidth={2} />
      {/* value dots on the best polygon so each rate is anchored to its vertex */}
      {repos.map((r, i) => {
        const bv = best[r];
        if (bv == null) return null;
        const [px, py] = pt(i, bv);
        return <circle key={r} cx={px} cy={py} r={2} fill="#10b981" />;
      })}
    </svg>
  );
}

export default function ValAnalysis({ runId, onOpenTrajectory }: Props) {
  const [data, setData] = useState<ValData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [repoFilter, setRepoFilter] = useState<string>("");

  // Which run the in-flight request belongs to; a reply that arrives after you
  // switched runs must not repaint this panel. Cold scans here take tens of
  // seconds, so that window is wide.
  const reqRef = useRef<string | null>(runId);

  const load = useCallback(async () => {
    if (!runId) return;
    const reqRun = runId;
    reqRef.current = runId;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/runs/${reqRun}/val-analysis`);
      const d = await res.json();
      if (reqRef.current !== reqRun) return; // superseded
      if (!res.ok || d.error) throw new Error(d.error || `HTTP ${res.status}`);
      setData(d);
    } catch (e: unknown) {
      if (reqRef.current !== reqRun) return;
      setError(e instanceof Error ? e.message : String(e));
      setData(null);
    } finally {
      if (reqRef.current === reqRun) setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  const first = data?.events[0];
  // "after training" reference = the PEAK eval (highest mean solve), not the
  // last one (which may have regressed).
  const best = useMemo(() => {
    if (!data?.events.length) return undefined;
    return data.events.reduce((a, b) => ((b.mean_solve_rate ?? 0) > (a.mean_solve_rate ?? 0) ? b : a), data.events[0]);
  }, [data]);
  const N = data?.num_events ?? 8;

  const gridTasks = useMemo(() => {
    let ts = data?.tasks ?? [];
    if (repoFilter) ts = ts.filter((t) => t.repo === repoFilter);
    return [...ts].sort((a, b) => a.solved_events - b.solved_events);
  }, [data, repoFilter]);

  // open the most-failing task's trajectory for a capability (weakness analysis)
  const openFailing = useCallback(
    (repo: string) => {
      if (!data) return;
      const t = (data.tasks.filter((x) => x.repo === repo).sort((a, b) => a.solved_events - b.solved_events))[0];
      if (t) onOpenTrajectory?.(data.last_step, t.task);
    },
    [data, onOpenTrajectory],
  );

  const hist = data?.events_histogram ?? [];
  const maxBar = Math.max(1, ...hist);
  const neverSolved = hist[0] ?? 0;
  const alwaysSolved = hist[N] ?? 0;

  return (
    <div className="mt-8 border-t border-slate-800/60 pt-6">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-base font-semibold text-slate-100">Validation Analysis</h3>
        <button onClick={load} disabled={loading || !runId} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 transition-colors disabled:opacity-40">
          {loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          Refresh
        </button>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Val set evaluated periodically (n=1 per task), deduped to the unique set. Per-capability (repo)
        change first-val → best-val (peak), cross-event consistency, and per-task transitions. Click a
        capability to drill into its weakest task's trajectory. Aggregated across {data?.num_exp_dirs ?? "?"} async worker dirs.
      </p>

      {error && <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4 text-xs text-rose-300 font-mono">{error}</div>}
      {loading && !data && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-6 flex items-center justify-center gap-3">
          <Loader2 size={18} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">Scanning val trials (~20s)…</span>
        </div>
      )}

      {data && first && best && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
            <Stat label="Val events" value={String(data.num_events)} />
            <Stat
              label="Unique tasks"
              value={data.unique_tasks.toLocaleString()}
            />
            <Stat
              label={`First → Best solve (peak @${stepLabel(best)})`}
              value={`${((first.mean_solve_rate ?? 0) * 100).toFixed(1)}% → ${((best.mean_solve_rate ?? 0) * 100).toFixed(1)}%`}
              accent={(best.mean_solve_rate ?? 0) >= (first.mean_solve_rate ?? 0) ? "text-emerald-400" : "text-rose-400"}
            />
            <Stat label={`Never solved (0/${N})`} value={String(neverSolved)} accent="text-rose-400" />
            <Stat label={`Always solved (${N}/${N})`} value={String(alwaysSolved)} accent="text-emerald-400" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            {/* Capability radar: first vs best */}
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 flex flex-col items-center">
              <div className="self-start text-xs font-semibold text-slate-200 mb-1">Capability radar (solve rate by repo)</div>
              <div className="self-start text-[10px] text-slate-500 mb-2 flex items-center gap-3">
                <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-dashed border-slate-400" />first val</span>
                <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-emerald-500" />best val (peak)</span>
              </div>
              <Radar repos={data.repos} first={first.repo_rates} best={best.repo_rates} />
            </div>

            {/* Per-capability evolution heatmap: repo × val event, solve rate */}
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 overflow-x-auto">
              <div className="text-xs font-semibold text-slate-200 mb-1">Capability × evaluation (solve rate)</div>
              <p className="text-[10px] text-slate-500 mb-3">
                Rows = repos (capabilities), columns = the {data.num_events} evaluations (oldest →
                newest). Cell = solve rate that eval (red→green). Read each row left→right to see a
                capability improve or regress over training.
              </p>
              <div className="inline-block min-w-full">
                {/* header: event indices */}
                <div className="flex items-center gap-0.5 mb-0.5">
                  <span className="w-20 shrink-0" />
                  {data.events.map((ev) => (
                    <span
                      key={ev.idx}
                      className={`flex-1 text-center text-[9px] font-mono ${ev === best ? "text-emerald-400 font-semibold" : "text-slate-500"}`}
                      title={`train step ${stepLabel(ev)} (${ev.step})`}
                    >
                      {stepLabel(ev)}
                    </span>
                  ))}
                </div>
                {data.repos.map((r) => (
                  <div key={r} className="flex items-center gap-0.5 mb-0.5">
                    <span className="w-20 shrink-0 text-[9px] font-mono text-slate-400 truncate text-right pr-1" title={r}>
                      {shortRepo(r)}
                    </span>
                    {data.events.map((ev) => {
                      const rate = ev.repo_rates[r];
                      return (
                        <div
                          key={ev.idx}
                          className="flex-1 h-5 rounded-[2px] flex items-center justify-center"
                          style={{ background: hslRamp(rate ?? null) }}
                          title={`${shortRepo(r)} @ step ${stepLabel(ev)} (val#${ev.idx + 1}, ${ev.step})\nsolve ${rate == null ? "—" : (rate * 100).toFixed(0) + "%"} · ${ev.repo_solved[r] ?? 0}/${ev.repo_counts[r] ?? 0}`}
                        >
                          <span className="text-[8px] text-black/55 font-medium">
                            {rate == null ? "" : Math.round(rate * 100)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
              <div className="text-[10px] text-slate-500 mt-1.5">val #1 (first) → #{data.num_events} (last) · column {(best?.idx ?? 0) + 1} = peak</div>
            </div>
          </div>

          {/* Per-repo before/after table — click row to filter grid; ⚠ opens weakest trajectory */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 overflow-hidden mb-4">
            <div className="grid grid-cols-[1fr_60px_90px_60px_70px] gap-2 px-4 py-2 text-[10px] uppercase tracking-wider text-slate-500 border-b border-slate-800/60">
              <span>Repo (capability)</span>
              <span className="text-right">tasks</span>
              <span className="text-right">first → best</span>
              <span className="text-right">Δ</span>
              <span className="text-right">weakest</span>
            </div>
            {data.repos.map((r) => {
              const fr = first.repo_rates[r];
              const br = best.repo_rates[r];
              const dl = fr != null && br != null ? br - fr : null;
              return (
                <div key={r} className={`grid grid-cols-[1fr_60px_90px_60px_70px] gap-2 px-4 py-1.5 items-center text-xs border-b border-slate-800/40 hover:bg-slate-800/40 ${repoFilter === r ? "bg-indigo-500/10" : ""}`}>
                  <button onClick={() => setRepoFilter(repoFilter === r ? "" : r)} className="font-mono text-slate-300 truncate text-left">
                    {shortRepo(r)}
                  </button>
                  <span className="text-slate-500 text-right">{best.repo_counts[r] ?? 0}</span>
                  <span className="font-mono text-slate-400 text-right">
                    {fr != null ? (fr * 100).toFixed(0) : "-"}% → {br != null ? (br * 100).toFixed(0) : "-"}%
                  </span>
                  <span className={`font-mono font-semibold text-right ${dl == null ? "text-slate-500" : dl > 0 ? "text-emerald-400" : dl < 0 ? "text-rose-400" : "text-slate-500"}`}>
                    {dl == null ? "—" : `${dl > 0 ? "+" : ""}${(dl * 100).toFixed(0)}`}
                  </span>
                  <button
                    onClick={() => openFailing(r)}
                    title="open the weakest task's trajectory in this repo"
                    className="flex items-center justify-end gap-1 text-[10px] text-amber-400/80 hover:text-amber-300"
                  >
                    <AlertTriangle size={11} /> traj
                  </button>
                </div>
              );
            })}
          </div>

          {/* Cross-event consistency histogram */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 mb-4">
            <div className="text-xs font-semibold text-slate-200 mb-1">Solving consistency — solved in k of {N} val events</div>
            <p className="text-[10px] text-slate-500 mb-3">
              Of the {data.unique_tasks} val tasks, how many pass in 0,1,…,{N} of the evaluations. Left pile (0/{N}) =
              persistently unsolved (the hard ceiling / blocked tasks); right pile ({N}/{N}) = robustly mastered;
              middle = flaky or in-transition (improving or regressing). A one-glance saturation/health view.
            </p>
            <div className="flex items-end gap-1.5 h-40">
              {hist.map((v, k) => (
                <div key={k} className="flex-1 flex flex-col items-center justify-end h-full">
                  <span className="text-[9px] text-slate-400 mb-0.5">{v}</span>
                  <div className="w-full rounded-t" style={{ height: `${(v / maxBar) * 100}%`, minHeight: v > 0 ? "2px" : "0", background: hslRamp(N > 0 ? k / N : 0) }} title={`solved in ${k}/${N} val events: ${v} tasks`} />
                  <span className="text-[9px] text-slate-500 mt-0.5 font-mono">{k}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Per-task val grid */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-semibold text-slate-200">
                Val tasks {repoFilter && <span className="text-indigo-300">· {shortRepo(repoFilter)}</span>}
                <span className="text-[10px] text-slate-500 ml-1">(color = #events solved, red=weak; click → last-val trajectory)</span>
              </div>
              {repoFilter && (
                <button onClick={() => setRepoFilter("")} className="text-[10px] text-slate-500 hover:text-slate-300">clear filter</button>
              )}
            </div>
            <div className="flex flex-wrap gap-1">
              {gridTasks.map((t) => (
                <button
                  key={t.task}
                  onClick={() => onOpenTrajectory?.(data.last_step, t.task)}
                  className="w-3.5 h-3.5 rounded-[3px] hover:ring-1 hover:ring-white/60"
                  style={{ background: hslRamp(N > 0 ? t.solved_events / N : 0) }}
                  title={`${t.task}\nsolved ${t.solved_events}/${t.appeared_events} val events\nfirst ${t.first === null ? "—" : t.first ? "✓" : "✗"} → last ${t.last === null ? "—" : t.last ? "✓" : "✗"}\nclick → last-val trajectory`}
                />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
