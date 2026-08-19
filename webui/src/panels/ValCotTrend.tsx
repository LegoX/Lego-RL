import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Loader2, RefreshCw } from "lucide-react";

interface Props {
  runId: string | null;
}

interface CotPoint {
  step: string;
  // Trainer global_step this val belongs to; `step` is the rollout counter
  // baked into the dump dir name, which is meaningless to read off an axis.
  train_step: number | null;
  // per-task mean: what a typical task spends on reasoning
  cot_ratio: number | null;
  // char-weighted: what share of ALL emitted text was reasoning
  cot_ratio_weighted: number | null;
  cot_chars: number | null;
  resp_chars: number | null;
  turns: number | null;
  n: number;
  // share of the fixed cohort this event actually contains
  coverage: number;
}

interface CotData {
  points: CotPoint[];
  num_events: number;
  cohort_size: number;
  num_exp_dirs: number;
}

const pct = (v: number | null | undefined, d = 1) =>
  v == null ? "—" : `${(v * 100).toFixed(d)}%`;
const compact = (v: number | null | undefined) =>
  v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v));

function label(p: CotPoint): string {
  return p.train_step != null ? String(p.train_step) : p.step.replace("step_", "");
}

function Stat({ label, value, accent, sub }: { label: string; value: string; accent?: string; sub?: string }) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-xl font-semibold ${accent ?? "text-slate-100"}`}>{value}</div>
      {sub && <div className="text-[10px] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export default function ValCotTrend({ runId }: Props) {
  const [data, setData] = useState<CotData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Which run the in-flight request belongs to. Switching runs while one is
  // still out is normal here — a cold val-cot scan takes ~40s — and without
  // this the late reply repaints the panel with the run you left.
  const reqRef = useRef<string | null>(runId);

  const load = useCallback(async () => {
    if (!runId) return;
    const reqRun = runId;
    reqRef.current = runId;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/runs/${reqRun}/val-cot?sample=120`);
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

  const pts = useMemo(() => (data?.points ?? []).filter((p) => p.cot_ratio != null), [data]);
  const first = pts[0];
  const last = pts[pts.length - 1];
  const delta = first && last && first.cot_ratio != null && last.cot_ratio != null
    ? last.cot_ratio - first.cot_ratio
    : null;

  // Autoscale: CoT share usually moves inside a 10-20pt band, so a fixed 0-100%
  // axis flattens the very trend this panel exists to show. Keep a floor so a
  // flat series doesn't get magnified into noise.
  const { lo, hi } = useMemo(() => {
    const vals = pts.flatMap((p) =>
      [p.cot_ratio, p.cot_ratio_weighted].filter((v): v is number => v != null),
    );
    if (!vals.length) return { lo: 0, hi: 1 };
    let a = Math.min(...vals);
    let b = Math.max(...vals);
    const pad = Math.max(0.05, (b - a) * 0.25);
    a = Math.max(0, a - pad);
    b = Math.min(1, b + pad);
    if (b - a < 0.1) {
      const mid = (a + b) / 2;
      a = Math.max(0, mid - 0.05);
      b = Math.min(1, mid + 0.05);
    }
    return { lo: a, hi: b };
  }, [pts]);

  const W = 720;
  const H = 230;
  const PL = 46;
  const PR = 16;
  const PT = 14;
  const PB = 30;
  const x = (i: number) => PL + (pts.length <= 1 ? (W - PL - PR) / 2 : (i * (W - PL - PR)) / (pts.length - 1));
  const y = (v: number) => PT + (1 - (v - lo) / (hi - lo || 1)) * (H - PT - PB);
  const line = (key: "cot_ratio" | "cot_ratio_weighted") =>
    pts
      .map((p, i) => (p[key] == null ? null : `${x(i)},${y(p[key] as number)}`))
      .filter(Boolean)
      .join(" ");

  const ticks = useMemo(() => {
    const out: number[] = [];
    const span = hi - lo;
    const stepv = span > 0.4 ? 0.1 : span > 0.2 ? 0.05 : 0.02;
    for (let v = Math.ceil(lo / stepv) * stepv; v <= hi + 1e-9; v += stepv) out.push(v);
    return out;
  }, [lo, hi]);

  return (
    <div className="mt-8 border-t border-slate-800/60 pt-6">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-base font-semibold text-slate-100">Val CoT Share</h3>
        <button
          onClick={load}
          disabled={loading || !runId}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 transition-colors disabled:opacity-40"
        >
          {loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          Refresh
        </button>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Chain-of-thought as a share of the response, per val event. The response includes tool-call
        arguments — a coding agent emits its patch there, and scoring CoT against visible text alone
        roughly doubles the number. Measured over a{" "}
        <span className="text-slate-400">fixed cohort of {data?.cohort_size ?? "?"} tasks</span> present
        in most events, so the line moves with the model rather than with which tasks landed in a dump.
      </p>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4 text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}
      {loading && !data && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-6 flex items-center justify-center gap-3">
          <Loader2 size={18} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">Reading val trajectories (~60s cold)…</span>
        </div>
      )}

      {data && pts.length > 0 && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
            <Stat label="First val" value={pct(first?.cot_ratio)} sub={`step ${label(first)}`} />
            <Stat label="Latest val" value={pct(last?.cot_ratio)} sub={`step ${label(last)}`} />
            <Stat
              label="Change"
              value={delta == null ? "—" : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pt`}
              accent={delta == null ? undefined : delta > 0.02 ? "text-amber-400" : delta < -0.02 ? "text-emerald-400" : "text-slate-100"}
              sub={delta != null && Math.abs(delta) > 0.02 ? (delta > 0 ? "more reasoning" : "less reasoning") : "roughly flat"}
            />
            <Stat
              label="Response size"
              value={`${compact(first?.resp_chars)} → ${compact(last?.resp_chars)}`}
              sub="chars/task, CoT + tool args + text"
            />
          </div>

          <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-3 mb-3 overflow-x-auto">
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[520px]">
              {ticks.map((v) => (
                <g key={v}>
                  <line x1={PL} y1={y(v)} x2={W - PR} y2={y(v)} stroke="#302a24" strokeWidth={1} />
                  <text x={PL - 6} y={y(v)} fontSize={9} fill="#8a847a" textAnchor="end" dominantBaseline="middle">
                    {(v * 100).toFixed(0)}%
                  </text>
                </g>
              ))}
              {pts.length > 1 && (
                <polyline points={line("cot_ratio_weighted")} fill="none" stroke="#8a847a" strokeWidth={1.5} strokeDasharray="5 4" />
              )}
              {pts.length > 1 && (
                <polyline points={line("cot_ratio")} fill="none" stroke="#9085e9" strokeWidth={2.5} />
              )}
              {pts.map((p, i) => (
                <g key={p.step}>
                  <circle
                    cx={x(i)}
                    cy={y(p.cot_ratio as number)}
                    r={p.coverage < 0.5 ? 5 : 4}
                    fill={p.coverage < 0.5 ? "#e0a01a" : "#9085e9"}
                    stroke="#1a1714"
                    strokeWidth={1.5}
                  >
                    <title>
                      {`step ${label(p)}  CoT ${pct(p.cot_ratio)} (weighted ${pct(p.cot_ratio_weighted)})\n`}
                      {`${compact(p.cot_chars)} / ${compact(p.resp_chars)} chars · ${p.turns?.toFixed(1) ?? "—"} turns\n`}
                      {`${p.n}/${data.cohort_size} tasks${p.coverage < 0.5 ? " (low coverage — indicative only)" : ""}`}
                    </title>
                  </circle>
                  <text x={x(i)} y={y(p.cot_ratio as number) - 10} fontSize={9} fill="#b3aae8" textAnchor="middle">
                    {(p.cot_ratio! * 100).toFixed(0)}
                  </text>
                  <text x={x(i)} y={H - PB + 14} fontSize={9} fill="#8a847a" textAnchor="middle">
                    {label(p)}
                  </text>
                </g>
              ))}
              <text x={PL} y={H - 4} fontSize={9} fill="#514740">
                trainer step
              </text>
            </svg>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-[10px] text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-0.5 bg-violet-400" /> mean per task
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-0.5 border-t border-dashed border-slate-500" /> character-weighted
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full bg-amber-500" /> this val covered &lt; 50% of the cohort
            </span>
            <span>{data.num_events} val events across {data.num_exp_dirs} worker directories</span>
          </div>
        </>
      )}

      {data && pts.length === 0 && !loading && (
        <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-6 text-center text-sm text-slate-500">
          No comparable val events yet — this needs at least one val dump whose trajectories contain <code className="text-slate-400">proxy_trajectory.json</code>
        </div>
      )}
    </div>
  );
}
