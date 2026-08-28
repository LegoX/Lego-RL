import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { MetricPoint } from "../types";

interface Props {
  runId: string | null;
  data?: MetricPoint[];
}

// ---------------------------------------------------------------------------
// Per-step termination-reason breakdown.
//
// Each rollout is tagged with exactly one `termination_reason` (taxonomy below)
// by the agent loop; the trajectory filter logs per-step COUNTS of each reason
// as `trajectory_filter/reason/<reason>` (plus invalid_count/_ratio). Those keys
// are already in the parsed per-step metrics, so this is a pure render of data
// the run already emits — no trial-dir scan, no backend change.
//
// `kept` reasons carry real learning signal and stay in the loss; `dropped`
// reasons (the default drop-set: timeout, env_setup_failed) are env noise and
// are neutralized out of training. `unknown` only appears if a non-taxonomy
// reason string leaks through.
// ---------------------------------------------------------------------------
const REASONS = [
  { key: "agent_completed", color: "#4a9440", group: "kept" as const },
  { key: "overlong", color: "#7a6ddb", group: "kept" as const },
  { key: "max_turns_reached", color: "#199e70", group: "kept" as const },
  { key: "timeout", color: "#e0a01a", group: "dropped" as const },
  { key: "env_setup_failed", color: "#d03b3b", group: "dropped" as const },
  { key: "unknown", color: "#8a847a", group: "other" as const },
] as const;

const REASON_KEY = (k: string) => `trajectory_filter/reason/${k}`;

function TerminationBreakdown({ data }: { data: MetricPoint[] }) {
  const [mode, setMode] = useState<"count" | "frac">("count");

  // Only the reasons that actually appear in THIS run — this is also the
  // empirical answer to "which termination reasons does this task hit?".
  const present = useMemo(
    () =>
      REASONS.filter((r) =>
        data.some((p) => p[REASON_KEY(r.key)] !== undefined),
      ),
    [data],
  );

  // One row per training step: per-reason value (count or within-step fraction)
  // plus the step total. Steps with no reason keys (e.g. pre-filter) are skipped.
  const rows = useMemo(() => {
    const out: Record<string, number>[] = [];
    for (const p of data) {
      let total = 0;
      const vals: Record<string, number> = {};
      for (const r of present) {
        const v = p[REASON_KEY(r.key)];
        if (typeof v === "number" && !Number.isNaN(v)) {
          vals[r.key] = v;
          total += v;
        }
      }
      if (total <= 0) continue;
      const row: Record<string, number> = { step: p.step, __total: total };
      for (const r of present) {
        const c = vals[r.key] ?? 0;
        row[r.key] = mode === "frac" ? c / total : c;
      }
      out.push(row);
    }
    return out;
  }, [data, present, mode]);

  // Run-wide totals per reason (answers "how often does each reason fire").
  const agg = useMemo(() => {
    const sums: Record<string, number> = {};
    let grand = 0;
    for (const p of data) {
      for (const r of present) {
        const v = p[REASON_KEY(r.key)];
        if (typeof v === "number" && !Number.isNaN(v)) {
          sums[r.key] = (sums[r.key] ?? 0) + v;
          grand += v;
        }
      }
    }
    return { sums, grand };
  }, [data, present]);

  if (present.length === 0) {
    return (
      <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-4 mb-6 text-xs text-slate-500">
        No per-step termination-reason metrics for this run yet
        (<code>trajectory_filter/reason/*</code> appears once the first training
        step is logged; snapshots and pre-filter runs won't have it).
      </div>
    );
  }

  const droppedTot = present
    .filter((r) => r.group === "dropped")
    .reduce((s, r) => s + (agg.sums[r.key] ?? 0), 0);
  const droppedShare = agg.grand > 0 ? droppedTot / agg.grand : 0;
  const pct = (x: number) => (agg.grand > 0 ? ((x / agg.grand) * 100).toFixed(1) + "%" : "0%");

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-slate-100">
          Termination Reasons per Step
        </h2>
        <div className="inline-flex rounded-lg border border-slate-700 overflow-hidden text-xs">
          <button
            onClick={() => setMode("count")}
            className={`px-3 py-1.5 transition-colors ${
              mode === "count"
                ? "bg-indigo-500/20 text-indigo-300"
                : "text-slate-400 hover:bg-slate-800"
            }`}
          >
            Counts
          </button>
          <button
            onClick={() => setMode("frac")}
            className={`px-3 py-1.5 transition-colors border-l border-slate-700 ${
              mode === "frac"
                ? "bg-indigo-500/20 text-indigo-300"
                : "text-slate-400 hover:bg-slate-800"
            }`}
          >
            100%
          </button>
        </div>
      </div>
      <p className="text-xs text-slate-500 mb-3">
        How each step's rollouts ended.{" "}
        <span className="text-emerald-300">Kept</span> (agent_completed, overlong,
        max_turns_reached) carry learning signal;{" "}
        <span className="text-rose-300">dropped</span> (timeout, env_setup_failed)
        are environment noise, neutralized out of the loss. A rising dropped band =
        env trouble, not the policy.
      </p>

      {/* run-wide per-reason totals */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
        {present.map((r) => (
          <Stat
            key={r.key}
            label={r.key}
            value={pct(agg.sums[r.key] ?? 0)}
            sub={`${(agg.sums[r.key] ?? 0).toLocaleString()} · ${r.group}`}
            accent={
              r.group === "dropped"
                ? "text-rose-400"
                : r.group === "kept"
                  ? "text-emerald-400"
                  : "text-slate-300"
            }
          />
        ))}
      </div>
      <p className="text-[10px] text-slate-500 mb-3">
        Run-wide share of {agg.grand.toLocaleString()} tagged rollouts.{" "}
        <span className="text-rose-400">Dropped (env noise)</span> ={" "}
        <span className="text-rose-400">{(droppedShare * 100).toFixed(1)}%</span>.
      </p>

      <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#302a24" />
            <XAxis
              dataKey="step"
              tick={{ fill: "#8a847a", fontSize: 11 }}
              stroke="#3d352d"
            />
            <YAxis
              tick={{ fill: "#8a847a", fontSize: 11 }}
              stroke="#3d352d"
              domain={mode === "frac" ? [0, 1] : [0, "auto"]}
              tickFormatter={(v: number) =>
                mode === "frac" ? `${Math.round(v * 100)}%` : `${v}`
              }
            />
            <Tooltip
              contentStyle={{
                background: "#1a1714",
                border: "1px solid #302a24",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "#ddd7cd" }}
              formatter={(v: number, name: string) => [
                mode === "frac" ? `${(v * 100).toFixed(1)}%` : v.toLocaleString(),
                name,
              ]}
              labelFormatter={(s) => `step ${s}`}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {present.map((r) => (
              <Area
                key={r.key}
                type="monotone"
                dataKey={r.key}
                name={r.key}
                stackId="1"
                stroke={r.color}
                fill={r.color}
                fillOpacity={0.7}
                isAnimationActive={false}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
        <div className="text-center text-[10px] text-slate-500 mt-1">
          training step → {mode === "frac" ? "share" : "count"} of rollouts by
          termination reason
        </div>
      </div>
    </div>
  );
}

interface DistData {
  expected_n: number;
  histogram: number[]; // index = #correct, value = #prompt-groups
  train_groups: number;
  val_like_groups: number;
  other_groups: number;
  all_correct: number;
  all_wrong: number;
  no_signal: number;
  mean_solve_rate: number | null;
  num_steps_scanned: number;
  truncated: boolean;
  num_exp_dirs: number;
}

function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={`text-xl font-semibold ${accent ?? "text-slate-100"}`}>
        {value}
      </div>
      {sub && <div className="text-[10px] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

// one distinct colour per correct-count category (0..n), ordered red→green
// so all n+1 buckets are visually separable.
function barColor(k: number, n: number): string {
  if (n <= 0) return "hsl(140,65%,48%)";
  const hue = (k / n) * 130; // 0 = red (all wrong) … 130 = green (all right)
  return `hsl(${hue}, 72%, 52%)`;
}

interface Bucket {
  idx: number;
  count: number;
  step_start: string;
  step_end: string;
  histogram: number[];
  frac: number[];
  no_signal_frac: number;
  mean_solve_rate: number;
}
interface SeriesData {
  expected_n: number;
  buckets: Bucket[];
  train_groups: number;
}

export default function RolloutDistPanel({ runId, data: metrics }: Props) {
  const [n, setN] = useState(8);
  // groups-per-bar for the evolution chart. Default 64 = a typical train_batch_size,
  // so each bar ≈ one training step's worth of prompt-groups (the on-disk step_NNNN
  // is a sample index, NOT the training step — see note under the chart).
  const [groupsPerBar, setGroupsPerBar] = useState(64);
  const [data, setData] = useState<DistData | null>(null);
  const [series, setSeries] = useState<SeriesData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // A snapshot is a single training step (its prompt-groups are materialized as
  // separate step_NNNN dirs), so there is no "over training" time axis — the
  // evolution chart would be misleading. Skip it for snapshots.
  const isSnapshot = !!runId?.startsWith("snap__");

  // Which run the in-flight request belongs to; a reply for the run you just
  // left must not repaint this panel.
  const reqRef = useRef<string | null>(runId);

  const load = useCallback(async () => {
    if (!runId) return;
    const reqRun = runId;
    reqRef.current = runId;
    setLoading(true);
    setError(null);
    try {
      // Normal runs: fetch dist + evolution series in parallel (original
      // behavior). Snapshots are a single step with no over-training axis, so
      // skip the series entirely.
      const [r1, r2] = await Promise.all([
        fetch(`/api/runs/${reqRun}/rollout-dist?n=${n}`),
        isSnapshot
          ? Promise.resolve(null)
          : fetch(
              `/api/runs/${reqRun}/rollout-dist-series?n=${n}&bucket_size=${groupsPerBar}`,
            ),
      ]);
      const d = await r1.json();
      if (reqRef.current !== reqRun) return; // superseded
      if (!r1.ok || d.error) throw new Error(d.error || `HTTP ${r1.status}`);
      setData(d);
      if (r2) {
        const sd = await r2.json();
        if (reqRef.current !== reqRun) return;
        setSeries(r2.ok && !sd.error ? sd : null);
      } else {
        setSeries(null);
      }
    } catch (e: unknown) {
      if (reqRef.current !== reqRun) return;
      setError(e instanceof Error ? e.message : String(e));
      setData(null);
      setSeries(null);
    } finally {
      if (reqRef.current === reqRun) setLoading(false);
    }
  }, [runId, n, isSnapshot, groupsPerBar]);

  useEffect(() => {
    load();
  }, [load]);

  const hist = data?.histogram ?? [];
  const maxBar = Math.max(1, ...hist);
  const total = data?.train_groups ?? 0;
  const allWrong = data?.all_wrong ?? 0;
  const allRight = data?.all_correct ?? 0;
  const mixed = Math.max(0, total - allWrong - allRight); // 1..n-1 correct
  const pct = (x: number) => (total > 0 ? ((x / total) * 100).toFixed(1) + "%" : "0%");

  // The percentages above are the shape; this is the number that actually sets
  // the gradient's signal-to-noise. Nominal prompts/step is read off the last
  // step that logged termination reasons (tagged rollouts / n) rather than the
  // config, so a resumed or retopologized run reports what it really ran.
  const promptsPerStep = useMemo(() => {
    for (let i = (metrics?.length ?? 0) - 1; i >= 0; i--) {
      let tagged = 0;
      for (const [k, v] of Object.entries(metrics![i])) {
        if (k.startsWith("trajectory_filter/reason/") && typeof v === "number" && !Number.isNaN(v)) {
          tagged += v;
        }
      }
      if (tagged > 0) return tagged / n;
    }
    return null;
  }, [metrics, n]);
  const effShare = total > 0 ? mixed / total : 0;

  return (
    <div>
      {/* Per-step termination-reason breakdown (from the run's own metrics —
          independent of the trial-dir scan below). */}
      <TerminationBreakdown data={metrics ?? []} />

      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-slate-100">
          Rollout Correctness Distribution
        </h2>
        <div className="flex items-center gap-2">
          <label className="text-[10px] text-slate-500 uppercase tracking-wider">
            rollouts / prompt
          </label>
          <input
            type="number"
            min={1}
            max={64}
            value={n}
            onChange={(e) => setN(Math.max(1, Math.min(64, +e.target.value || 1)))}
            className="w-16 bg-slate-800 text-slate-200 text-xs border border-slate-700 rounded-lg px-2 py-1 focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={load}
            disabled={loading || !runId}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 transition-colors disabled:opacity-40"
          >
            {loading ? (
              <Loader2 size={13} className="animate-spin" />
            ) : (
              <RefreshCw size={13} />
            )}
            Refresh
          </button>
        </div>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        For each prompt, how many of its {n} rollouts were correct (reward &gt; 0).
        Groups at 0/{n} or {n}/{n} give zero GRPO advantage (no learning signal).
        {isSnapshot
          ? " This is a single imported step — one bar = the whole step's prompt-groups (no over-training view)."
          : ` Grouped by (rollout-batch, task) and aggregated across ${data?.num_exp_dirs ?? "?"} async worker dirs.`}
      </p>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4 text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}

      {loading && !data && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-6 flex items-center justify-center gap-3">
          <Loader2 size={18} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">
            Scanning trial dirs (networked FS, may take ~20s)…
          </span>
        </div>
      )}

      {data && (
        <>
          {/* group breakdown: all-wrong + mixed + all-right = 100% of prompt groups */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-2">
            <Stat
              label="Effective groups / step"
              value={
                promptsPerStep
                  ? (promptsPerStep * effShare).toFixed(1)
                  : (effShare * 100).toFixed(1) + "%"
              }
              sub={
                promptsPerStep
                  ? `of ${promptsPerStep.toFixed(0)} nominal · ${(effShare * 100).toFixed(1)}% carry gradient`
                  : "nominal batch unknown — no reason metrics"
              }
              accent="text-indigo-300"
            />
            <Stat
              label="Prompt groups"
              value={total.toLocaleString()}
              sub={`${data.num_steps_scanned} batches · val-like ${data.val_like_groups.toLocaleString()}`}
            />
            <Stat
              label="Mean solve rate"
              value={
                data.mean_solve_rate != null
                  ? (data.mean_solve_rate * 100).toFixed(1) + "%"
                  : "—"
              }
              sub={`avg of (correct/${n}) per prompt`}
              accent="text-emerald-400"
            />
            <Stat
              label={`All-wrong (0/${n})`}
              value={pct(allWrong)}
              sub={`${allWrong.toLocaleString()} groups · no signal`}
              accent="text-rose-400"
            />
            <Stat
              label={`Mixed (1–${n - 1})`}
              value={pct(mixed)}
              sub={`${mixed.toLocaleString()} groups · has gradient`}
              accent="text-emerald-400"
            />
            <Stat
              label={`All-right (${n}/${n})`}
              value={pct(allRight)}
              sub={`${allRight.toLocaleString()} groups · no signal`}
              accent="text-slate-300"
            />
          </div>
          <p className="text-[10px] text-slate-500 mb-5">
            All-wrong + Mixed + All-right = 100% of prompt groups. <span className="text-amber-400">No-signal</span> ={" "}
            all-wrong + all-right = <span className="text-amber-400">{pct(allWrong + allRight)}</span> (zero GRPO advantage).
            "Mean solve rate" is a separate axis — the average correctness across rollouts, not a share of groups.
          </p>

          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-5">
            <div className="flex items-end gap-2 h-64">
              {hist.map((v, k) => (
                <div
                  key={k}
                  className="flex-1 flex flex-col items-center justify-end h-full"
                >
                  <span className="text-[10px] text-slate-400 mb-1">{v}</span>
                  <div
                    className="w-full rounded-t transition-all"
                    style={{
                      height: `${(v / maxBar) * 100}%`,
                      minHeight: v > 0 ? "2px" : "0",
                      background: barColor(k, n),
                    }}
                    title={`${k}/${n} correct: ${v} prompt-groups`}
                  />
                  <span className="text-[10px] text-slate-500 mt-1 font-mono">
                    {k}
                  </span>
                </div>
              ))}
            </div>
            <div className="text-center text-[10px] text-slate-500 mt-2">
              # correct out of {n} rollouts  →  # prompt-groups
            </div>
          </div>

          {/* Evolution over training: stacked fraction per step-bucket */}
          {series && series.buckets.length > 0 && (
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-5 mt-4">
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs font-semibold text-slate-200">
                  Distribution over training
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[10px] text-slate-500 uppercase tracking-wider">
                    groups / bar (≈ train_bsz)
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={4096}
                    value={groupsPerBar}
                    onChange={(e) =>
                      setGroupsPerBar(
                        Math.max(1, Math.min(4096, +e.target.value || 1)),
                      )
                    }
                    className="w-16 bg-slate-800 text-slate-200 text-xs border border-slate-700 rounded-lg px-2 py-1 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>
              <p className="text-[10px] text-slate-500 mb-3">
                Each bar = ~{groupsPerBar} consecutive prompt-groups (oldest →
                newest by generation order). Set "groups/bar" to your{" "}
                <code>train_batch_size</code> so <b>one bar ≈ one training step</b>{" "}
                ({series.train_groups.toLocaleString()} groups →{" "}
                {series.buckets.length} bars). Stack = fraction of groups at each
                correct-count; watch the {n}/{n} (green) band grow and 0/{n} (red)
                shrink as the model learns. ⚠ This is generation order, not the
                exact training step (async staleness can reorder slightly).
              </p>
              <div className="flex items-end gap-0.5 h-56">
                {series.buckets.map((b) => (
                  <div
                    key={b.idx}
                    className="flex-1 flex flex-col h-full rounded overflow-hidden"
                    title={`bar ${b.idx + 1}/${series.buckets.length} (≈ training step ${b.idx + 1}) · ${b.count} prompt-groups\ngeneration-order step_${b.step_start}..${b.step_end} (sample index, NOT training step)\nmean solve ${(b.mean_solve_rate * 100).toFixed(0)}% · no-signal ${(b.no_signal_frac * 100).toFixed(0)}%`}
                  >
                    {Array.from({ length: n + 1 }, (_, j) => n - j).map((k) => (
                      <div
                        key={k}
                        style={{
                          height: `${(b.frac[k] ?? 0) * 100}%`,
                          background: barColor(k, n),
                        }}
                      />
                    ))}
                  </div>
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-slate-500 mt-1.5">
                <span>← earliest (≈ step 1)</span>
                <span>training progress (~1 bar = 1 step)</span>
                <span>latest →</span>
              </div>
              <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500">
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-[2px]" style={{ background: barColor(0, n) }} />0/{n}
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-[2px]" style={{ background: barColor(Math.floor(n / 2), n) }} />mid
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-[2px]" style={{ background: barColor(n, n) }} />{n}/{n}
                </span>
              </div>
            </div>
          )}

          {data.truncated && (
            <p className="text-[10px] text-amber-500/80 mt-3">
              ⚠ Based on the most recent {data.num_steps_scanned} rollout batches
              (run has more). Pass a larger window via the API ?max_steps= to
              include all.
            </p>
          )}
          {!isSnapshot && (
            <p className="text-[10px] text-slate-600 mt-2">
              Note: for async runs the on-disk <code>step_NNNN</code> folder is a
              global rollout/sample index, not the training step — and trial files
              carry no training-step / param-version, so the x-axis is
              rollout-batch order (oldest→newest), not training step.
            </p>
          )}
        </>
      )}
    </div>
  );
}
