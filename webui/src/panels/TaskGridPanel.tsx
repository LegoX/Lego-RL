import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Loader2, RefreshCw, TrendingUp, TrendingDown } from "lucide-react";

interface Props {
  runId: string | null;
  onOpenTrajectory?: (step: string, task: string) => void;
}

interface Epoch {
  step: string;
  n: number;
  solved: number;
  rate: number | null;
}

interface TaskStat {
  task: string;
  epochs: Epoch[];
  num_epochs: number;
  rollouts: number;
  solved: number;
  solve_rate: number | null;
  first_rate: number | null;
  last_rate: number | null;
  first_step: string;
  last_step: string;
  delta: number | null;
}

interface StatsData {
  expected_n: number;
  tasks: TaskStat[];
  num_tasks: number;
  never_solved: number;
  always_solved: number;
  multi_epoch: number;
  improved: number;
  regressed: number;
  num_steps_scanned: number;
  truncated: boolean;
  num_exp_dirs: number;
}

// Trajectory SHAPE per task, first epoch vs last (served separately from the
// solve rates because the cold compute reads ~12k trajectory files, ~30s — the
// grid must not wait for it).
interface TrajStat {
  task: string;
  first_turns: number | null;
  last_turns: number | null;
  first_in_tok: number | null;
  last_in_tok: number | null;
  first_tool_chars: number | null;
  last_tool_chars: number | null;
  first_out_tok: number | null;
  last_out_tok: number | null;
  first_cot_chars: number | null;
  last_cot_chars: number | null;
  first_resp_chars: number | null;
  last_resp_chars: number | null;
  first_cot_ratio: number | null;
  last_cot_ratio: number | null;
}

interface TrajSummaryEntry {
  first: number | null;
  last: number | null;
  delta: number | null;
  pct: number | null;
}

interface TrajData {
  tasks: TrajStat[];
  num_tasks: number;
  trials_read: number;
  summary: Record<string, TrajSummaryEntry>;
}

type ColorMode = "solve" | "delta";

function compact(v: number | null): string {
  if (v == null) return "—";
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return v.toFixed(0);
}
// One LLM call emits a single token stream: reasoning, `</think>`, the visible
// text, then the tool call — so the response is all three, and `resp_chars`
// counts `content` PLUS the tool-call arguments (verified against the model's
// own tokenizer: reconstructed tokens match n_output_tokens within 1%).
function cotBreakdown(tj: TrajStat): string {
  const line = (side: "first" | "last") => {
    const resp = tj[`${side}_resp_chars`];
    const cot = tj[`${side}_cot_chars`];
    const tool = tj[`${side}_tool_chars`];
    if (resp == null) return `${side}: —`;
    const other = resp - (cot ?? 0) - (tool ?? 0);
    return `${side}: ${compact(cot)} CoT + ${compact(tool)} tool args + ${compact(other)} text = ${compact(resp)} chars`;
  };
  return [
    `CoT share of response: ${((tj.first_cot_ratio ?? 0) * 100).toFixed(0)}% → ${((tj.last_cot_ratio ?? 0) * 100).toFixed(0)}%`,
    line("first"),
    line("last"),
    `prompt tokens (summed over turns): ${compact(tj.first_in_tok)} → ${compact(tj.last_in_tok)}`,
  ].join("\n");
}

// Growth is not automatically bad (nor good) — a longer trajectory that solves
// more is fine — so these are tinted, not scored: amber = grew, slate = shrank.
function growthTint(pct: number | null): string {
  if (pct == null) return "text-slate-500";
  if (pct > 0.15) return "text-amber-400";
  if (pct < -0.15) return "text-sky-400";
  return "text-slate-400";
}

function lerp(a: number, b: number, t: number) {
  return Math.round(a + (b - a) * t);
}
function rgb(r: number, g: number, b: number) {
  return `rgb(${r},${g},${b})`;
}
function solveColor(v: number | null): string {
  if (v === null) return "#302a24";
  if (v < 0.5) {
    const t = v / 0.5;
    return rgb(lerp(244, 245, t), lerp(63, 158, t), lerp(94, 11, t));
  }
  const t = (v - 0.5) / 0.5;
  return rgb(lerp(245, 16, t), lerp(158, 185, t), lerp(11, 129, t));
}
function deltaColor(v: number | null): string {
  if (v === null) return "#302a24";
  const c = Math.max(-1, Math.min(1, v));
  if (c < 0) return rgb(244, lerp(63, 100, 1 + c), lerp(94, 116, 1 + c));
  return rgb(lerp(100, 16, c), lerp(116, 185, c), lerp(139, 129, c));
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-xl font-semibold ${accent ?? "text-slate-100"}`}>{value}</div>
    </div>
  );
}

// per-epoch solve-rate sparkline; each bar is clickable -> open that epoch's trajectory
function EpochSparkline({
  task,
  onOpen,
}: {
  task: TaskStat;
  onOpen?: (step: string, task: string) => void;
}) {
  return (
    <div className="flex items-end gap-0.5 h-7">
      {task.epochs.map((e, i) => (
        <button
          key={i}
          onClick={(ev) => {
            ev.stopPropagation();
            onOpen?.(e.step, task.task);
          }}
          title={`epoch ${i + 1} (${e.step}): ${e.solved}/${e.n} → ${((e.rate ?? 0) * 100).toFixed(0)}%  ·  click to view trajectory`}
          className="w-2.5 rounded-t hover:ring-1 hover:ring-white/50"
          style={{
            height: `${Math.max(8, (e.rate ?? 0) * 100)}%`,
            background: solveColor(e.rate),
          }}
        />
      ))}
    </div>
  );
}

const ROW_COLS =
  "grid grid-cols-[1fr_100px_92px_56px_74px_84px_74px] gap-2 px-4 py-1.5 items-center";

function TaskRow({
  t,
  tj,
  onOpen,
}: {
  t: TaskStat;
  tj?: TrajStat;
  onOpen?: (step: string, task: string) => void;
}) {
  const pct = (a: number | null | undefined, b: number | null | undefined) =>
    a && b != null ? (b - a) / a : null;
  return (
    <div
      onClick={() => onOpen?.(t.last_step, t.task)}
      className={`${ROW_COLS} text-xs border-b border-slate-800/40 hover:bg-slate-800/40 cursor-pointer`}
    >
      <span className="font-mono text-slate-300 truncate" title={`${t.task}\nclick to view latest trajectory`}>
        {t.task}
      </span>
      <EpochSparkline task={t} onOpen={onOpen} />
      <span className="text-slate-400 font-mono">
        {((t.first_rate ?? 0) * 100).toFixed(0)}% → {((t.last_rate ?? 0) * 100).toFixed(0)}%
      </span>
      <span
        className={`font-mono font-semibold text-right ${
          (t.delta ?? 0) > 0 ? "text-emerald-400" : (t.delta ?? 0) < 0 ? "text-rose-400" : "text-slate-500"
        }`}
      >
        {t.delta != null ? `${t.delta > 0 ? "+" : ""}${(t.delta * 100).toFixed(0)}` : "—"}
      </span>
      <span
        className={`font-mono text-right ${growthTint(pct(tj?.first_turns, tj?.last_turns))}`}
        title={tj ? `turns (assistant completions): ${compact(tj.first_turns)} → ${compact(tj.last_turns)}` : "no trajectory data"}
      >
        {tj ? `${compact(tj.first_turns)}→${compact(tj.last_turns)}` : "—"}
      </span>
      <span
        className={`font-mono text-right ${growthTint(pct(tj?.first_out_tok, tj?.last_out_tok))}`}
        title={tj ? `response tokens: ${compact(tj.first_out_tok)} → ${compact(tj.last_out_tok)}` : ""}
      >
        {tj ? `${compact(tj.first_out_tok)}→${compact(tj.last_out_tok)}` : "—"}
      </span>
      <span
        className={`font-mono text-right ${growthTint(pct(tj?.first_cot_ratio, tj?.last_cot_ratio))}`}
        title={tj ? cotBreakdown(tj) : ""}
      >
        {tj && tj.first_cot_ratio != null
          ? `${(tj.first_cot_ratio * 100).toFixed(0)}→${((tj.last_cot_ratio ?? 0) * 100).toFixed(0)}%`
          : "—"}
      </span>
    </div>
  );
}

function Section({
  title,
  icon,
  accent,
  rows,
  trajBy,
  onOpen,
}: {
  title: string;
  icon: React.ReactNode;
  accent: string;
  rows: TaskStat[];
  trajBy: Map<string, TrajStat>;
  onOpen?: (step: string, task: string) => void;
}) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 overflow-hidden">
      <div className={`flex items-center gap-2 px-4 py-2.5 border-b border-slate-800/60 ${accent}`}>
        {icon}
        <span className="text-sm font-semibold">{title}</span>
        <span className="text-[10px] text-slate-500">({rows.length})</span>
      </div>
      <div className={`${ROW_COLS} text-[10px] uppercase tracking-wider text-slate-500 border-b border-slate-800/40`}>
        <span>Task</span>
        <span>Per-epoch</span>
        <span>First → Last</span>
        <span className="text-right">Δ%</span>
        <span className="text-right" title="Assistant completions per rollout, first epoch → last">Turns</span>
        <span className="text-right" title="Response tokens per rollout, first epoch → last">Resp tok</span>
        <span className="text-right" title="Chain-of-thought as a share of the response, first epoch → last">CoT %</span>
      </div>
      <div className="max-h-[360px] overflow-y-auto">
        {rows.length === 0 ? (
          <div className="px-4 py-6 text-center text-xs text-slate-600">none</div>
        ) : (
          rows.map((t) => (
            <TaskRow key={t.task} t={t} tj={trajBy.get(t.task)} onOpen={onOpen} />
          ))
        )}
      </div>
    </div>
  );
}

export default function TaskGridPanel({ runId, onOpenTrajectory }: Props) {
  const [data, setData] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [colorMode, setColorMode] = useState<ColorMode>("solve");
  const [traj, setTraj] = useState<TrajData | null>(null);
  const [trajLoading, setTrajLoading] = useState(false);

  // Which run the in-flight request belongs to. This one matters most: a cold
  // task-stats takes minutes, so a reply for the previous run is very likely to
  // land after you have switched.
  const reqRef = useRef<string | null>(runId);

  const load = useCallback(async () => {
    if (!runId) return;
    const reqRun = runId;
    reqRef.current = runId;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/runs/${reqRun}/task-stats?n=8`);
      // A cold compute can outlast an upstream proxy's timeout (a Cloudflare
      // quick tunnel cuts at ~100s), and what comes back is that proxy's HTML
      // error page, not our JSON. Say so, instead of surfacing a parse error.
      const body = await res.text();
      if (reqRef.current !== reqRun) return; // superseded
      let d: StatsData & { error?: string };
      try {
        d = JSON.parse(body);
      } catch {
        throw new Error(
          body.trimStart().startsWith("<")
            ? "The first pass is still computing (a per-task scan of the whole run, which can take minutes from cold) and the gateway timed out. Hit Refresh in a moment — the result is cached, so every later load is instant."
            : `Could not parse the response (HTTP ${res.status})`,
        );
      }
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

  // Fired independently of `load` so a 30s cold trajectory scan never delays the
  // solve-rate grid; the extra columns fill in when it lands.
  useEffect(() => {
    if (!runId) {
      setTraj(null);
      return;
    }
    let cancelled = false;
    setTrajLoading(true);
    fetch(`/api/runs/${runId}/task-traj-stats?n=8`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled) return;
        setTraj(d && !d.error ? (d as TrajData) : null);
      })
      .catch(() => {
        if (!cancelled) setTraj(null);
      })
      .finally(() => {
        if (!cancelled) setTrajLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const trajBy = useMemo(
    () => new Map((traj?.tasks ?? []).map((t) => [t.task, t])),
    [traj],
  );

  const q = search.trim().toLowerCase();
  const filtered = useMemo(() => {
    const ts = data?.tasks ?? [];
    return q ? ts.filter((t) => t.task.toLowerCase().includes(q)) : ts;
  }, [data, q]);

  const improved = useMemo(
    () => filtered.filter((t) => t.delta != null && t.delta > 0).sort((a, b) => (b.delta ?? 0) - (a.delta ?? 0)),
    [filtered],
  );
  const regressed = useMemo(
    () => filtered.filter((t) => t.delta != null && t.delta < 0).sort((a, b) => (a.delta ?? 0) - (b.delta ?? 0)),
    [filtered],
  );
  const gridTasks = useMemo(
    () => [...filtered].sort((a, b) => (a.solve_rate ?? -1) - (b.solve_rate ?? -1)),
    [filtered],
  );

  const colorOf = (t: TaskStat) =>
    colorMode === "solve" ? solveColor(t.solve_rate) : deltaColor(t.delta);

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-slate-100">Task Solve-Rate Grid</h2>
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
        Per-task solve rate across the whole run. Each task is re-rolled out once
        per epoch ({data?.expected_n ?? 8} rollouts each), so first-epoch vs
        last-epoch shows learning / forgetting. Click a task (or an epoch bar) to
        open its trajectory. Aggregated across {data?.num_exp_dirs ?? "?"} async
        worker dirs.
      </p>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4 text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}

      {loading && !data && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-6 flex items-center justify-center gap-3">
          <Loader2 size={18} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">Scanning full run (~20s)…</span>
        </div>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
            <Stat label="Tasks" value={data.num_tasks.toLocaleString()} />
            <Stat label="Never solved" value={String(data.never_solved)} accent="text-rose-400" />
            <Stat label="Always solved" value={String(data.always_solved)} accent="text-emerald-400" />
            <Stat label="Improved" value={String(data.improved)} accent="text-emerald-400" />
            <Stat label="Regressed" value={String(data.regressed)} accent="text-rose-400" />
          </div>

          {/* Trajectory shape, first epoch vs last — the cost side of the solve
              rate above: same score with fewer turns is real improvement, a
              better score bought with 2x the CoT may not be. */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 mb-5">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-semibold text-slate-200">
                Trajectory shape · first epoch → last epoch
              </span>
              {trajLoading && <Loader2 size={13} className="text-indigo-400 animate-spin" />}
              {traj && (
                <span className="text-[10px] text-slate-500">
                  {traj.num_tasks} tasks with ≥2 epochs · {traj.trials_read.toLocaleString()} rollouts read
                </span>
              )}
            </div>
            {!traj && !trajLoading && (
              <div className="text-xs text-slate-600">no trajectory data for this run</div>
            )}
            {traj && (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {(
                  [
                    ["turns", "Turns / rollout", (v: number) => v.toFixed(1)],
                    ["in_tok", "Prompt tokens", (v: number) => Math.round(v).toLocaleString()],
                    ["out_tok", "Response tokens", (v: number) => Math.round(v).toLocaleString()],
                    ["cot_chars", "CoT chars", (v: number) => Math.round(v).toLocaleString()],
                    ["cot_ratio", "CoT share of response", (v: number) => `${(v * 100).toFixed(1)}%`],
                  ] as const
                ).map(([key, label, fmt]) => {
                  const s = traj.summary?.[key];
                  if (!s || s.first == null || s.last == null) return null;
                  return (
                    <div key={key} className="rounded-lg bg-slate-950/50 border border-slate-800/60 p-3">
                      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{label}</div>
                      <div className="text-sm font-mono text-slate-300">
                        {fmt(s.first)} <span className="text-slate-600">→</span>{" "}
                        <span className="text-slate-100 font-semibold">{fmt(s.last)}</span>
                      </div>
                      {s.pct != null && (
                        <div className={`text-xs font-mono mt-0.5 ${growthTint(s.pct)}`}>
                          {s.pct > 0 ? "+" : ""}
                          {(s.pct * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2 flex-wrap mb-3">
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search tasks…"
              className="bg-slate-800 text-slate-200 text-xs border border-slate-700 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-500 w-56"
            />
            <div className="flex rounded-lg border border-slate-700 overflow-hidden text-xs">
              <button
                onClick={() => setColorMode("solve")}
                className={`px-2.5 py-1.5 ${colorMode === "solve" ? "bg-indigo-500 text-white" : "text-slate-400 hover:bg-slate-800"}`}
              >
                heatmap: solve rate
              </button>
              <button
                onClick={() => setColorMode("delta")}
                className={`px-2.5 py-1.5 ${colorMode === "delta" ? "bg-indigo-500 text-white" : "text-slate-400 hover:bg-slate-800"}`}
              >
                heatmap: Δ last−first
              </button>
            </div>
          </div>

          {/* Heatmap overview */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 mb-4">
            <div className="flex flex-wrap gap-1">
              {gridTasks.map((t) => (
                <button
                  key={t.task}
                  onClick={() => onOpenTrajectory?.(t.last_step, t.task)}
                  className="w-3.5 h-3.5 rounded-[3px] hover:ring-1 hover:ring-white/60"
                  style={{ background: colorOf(t) }}
                  title={`${t.task}\nsolve ${((t.solve_rate ?? 0) * 100).toFixed(0)}% (${t.solved}/${t.rollouts}) over ${t.num_epochs} epochs${t.delta != null ? `\nfirst ${((t.first_rate ?? 0) * 100).toFixed(0)}% → last ${((t.last_rate ?? 0) * 100).toFixed(0)}% (Δ${(t.delta * 100).toFixed(0)})` : ""}\nclick to view trajectory`}
                />
              ))}
            </div>
          </div>

          {/* Improved / Regressed split */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Section
              title="Improved (first → last epoch)"
              icon={<TrendingUp size={14} />}
              accent="text-emerald-400"
              rows={improved}
              trajBy={trajBy}
              onOpen={onOpenTrajectory}
            />
            <Section
              title="Regressed (first → last epoch)"
              icon={<TrendingDown size={14} />}
              accent="text-rose-400"
              rows={regressed}
              trajBy={trajBy}
              onOpen={onOpenTrajectory}
            />
          </div>

          {data.truncated && (
            <p className="text-[10px] text-amber-500/80 mt-3">
              ⚠ Based on the most recent {data.num_steps_scanned} rollout batches (run has more).
            </p>
          )}
        </>
      )}
    </div>
  );
}
