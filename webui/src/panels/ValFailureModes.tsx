import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { useT } from "../i18n";

interface Props {
  runId: string | null;
  onOpenTrajectory?: (step: string, task: string) => void;
}

interface FMTask {
  task: string;
  repo: string;
  cat: string;
  step: string;
  task_dir: string;
}
interface FMData {
  val_steps: string[];
  step: string | null;
  categories: string[];
  counts: Record<string, number>;
  num_tasks: number;
  has_judgments: boolean;
  tasks: FMTask[];
  num_exp_dirs: number;
}

// display order, color, and bilingual label per category
const CAT_META: Record<string, { color: string; en: string; zh: string }> = {
  success: { color: "#10b981", en: "Solved", zh: "成功" },
  self_verif: { color: "#f59e0b", en: "Self-verif fail", zh: "自我验证失败" },
  wrong_fix: { color: "#ef4444", en: "Wrong fix", zh: "错误修复方案" },
  gold_edited_failed: { color: "#fb923c", en: "Right file, failed", zh: "改对文件但失败" },
  file_loc_fail: { color: "#8b5cf6", en: "File-loc fail", zh: "文件定位失败" },
  no_impl: { color: "#38bdf8", en: "No implementation", zh: "只分析未动手" },
  dead_loop: { color: "#ec4899", en: "Dead loop", zh: "死循环" },
  infra: { color: "#64748b", en: "Infra noise", zh: "基建噪声" },
};
const ORDER = [
  "success",
  "self_verif",
  "wrong_fix",
  "gold_edited_failed",
  "file_loc_fail",
  "no_impl",
  "dead_loop",
  "infra",
];

function pct(n: number, d: number): string {
  return d ? `${((100 * n) / d).toFixed(1)}%` : "—";
}

export default function ValFailureModes({ runId, onOpenTrajectory }: Props) {
  const { lang } = useT();
  const catLabel = (c: string) => CAT_META[c]?.[lang] ?? c;
  const [data, setData] = useState<FMData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<string>("");
  const [sel, setSel] = useState<string | null>(null);

  // Which run the in-flight request belongs to; a reply for the run you just
  // left must not repaint this panel.
  const reqRef = useRef<string | null>(runId);

  const load = useCallback(
    async (st?: string) => {
      if (!runId) return;
      const reqRun = runId;
      reqRef.current = runId;
      setLoading(true);
      setError(null);
      try {
        const q = st ? `?step=${encodeURIComponent(st)}` : "";
        const res = await fetch(`/api/runs/${reqRun}/val-failure-modes${q}`);
        const d = await res.json();
        if (reqRef.current !== reqRun) return; // superseded
        if (!res.ok || d.error) throw new Error(d.error || `HTTP ${res.status}`);
        setData(d);
        setStep(d.step ?? "");
      } catch (e: unknown) {
        if (reqRef.current !== reqRun) return;
        setError(e instanceof Error ? e.message : String(e));
        setData(null);
      } finally {
        if (reqRef.current === reqRun) setLoading(false);
      }
    },
    [runId],
  );

  useEffect(() => {
    load();
  }, [load]);

  const cats = useMemo(
    () => ORDER.filter((c) => (data?.counts[c] ?? 0) > 0),
    [data],
  );
  const total = data?.num_tasks ?? 0;
  const failures = total - (data?.counts.success ?? 0);

  const selTasks = useMemo(() => {
    if (!data || !sel) return [];
    return data.tasks.filter((t) => t.cat === sel).sort((a, b) => a.task.localeCompare(b.task));
  }, [data, sel]);

  return (
    <div className="mt-8 border-t border-slate-800/60 pt-6">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-base font-semibold text-slate-100">Validation Failure Modes</h3>
        <div className="flex items-center gap-2">
          {data && data.val_steps.length > 1 && (
            <select
              value={step}
              onChange={(e) => {
                setStep(e.target.value);
                setSel(null);
                load(e.target.value);
              }}
              className="px-2 py-1.5 rounded-lg border border-slate-700 bg-slate-900 text-slate-300 text-xs"
            >
              {data.val_steps.map((s, i) => (
                <option key={s} value={s}>
                  val #{i + 1} ({s}){i === data.val_steps.length - 1 ? " · final" : ""}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={() => load(step || undefined)}
            disabled={loading || !runId}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 transition-colors disabled:opacity-40"
          >
            {loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            Refresh
          </button>
        </div>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Each task of the selected val event classified into one outcome. Failure modes:
        located+implemented but unverified (self-verif), right file but wrong/incomplete patch (wrong-fix),
        edits in wrong files (file-loc), explored/described but never patched (no-impl), and repetition/budget
        collapse (dead-loop).{" "}
        {data && !data.has_judgments && (
          <span className="text-amber-500/80">
            self-verif vs wrong-fix needs offline LLM judging — shown combined as “right file, failed” for this event.
          </span>
        )}
      </p>

      {error && <div className="text-rose-400 text-sm mb-3">{error}</div>}

      {data && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Tasks</div>
              <div className="text-xl font-semibold text-slate-100">{total}</div>
            </div>
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Solved</div>
              <div className="text-xl font-semibold text-emerald-400">
                {data.counts.success ?? 0} <span className="text-sm text-slate-500">{pct(data.counts.success ?? 0, total)}</span>
              </div>
            </div>
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Failures</div>
              <div className="text-xl font-semibold text-rose-400">{failures}</div>
            </div>
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Event</div>
              <div className="text-xl font-semibold text-slate-100">{data.step}</div>
            </div>
          </div>

          {/* stacked bar over all tasks */}
          <div className="flex w-full h-6 rounded-lg overflow-hidden mb-2 border border-slate-800/60">
            {cats.map((c) => {
              const n = data.counts[c] ?? 0;
              return (
                <div
                  key={c}
                  title={`${catLabel(c)}: ${n} (${pct(n, total)})`}
                  onClick={() => setSel(sel === c ? null : c)}
                  style={{ width: `${(100 * n) / Math.max(total, 1)}%`, background: CAT_META[c]?.color ?? "#475569" }}
                  className="cursor-pointer hover:opacity-80 transition-opacity"
                />
              );
            })}
          </div>

          {/* legend / table */}
          <table className="w-full text-xs mb-4">
            <thead>
              <tr className="text-slate-500 text-left">
                <th className="py-1 font-medium">Category</th>
                <th className="py-1 font-medium text-right">Count</th>
                <th className="py-1 font-medium text-right">% of all</th>
                <th className="py-1 font-medium text-right">% of failures</th>
              </tr>
            </thead>
            <tbody>
              {cats.map((c) => {
                const n = data.counts[c] ?? 0;
                const active = sel === c;
                return (
                  <tr
                    key={c}
                    onClick={() => setSel(active ? null : c)}
                    className={`cursor-pointer border-t border-slate-800/40 ${active ? "bg-slate-800/60" : "hover:bg-slate-900/60"}`}
                  >
                    <td className="py-1.5 flex items-center gap-2">
                      <span className="inline-block w-3 h-3 rounded-sm" style={{ background: CAT_META[c]?.color ?? "#475569" }} />
                      <span className="text-slate-200">{catLabel(c)}</span>
                    </td>
                    <td className="py-1.5 text-right text-slate-200">{n}</td>
                    <td className="py-1.5 text-right text-slate-400">{pct(n, total)}</td>
                    <td className="py-1.5 text-right text-slate-400">{c === "success" ? "—" : pct(n, failures)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* drill-down task list for selected category */}
          {sel && (
            <div className="rounded-xl bg-slate-900/60 border border-slate-800/60 p-3">
              <div className="text-xs text-slate-400 mb-2">
                {selTasks.length} tasks · <span style={{ color: CAT_META[sel]?.color }}>{catLabel(sel)}</span>
                {" "}— click a task to open its trajectory
              </div>
              <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto">
                {selTasks.map((t) => (
                  <button
                    key={t.task}
                    onClick={() => onOpenTrajectory?.(t.step, t.task)}
                    className="px-2 py-1 rounded-md bg-slate-800/70 hover:bg-slate-700 text-[11px] text-slate-300 border border-slate-700/60"
                    title={t.task}
                  >
                    {t.task}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
