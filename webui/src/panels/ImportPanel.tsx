import { useState, useCallback } from "react";
import { Upload, Loader2, CheckCircle2, AlertCircle, Trash2, ArrowRight } from "lucide-react";
import { ingestSnapshot, fetchIngestJob, deleteSnapshot } from "../api/client";
import { useT } from "../i18n";
import type { RunInfo, IngestJob } from "../types";

interface Props {
  runs: RunInfo[];
  onSelectRun: (id: string) => void;
  onChanged: () => void;
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="text-sm font-mono text-slate-200 mt-0.5">{value}</div>
    </div>
  );
}

export default function ImportPanel({ runs, onSelectRun, onChanged }: Props) {
  const { t } = useT();
  const [path, setPath] = useState("");
  const [logPath, setLogPath] = useState("");
  const [name, setName] = useState("");
  const [job, setJob] = useState<IngestJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const snapshots = runs.filter((r) => r.source === "snapshot");

  const poll = useCallback(
    (jobId: string) => {
      const tick = async () => {
        try {
          const j = await fetchIngestJob(jobId);
          setJob(j);
          if (j.status === "done") {
            setBusy(false);
            onChanged();
            if (j.run_id) onSelectRun(j.run_id);
            return;
          }
          if (j.status === "error") {
            setBusy(false);
            setErr(j.error || "ingest failed");
            return;
          }
          setTimeout(tick, 800);
        } catch (e) {
          setBusy(false);
          setErr(String(e));
        }
      };
      tick();
    },
    [onChanged, onSelectRun],
  );

  const start = useCallback(async () => {
    if (!path.trim()) return;
    setErr("");
    setJob(null);
    setBusy(true);
    try {
      const { job_id } = await ingestSnapshot(path.trim(), name.trim(), logPath.trim());
      poll(job_id);
    } catch (e) {
      setBusy(false);
      setErr(String(e));
    }
  }, [path, logPath, name, poll]);

  const onDelete = useCallback(
    async (runId: string) => {
      if (!confirm(t("import.confirmDelete"))) return;
      await deleteSnapshot(runId);
      onChanged();
    },
    [onChanged, t],
  );

  const pct = job && job.total ? Math.round((100 * (job.done || 0)) / job.total) : 0;
  const s = job?.summary;

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">{t("import.title")}</h2>
        <p className="text-xs text-slate-500 mt-1">{t("import.desc")}</p>
      </div>

      {/* Import form */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
        <div>
          <label className="text-[11px] uppercase tracking-wider text-slate-500">
            {t("import.pathLabel")}
          </label>
          <input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/.../harbor_trials/<project>/<exp-dir> or /path/to/.../bundle.zip"
            className="mt-1 w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">{t("import.pathHint")}</p>
        </div>
        <div>
          <label className="text-[11px] uppercase tracking-wider text-slate-500">
            {t("import.logPathLabel")}
          </label>
          <input
            value={logPath}
            onChange={(e) => setLogPath(e.target.value)}
            placeholder="/path/to/.../logs/<run>.log"
            className="mt-1 w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">{t("import.logPathHint")}</p>
        </div>
        <div>
          <label className="text-[11px] uppercase tracking-wider text-slate-500">
            {t("import.nameLabel")}
          </label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={t("import.namePlaceholder")}
            className="mt-1 w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
          />
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={start}
            disabled={busy || !path.trim()}
            className="flex items-center gap-2 bg-indigo-500 hover:bg-indigo-400 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            {busy ? <Loader2 size={15} className="animate-spin" /> : <Upload size={15} />}
            {t("import.button")}
          </button>
          {busy && job && (
            <div className="flex-1">
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                <span>{job.message || job.status}</span>
                <span>{job.total ? `${job.done}/${job.total}` : ""}</span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )}
        </div>
        {err && (
          <div className="flex items-center gap-2 text-xs text-rose-400">
            <AlertCircle size={14} /> {err}
          </div>
        )}
      </div>

      {/* Just-imported summary */}
      {s && job?.status === "done" && (
        <div className="bg-emerald-500/5 border border-emerald-500/30 rounded-xl p-4">
          <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium mb-3">
            <CheckCircle2 size={16} /> {t("import.done")}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Stat label={t("import.samples")} value={`${s.num_groups} × ${s.expected_n} = ${s.num_samples}`} />
            <Stat label={t("import.meanReward")} value={(s.mean_reward * 100).toFixed(1) + "%"} />
            <Stat label={t("import.zeroAdv")} value={`${s.zero_advantage_groups}/${s.num_groups}`} />
            <Stat label={t("import.repos")} value={s.num_repos} />
            <Stat label={t("import.respLen")} value={`${s.resp_len?.p50 ?? "—"} / ${s.resp_len?.max ?? "—"}`} />
            <Stat label={t("import.meanLogprob")} value={s.mean_logprob ?? "—"} />
            <Stat label={t("import.tokenFile")} value={s.has_token_file ? "✓" : "—"} />
            <Stat label={t("import.rewardSource")} value={s.reward_source} />
          </div>
          {job.run_id && (
            <button
              onClick={() => onSelectRun(job.run_id!)}
              className="mt-3 flex items-center gap-1.5 text-xs text-indigo-300 hover:text-indigo-200"
            >
              {t("import.openInPanels")} <ArrowRight size={13} />
            </button>
          )}
        </div>
      )}

      {/* Existing snapshots */}
      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-2">
          {t("import.existing")} ({snapshots.length})
        </h3>
        {snapshots.length === 0 ? (
          <p className="text-xs text-slate-600">{t("import.none")}</p>
        ) : (
          <div className="space-y-1.5">
            {snapshots.map((r) => (
              <div
                key={r.id}
                className="flex items-center justify-between bg-slate-900/40 border border-slate-800 rounded-lg px-3 py-2"
              >
                <button onClick={() => onSelectRun(r.id)} className="flex-1 text-left">
                  <div className="text-xs font-mono text-slate-200">{r.name}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">
                    {r.summary
                      ? `${r.summary.num_samples} samples · ${(r.summary.mean_reward * 100).toFixed(0)}% reward · ${r.summary.num_repos} repos`
                      : r.id}
                  </div>
                </button>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => onSelectRun(r.id)}
                    className="text-[11px] text-indigo-400 hover:text-indigo-300"
                  >
                    {t("import.open")}
                  </button>
                  <button
                    onClick={() => onDelete(r.id)}
                    className="text-slate-500 hover:text-rose-400"
                    title={t("import.delete")}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
