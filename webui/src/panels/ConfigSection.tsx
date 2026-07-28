import { useEffect, useState } from "react";
import { Loader2, SlidersHorizontal, ChevronRight, Server } from "lucide-react";
import { fetchRunConfig } from "../api/client";
import type { RunConfig } from "../api/client";
import { useT } from "../i18n";

interface Props {
  runId: string | null;
}

// Task-config board: shows the key knobs of a training run (model, data,
// rollout, timeouts, R3, drop reasons, val temp, nodes, ...) parsed from the
// run's console log by the backend `/api/runs/{id}/config` endpoint.
export default function ConfigSection({ runId }: Props) {
  const { t, lang } = useT();
  // Config labels come from the backend in both languages (see RunConfigItem);
  // the English variant is omitted where the label is already neutral.
  const loc = (zh: string, en?: string) => (lang === "en" ? (en ?? zh) : zh);
  const [cfg, setCfg] = useState<RunConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    if (!runId) {
      setCfg(null);
      return;
    }
    let alive = true;
    setLoading(true);
    fetchRunConfig(runId)
      .then((d) => alive && setCfg(d))
      .catch(() => alive && setCfg({ available: false, error: t("config.loadFailed") }))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId]);

  if (!runId) return null;

  return (
    <div className="mt-8">
      <div className="flex items-center gap-2 mb-1">
        <SlidersHorizontal className="w-4 h-4 text-indigo-400" />
        <h3 className="text-base font-semibold text-slate-100">
          {t("config.title")}
        </h3>
        {loading && <Loader2 className="w-3.5 h-3.5 text-slate-500 animate-spin" />}
      </div>
      <p className="text-xs text-slate-500 mb-4">{t("config.subtitle")}</p>

      {cfg && !cfg.available && (
        <div className="rounded-lg bg-slate-800/40 border border-slate-700/50 p-3 text-xs text-slate-400">
          {cfg.error || t("config.unavailable")}
        </div>
      )}

      {cfg?.available && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {cfg.sections?.map((sec) => (
              <div
                key={sec.title}
                className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4"
              >
                <div className="text-[10px] uppercase tracking-wider text-indigo-300/80 mb-1">
                  {loc(sec.title, sec.title_en)}
                </div>
                {sec.note && (
                  <div className="text-[10px] text-slate-500 mb-2 leading-snug">
                    {loc(sec.note, sec.note_en)}
                  </div>
                )}
                <dl className="divide-y divide-slate-800/60">
                  {sec.items.map((it) => (
                    <div
                      key={it.label}
                      className="flex items-baseline justify-between gap-3 py-1.5"
                    >
                      <dt className="text-xs text-slate-400 shrink-0">
                        {loc(it.label, it.label_en)}
                      </dt>
                      <dd
                        className={`text-xs font-mono text-slate-200 text-right ${
                          it.wide ? "break-all" : "truncate"
                        }`}
                        title={it.hint ? String(it.hint) : String(it.value)}
                      >
                        {loc(String(it.value), it.value_en == null ? undefined : String(it.value_en))}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>

          {cfg.nodes && cfg.nodes.length > 0 && (
            <div className="mt-3 rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-indigo-300/80 mb-2">
                <Server className="w-3 h-3" />
                {t("config.nodes")} ({cfg.nodes.length})
              </div>
              <div className="space-y-1.5">
                {cfg.nodes.map((n) => (
                  <div key={n.ip} className="flex items-center gap-3 text-xs">
                    <span className="font-mono text-slate-200 w-32 shrink-0">
                      {n.ip}
                    </span>
                    <span className="flex flex-wrap gap-1">
                      {n.roles.map((r) => (
                        <span
                          key={r}
                          className="px-1.5 py-0.5 rounded bg-slate-800/70 text-slate-400 text-[10px]"
                        >
                          {r}
                        </span>
                      ))}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {cfg.raw && (
            <div className="mt-3">
              <button
                onClick={() => setShowRaw((v) => !v)}
                className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors"
              >
                <ChevronRight
                  className={`w-3.5 h-3.5 transition-transform ${
                    showRaw ? "rotate-90" : ""
                  }`}
                />
                {t("config.rawToggle")}
              </button>
              {showRaw && (
                <pre className="mt-2 max-h-96 overflow-auto rounded-xl bg-slate-950/80 border border-slate-800/60 p-3 text-[11px] font-mono text-slate-400 leading-relaxed">
                  {JSON.stringify(cfg.raw, null, 2)}
                </pre>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
