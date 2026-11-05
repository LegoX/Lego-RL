import { useEffect, useState } from "react";
import MultiTurnPanel from "./MultiTurnPanel";
import type { MetricPoint } from "../types";

// Explicit disk snapshot; never replace trainer validation curves with it.
export default function MultiTurnArtifacts({ runId }: { runId?: string | null }) {
  const [steps, setSteps] = useState<string[]>([]);
  const [step, setStep] = useState("");
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [error, setError] = useState("");
  useEffect(() => { setStep(""); setMetrics([]); setSteps([]); }, [runId]);
  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    const query = step ? `?step=${encodeURIComponent(step)}` : "";
    fetch(`/api/runs/${encodeURIComponent(runId)}/multiturn${query}`, { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => { setSteps(d.steps ?? []); setMetrics(Object.keys(d.metrics ?? {}).length ? [{ step: 0, ...d.metrics }] : []); setError(""); })
      .catch(e => { if (e.name !== "AbortError") setError("Saved multi-turn artifacts unavailable."); });
    return () => controller.abort();
  }, [runId, step]);
  if (!runId || (!steps.length && !error)) return null;
  return <details className="my-5 border border-slate-800 rounded p-3">
    <summary className="text-sm text-slate-300 cursor-pointer">Saved multi-turn evaluation / rollout artifacts</summary>
    <p className="text-xs text-slate-500 my-2">A snapshot of the selected artifact directory, not an optimizer-step curve. Select the directory for the evaluation or rollout you want to inspect.</p>
    <select aria-label="Artifact directory" value={step} onChange={e => setStep(e.target.value)} className="bg-slate-800 text-xs text-slate-200 rounded p-2"><option value="">Select directory</option>{steps.map(s => <option key={s}>{s}</option>)}</select>
    {error && <p className="text-xs text-amber-400">{error}</p>}
    {step && !metrics.length && !error && <p className="text-xs text-slate-500 mt-2">No multi-turn results in this directory.</p>}
    <MultiTurnPanel data={metrics} prefix="eval-multiturn" />
  </details>;
}
