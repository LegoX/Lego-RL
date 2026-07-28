import { useMemo } from "react";
import type { MetricPoint } from "../types";

// ---------------------------------------------------------------------------
// Is the reward curve actually moving, or is it noise?
//
// Reward per step on an agentic SWE task is loud: a 64-prompt batch of binary
// rewards scatters by several points step to step, which is routinely larger
// than a real trend over tens of steps. Eyeballing the chart cannot separate
// them, so this fits a least-squares line and reports the slope against its own
// standard error. |t| >= 2 ~ the trend clears the scatter; below that the run
// has not demonstrably moved, however encouraging the line looks.
//
// Cheap and local: reads the per-step metrics already loaded for the page, no
// backend call and no trial-dir scan.
// ---------------------------------------------------------------------------

interface Fit {
  n: number;
  slope: number;
  se: number;
  t: number | null;
  sd: number;
  firstK: number;
  lastK: number;
  k: number;
  span: number;
}

function linfit(xs: number[], ys: number[]): Fit | null {
  const n = xs.length;
  if (n < 4) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const se = Math.sqrt(ss / (n - 2) / sxx);
  const sd = Math.sqrt(ys.reduce((a, y) => a + (y - my) ** 2, 0) / n);
  const k = Math.max(1, Math.min(5, Math.floor(n / 4)));
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  return {
    n,
    slope,
    se,
    t: se > 0 ? slope / se : null,
    sd,
    firstK: mean(ys.slice(0, k)),
    lastK: mean(ys.slice(-k)),
    k,
    span: xs[n - 1] - xs[0],
  };
}

function pickSeries(data: MetricPoint[]): { key: string; pts: [number, number][] } | null {
  for (const key of ["critic/score/mean", "critic/rewards/mean"]) {
    const pts: [number, number][] = [];
    data.forEach((p, i) => {
      const v = p[key];
      if (typeof v === "number" && !Number.isNaN(v)) {
        pts.push([typeof p.step === "number" ? p.step : i, v]);
      }
    });
    if (pts.length >= 4) return { key, pts };
  }
  return null;
}

function Tile({
  label,
  value,
  sub,
  accent = "text-slate-100",
}: {
  label: string;
  value: string;
  sub: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 px-4 py-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
        {label}
      </div>
      <div className={`text-xl font-semibold tabular-nums ${accent}`}>{value}</div>
      <div className="text-[10px] text-slate-500 mt-0.5 leading-snug">{sub}</div>
    </div>
  );
}

export default function TrendSignificance({ data }: { data: MetricPoint[] }) {
  const res = useMemo(() => {
    const s = pickSeries(data);
    if (!s) return null;
    const fit = linfit(
      s.pts.map((p) => p[0]),
      s.pts.map((p) => p[1]),
    );
    return fit ? { key: s.key, fit } : null;
  }, [data]);

  if (!res) return null;
  const { key, fit } = res;
  const t = fit.t;

  const verdict =
    t === null
      ? { text: "—", accent: "text-slate-400", note: "slope has no error estimate" }
      : t >= 2
        ? { text: "Rising", accent: "text-emerald-400", note: "trend beats the noise" }
        : t <= -2
          ? { text: "Falling", accent: "text-rose-400", note: "trend beats the noise" }
          : { text: "Inconclusive", accent: "text-amber-400", note: "noise dominates; you cannot call this a rise" };

  const drift = Math.abs(fit.slope * fit.span);
  const ratio = drift > 0 ? fit.sd / drift : Infinity;
  const delta = fit.lastK - fit.firstK;

  return (
    <div className="mb-8">
      <div className="flex items-baseline gap-2 mb-1">
        <h3 className="text-base font-semibold text-slate-100">Trend significance</h3>
        <code className="text-[10px] text-indigo-300">{key}</code>
      </div>
      <p className="text-xs text-slate-500 mb-3">
        Least-squares fit over {fit.n} step points; t is the slope divided by its own
        standard error.<span className="text-slate-300"> |t| ≥ 2</span> is the bar for
        a real move. Below it, a curve that looks like it is climbing is just
        step-to-step jitter — on agentic tasks the per-step noise of a binary reward
        routinely swamps the real trend over dozens of steps.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Tile
          label="Verdict"
          value={verdict.text}
          sub={verdict.note}
          accent={verdict.accent}
        />
        <Tile
          label="t value"
          value={t === null ? "—" : (t >= 0 ? "+" : "") + t.toFixed(2)}
          sub={`slope ${fit.slope >= 0 ? "+" : ""}${fit.slope.toPrecision(3)} / step`}
          accent={verdict.accent}
        />
        <Tile
          label={`first ${fit.k} steps → last ${fit.k} steps`}
          value={`${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`}
          sub={`${fit.firstK.toFixed(3)} → ${fit.lastK.toFixed(3)}`}
        />
        <Tile
          label="Noise / trend"
          value={
            ratio === Infinity ? "∞" : `${ratio < 10 ? ratio.toFixed(1) : ratio.toFixed(0)}×`
          }
          sub={`step-to-step sd ${fit.sd.toFixed(3)}, total drift ${drift.toFixed(3)}`}
          accent={ratio > 3 ? "text-amber-400" : "text-slate-100"}
        />
      </div>
      {t !== null && Math.abs(t) < 2 && (
        <p className="text-[11px] text-amber-400/90 mt-2">
          Note: this curve has not moved in any statistical sense yet. Whether training
          is working is a question for a held-out validation set; at this t value the
          training reward on its own is not evidence.
        </p>
      )}
    </div>
  );
}
