import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

interface SampleStat {
  idx: number;
  reward: number | null;
  n_tokens: number;
  mean_old: number | null;
  mean_roll: number | null;
  mean_log_ratio: number | null;
  ess_frac: number | null;
  seq_log_ratio: number | null;
  max_abs_log_ratio: number | null;
  pearson: number | null;
}

interface TokenStats {
  global: {
    pearson: number | null;
    mean_log_ratio: number;
    total_valid_tokens: number;
    mean_tokens_per_sample: number;
    mean_ess_frac: number | null;
    hist: { lo: number; hi: number; bins: number; counts: number[] };
  };
  per_sample: SampleStat[];
  series: number[][];
  series_points: number;
}

interface Props {
  runId: string | null;
  data?: MetricPoint[];
  availableKeys?: string[];
}

// Live-run trend charts built from metrics verl already logs every training
// step. Same rollout-vs-training mismatch diagnostics as the snapshot per-token
// view, but as time-series. Keys are filtered to those actually present.
const LIVE_GROUPS: {
  title: string;
  desc: string;
  keys: string[];
  colors: string[];
}[] = [
  {
    title: "Rollout↔Actor prob agreement (pearson)",
    desc: "Correlation between training-policy and rollout-policy token probs. →1 = they match (healthy). Same as the snapshot per-token pearson, per step.",
    keys: ["training/rollout_actor_probs_pearson_corr"],
    colors: ["#10b981"],
  },
  {
    title: "Rollout vs training prob diff",
    desc: "How far the two policies' token probabilities drift apart. →0 = agree.",
    keys: [
      "training/rollout_probs_diff_mean",
      "training/rollout_probs_diff_std",
      "training/rollout_probs_diff_max",
    ],
    colors: ["#6366f1", "#818cf8", "#a5b4fc"],
  },
  {
    title: "Rollout-correction KL",
    desc: "KL divergence between rollout and training policy. →0 = on-policy. Rising = staleness/mismatch.",
    keys: ["rollout_corr/kl", "rollout_corr/k3_kl"],
    colors: ["#f43f5e", "#fb7185"],
  },
  {
    title: "Chi² (ESS proxy)",
    desc: "Chi-square divergence; effective sample size ≈ N/(1+chi²). High = IS weights skewed, gradient starved.",
    keys: ["rollout_corr/chi2_token", "rollout_corr/chi2_seq"],
    colors: ["#06b6d4", "#0891b2"],
  },
  {
    title: "Log-perplexity diff (percentiles)",
    desc: "Per-step spread of |rollout − training| log-perplexity. p99 spiking = a tail of badly-mismatched tokens.",
    keys: [
      "rollout_corr/log_ppl_diff_p50",
      "rollout_corr/log_ppl_diff_p90",
      "rollout_corr/log_ppl_diff_p99",
    ],
    colors: ["#8b5cf6", "#c4b5fd", "#7c3aed"],
  },
  {
    title: "Fraction of badly-mismatched tokens",
    desc: "Share of tokens whose log-ppl diff exceeds 1 / 2 nats. Watch these grow when training destabilizes.",
    keys: ["rollout_corr/log_ppl_diff_gt1_frac", "rollout_corr/log_ppl_diff_gt2_frac"],
    colors: ["#f59e0b", "#ef4444"],
  },
  {
    title: "Rollout vs training perplexity",
    desc: "Absolute perplexity of each engine on the same tokens; a persistent gap = engine/precision mismatch.",
    keys: ["rollout_corr/rollout_log_ppl", "rollout_corr/training_log_ppl"],
    colors: ["#22d3ee", "#fbbf24"],
  },
  {
    title: "PPO KL & clip fraction",
    desc: "Policy-update KL and how often the PPO ratio is clipped. Large = aggressive/off-policy updates.",
    keys: ["actor/ppo_kl", "actor/pg_clipfrac"],
    colors: ["#ec4899", "#f59e0b"],
  },
];

function Card({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: "good" | "warn" | "bad";
}) {
  const color =
    tone === "bad"
      ? "text-rose-300"
      : tone === "warn"
        ? "text-amber-300"
        : tone === "good"
          ? "text-emerald-300"
          : "text-slate-100";
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-xl font-mono mt-1 ${color}`}>{value}</div>
      {hint && <div className="text-[10px] text-slate-600 mt-0.5">{hint}</div>}
    </div>
  );
}

// log-ratio histogram (per token): counts[0]=underflow, [1..bins]=range, [bins+1]=overflow
function Histogram({ hist }: { hist: TokenStats["global"]["hist"] }) {
  const { lo, hi, bins, counts } = hist;
  const maxLog = Math.max(...counts.map((c) => Math.log10(c + 1)), 1);
  const W = bins + 2;
  return (
    <div>
      <div className="flex items-end gap-px h-40">
        {counts.map((c, i) => {
          const h = (Math.log10(c + 1) / maxLog) * 100;
          const isOverflow = i === 0 || i === W - 1;
          const center = i === 0 ? lo - 0.1 : i === W - 1 ? hi + 0.1 : lo + ((i - 0.5) / bins) * (hi - lo);
          const near0 = Math.abs(center) < 0.05;
          return (
            <div
              key={i}
              className="flex-1"
              style={{
                height: `${Math.max(h, c > 0 ? 1.5 : 0)}%`,
                background: isOverflow ? "#f43f5e" : near0 ? "#10b981" : "#6366f1",
              }}
              title={`log-ratio ≈ ${center.toFixed(2)}${isOverflow ? " (overflow)" : ""}\n${c.toLocaleString()} tokens`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-slate-500 mt-1">
        <span>&lt;{lo} (train≪rollout)</span>
        <span>0 (identical)</span>
        <span>&gt;{hi} (train≫rollout)</span>
      </div>
      <div className="text-[10px] text-slate-600 mt-1">
        Per-token log importance ratio log(π_train/π_rollout) = old_log_prob −
        rollout_log_prob. Bar height is log-scaled. A tall spike at 0 = the two
        policies agree; fat/heavy tails or a shifted peak = rollout-vs-training
        mismatch (the thing that starves gradients / collapses ESS).
      </div>
    </div>
  );
}

// per-sample downsampled log-ratio trace
function Trace({ series }: { series: number[] }) {
  if (!series || series.length === 0)
    return <div className="text-xs text-slate-600">no response tokens</div>;
  const max = Math.max(...series.map(Math.abs), 0.2);
  const W = 600;
  const H = 120;
  const pts = series
    .map((v, i) => `${(i / (series.length - 1)) * W},${H / 2 - (v / max) * (H / 2 - 4)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 120 }}>
      <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="#475569" strokeWidth="0.5" />
      <polyline points={pts} fill="none" stroke="#6366f1" strokeWidth="1" />
      <text x="2" y="10" fill="#64748b" fontSize="9">
        +{max.toFixed(2)}
      </text>
      <text x="2" y={H - 3} fill="#64748b" fontSize="9">
        −{max.toFixed(2)}
      </text>
    </svg>
  );
}

type SortKey = "pearson" | "ess_frac" | "max_abs_log_ratio" | "n_tokens" | "idx";

function LiveISView({
  data,
  availableKeys,
}: {
  data: MetricPoint[];
  availableKeys: string[];
}) {
  const keySet = useMemo(() => new Set(availableKeys), [availableKeys]);
  const groups = useMemo(
    () =>
      LIVE_GROUPS.map((g) => ({
        ...g,
        keys: g.keys.filter((k) => keySet.has(k)),
      })).filter((g) => g.keys.length > 0),
    [keySet],
  );

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">
        Token &amp; Importance-Sampling
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        Rollout-vs-training mismatch diagnostics this run logs every training
        step. (Per-token drill-down is only available for imported snapshots that
        include the raw token tensor file — live runs don&apos;t dump it, but
        they log these aggregates instead.)
      </p>
      {groups.length === 0 ? (
        <div className="text-sm text-slate-400">
          This run has no rollout-correction / IS metrics logged
          (training/rollout_* or rollout_corr/*). Nothing to chart.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {groups.map((g) => (
            <div key={g.title}>
              <ChartPanel title={g.title} data={data} keys={g.keys} colors={g.colors} />
              <p className="text-[10px] text-slate-600 mt-1 px-1">{g.desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function TokenISPanel({ runId, data = [], availableKeys = [] }: Props) {
  const [stats, setStats] = useState<TokenStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("pearson");
  const [sel, setSel] = useState<number | null>(null);

  const isSnapshot = !!runId?.startsWith("snap__");

  // Which run the in-flight request belongs to; a reply for the run you just
  // left must not repaint this panel.
  const reqRef = useRef<string | null>(runId);

  const load = useCallback(async () => {
    if (!runId || !isSnapshot) return;
    const reqRun = runId;
    reqRef.current = runId;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`/api/runs/${reqRun}/token-stats`);
      const d = await r.json();
      if (reqRef.current !== reqRun) return; // superseded
      if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
      setStats(d);
      setSel(d.per_sample?.[0]?.idx ?? null);
    } catch (e: unknown) {
      if (reqRef.current !== reqRun) return;
      setError(e instanceof Error ? e.message : String(e));
      setStats(null);
    } finally {
      if (reqRef.current === reqRun) setLoading(false);
    }
  }, [runId, isSnapshot]);

  useEffect(() => {
    load();
  }, [load]);

  const sorted = useMemo(() => {
    if (!stats) return [];
    const rows = [...stats.per_sample].filter((s) => s.n_tokens > 0);
    const asc = sortKey === "pearson" || sortKey === "ess_frac";
    rows.sort((a, b) => {
      const av = (a[sortKey] ?? 0) as number;
      const bv = (b[sortKey] ?? 0) as number;
      return asc ? av - bv : bv - av;
    });
    return rows;
  }, [stats, sortKey]);

  if (!isSnapshot) {
    return <LiveISView data={data} availableKeys={availableKeys} />;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-slate-100">Token &amp; Importance-Sampling</h2>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 disabled:opacity-40"
        >
          {loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          Refresh
        </button>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Per-token comparison of the training policy (old_log_probs) vs the rollout
        policy (rollout_log_probs) from the token tensor file. High agreement
        (pearson→1, ESS→1, log-ratio→0) means the gradient sees what was actually
        sampled; divergence is the rollout/training-mismatch failure mode.
      </p>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4 text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}

      {stats && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-5">
            <Card
              label="Pearson(old, rollout)"
              value={stats.global.pearson?.toFixed(4) ?? "—"}
              tone={
                (stats.global.pearson ?? 0) > 0.99
                  ? "good"
                  : (stats.global.pearson ?? 0) > 0.95
                    ? "warn"
                    : "bad"
              }
              hint="1.0 = perfect match"
            />
            <Card
              label="Mean ESS fraction"
              value={
                stats.global.mean_ess_frac != null
                  ? (stats.global.mean_ess_frac * 100).toFixed(1) + "%"
                  : "—"
              }
              tone={
                (stats.global.mean_ess_frac ?? 0) > 0.9
                  ? "good"
                  : (stats.global.mean_ess_frac ?? 0) > 0.5
                    ? "warn"
                    : "bad"
              }
              hint="per-sample, token IS weights"
            />
            <Card
              label="Mean log-ratio"
              value={stats.global.mean_log_ratio.toFixed(4)}
              hint="0 = unbiased"
            />
            <Card label="Total resp tokens" value={stats.global.total_valid_tokens.toLocaleString()} />
            <Card label="Tokens / sample" value={Math.round(stats.global.mean_tokens_per_sample).toLocaleString()} />
          </div>

          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-5 mb-5">
            <div className="text-xs font-semibold text-slate-200 mb-3">
              Per-token log importance ratio (all {stats.global.total_valid_tokens.toLocaleString()} tokens)
            </div>
            <Histogram hist={stats.global.hist} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* per-sample table */}
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs font-semibold text-slate-200">
                  Per-sample (sorted by {sortKey})
                </div>
                <select
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                  className="bg-slate-800 text-slate-300 text-[11px] border border-slate-700 rounded px-1.5 py-0.5"
                >
                  <option value="pearson">worst pearson</option>
                  <option value="ess_frac">worst ESS</option>
                  <option value="max_abs_log_ratio">max |log-ratio|</option>
                  <option value="n_tokens">most tokens</option>
                  <option value="idx">sample index</option>
                </select>
              </div>
              <div className="overflow-y-auto max-h-80 text-[11px] font-mono">
                <table className="w-full">
                  <thead className="text-slate-500 sticky top-0 bg-slate-900/95">
                    <tr className="text-left">
                      <th className="py-1">#</th>
                      <th>rew</th>
                      <th>toks</th>
                      <th>pearson</th>
                      <th>ESS</th>
                      <th>|lr|max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.slice(0, 60).map((s) => (
                      <tr
                        key={s.idx}
                        onClick={() => setSel(s.idx)}
                        className={`cursor-pointer ${sel === s.idx ? "bg-indigo-500/20" : "hover:bg-slate-800/60"}`}
                      >
                        <td className="py-0.5">{s.idx}</td>
                        <td className={s.reward ? "text-emerald-400" : "text-rose-400"}>
                          {s.reward ?? "—"}
                        </td>
                        <td>{s.n_tokens}</td>
                        <td className={(s.pearson ?? 1) < 0.95 ? "text-amber-400" : "text-slate-300"}>
                          {s.pearson?.toFixed(3) ?? "—"}
                        </td>
                        <td className={(s.ess_frac ?? 1) < 0.5 ? "text-rose-400" : "text-slate-300"}>
                          {s.ess_frac != null ? (s.ess_frac * 100).toFixed(0) + "%" : "—"}
                        </td>
                        <td>{s.max_abs_log_ratio?.toFixed(2) ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* selected sample trace */}
            <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4">
              <div className="text-xs font-semibold text-slate-200 mb-2">
                Sample #{sel} — log-ratio along the response
              </div>
              {sel != null && stats.series[sel] ? (
                <>
                  <Trace series={stats.series[sel]} />
                  {(() => {
                    const st = stats.per_sample[sel];
                    return st ? (
                      <div className="grid grid-cols-3 gap-2 mt-3 text-[11px]">
                        <div>
                          <span className="text-slate-500">reward </span>
                          <span className="font-mono">{st.reward ?? "—"}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">tokens </span>
                          <span className="font-mono">{st.n_tokens}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">pearson </span>
                          <span className="font-mono">{st.pearson?.toFixed(4) ?? "—"}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">ESS frac </span>
                          <span className="font-mono">
                            {st.ess_frac != null ? (st.ess_frac * 100).toFixed(1) + "%" : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500">mean lr </span>
                          <span className="font-mono">{st.mean_log_ratio?.toFixed(4) ?? "—"}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">seq lr </span>
                          <span className="font-mono">{st.seq_log_ratio?.toFixed(1) ?? "—"}</span>
                        </div>
                      </div>
                    ) : null;
                  })()}
                  <p className="text-[10px] text-slate-600 mt-3">
                    Downsampled to {stats.series_points} points (mean-pooled over
                    response tokens). Spikes = tokens where the training policy
                    disagrees most with what the rollout actually sampled.
                  </p>
                </>
              ) : (
                <div className="text-xs text-slate-600">select a sample</div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
