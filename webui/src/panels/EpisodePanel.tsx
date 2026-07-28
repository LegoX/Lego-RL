import { useMemo } from "react";
import ChartPanel, { MinMaxChart } from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

// latest defined value of a metric key across steps
function latest(data: MetricPoint[], key: string): number | null {
  for (let i = data.length - 1; i >= 0; i--) {
    const v = data[i][key];
    if (typeof v === "number" && !Number.isNaN(v)) return v;
  }
  return null;
}

function fmt(v: number | null, opts?: { pct?: boolean; digits?: number }): string {
  if (v === null) return "—";
  if (opts?.pct) return (v * 100).toFixed(opts.digits ?? 1) + "%";
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + "k";
  return v.toFixed(opts?.digits ?? (Number.isInteger(v) ? 0 : 2));
}

function Card({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-xl font-semibold ${accent ?? "text-slate-100"}`}>{value}</div>
    </div>
  );
}

export default function EpisodePanel({ data }: Props) {
  const s = useMemo(
    () => ({
      avgTurns: latest(data, "num_turns/mean"),
      maxTurns: latest(data, "num_turns/max"),
      avgLen: latest(data, "response_length/mean"),
      maxLen: latest(data, "response_length/max"),
      avgTokens: latest(data, "timing_s/agent_loop/proxy_total_completion_tokens/mean"),
      avgDuration: latest(data, "timing_s/agent_loop/harbor_total/mean"),
      maxDuration: latest(data, "timing_s/agent_loop/harbor_total/max"),
      aborted: latest(data, "response/aborted_ratio"),
      lenClip: latest(data, "response_length/clip_ratio"),
      invalidRatio: latest(data, "trajectory_filter/invalid_ratio"),
    }),
    [data],
  );

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">Episode Summary</h2>
      <p className="text-xs text-slate-500 mb-4">
        Per-rollout turns, length, tokens and duration, plus abnormal-exit and
        truncation rates. Cards show the latest step.
      </p>

      {/* Summary cards (latest step) */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3 mb-5">
        <Card label="Avg turns" value={fmt(s.avgTurns, { digits: 1 })} />
        <Card label="Max turns" value={fmt(s.maxTurns)} />
        <Card label="Avg resp length" value={fmt(s.avgLen)} />
        <Card label="Max resp length" value={fmt(s.maxLen)} />
        <Card label="Avg completion tokens" value={fmt(s.avgTokens)} />
        <Card label="Avg duration (s)" value={fmt(s.avgDuration, { digits: 1 })} />
        <Card label="Max duration (s)" value={fmt(s.maxDuration, { digits: 1 })} />
        <Card
          label="Aborted ratio"
          value={fmt(s.aborted, { pct: true })}
          accent={s.aborted && s.aborted > 0.05 ? "text-rose-400" : undefined}
        />
        <Card
          label="Length-clip (trunc)"
          value={fmt(s.lenClip, { pct: true })}
          accent={s.lenClip && s.lenClip > 0.05 ? "text-amber-400" : undefined}
        />
        <Card
          label="Filtered ratio"
          value={fmt(s.invalidRatio, { pct: true })}
          accent={s.invalidRatio && s.invalidRatio > 0.1 ? "text-amber-400" : undefined}
        />
      </div>

      {/* Trend charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MinMaxChart
          title="Num Turns"
          data={data}
          meanKey="num_turns/mean"
          maxKey="num_turns/max"
          minKey="num_turns/min"
          color="#f59e0b"
        />
        <MinMaxChart
          title="Response Length (tokens)"
          data={data}
          meanKey="response_length/mean"
          maxKey="response_length/max"
          minKey="response_length/min"
          color="#06b6d4"
        />
        <MinMaxChart
          title="Episode Duration (s)"
          data={data}
          meanKey="timing_s/agent_loop/harbor_total/mean"
          maxKey="timing_s/agent_loop/harbor_total/max"
          minKey="timing_s/agent_loop/harbor_total/min"
          color="#6366f1"
        />
        <ChartPanel
          title="Completion Tokens / response"
          data={data}
          keys={[
            "timing_s/agent_loop/proxy_total_completion_tokens/mean",
            "timing_s/agent_loop/proxy_total_completion_tokens/max",
          ]}
          colors={["#8b5cf6", "#c4b5fd"]}
        />
        <ChartPanel
          title="Abnormal Exits (ratio / count)"
          data={data}
          keys={[
            "response/aborted_ratio",
            "trajectory_filter/reason/env_setup_failed",
            "trajectory_filter/reason/timeout",
          ]}
          colors={["#f43f5e", "#fb7185", "#f59e0b"]}
        />
        <ChartPanel
          title="Truncation (length-clip / overlong / max-turns)"
          data={data}
          keys={[
            "response_length/clip_ratio",
            "prompt_length/clip_ratio",
            "trajectory_filter/reason/overlong",
            "trajectory_filter/reason/max_turns_reached",
          ]}
          colors={["#f59e0b", "#fbbf24", "#8b5cf6", "#ec4899"]}
        />
      </div>
    </div>
  );
}
