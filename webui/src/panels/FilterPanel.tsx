import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

export default function FilterPanel({ data }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">
        Trajectory Filter
      </h2>
      <p className="text-xs text-slate-500 mb-3">
        Each rollout gets one <span className="text-slate-300">termination reason</span>. Trajectories
        whose reason is "broken" are <span className="text-slate-300">neutralized</span> — their
        advantage is zeroed so they don't contribute to the loss (batch shape unchanged).
      </p>
      <div className="rounded-lg bg-slate-800/40 border border-slate-700/50 p-3 mb-4 text-[11px] text-slate-400 leading-relaxed">
        <span className="text-rose-300 font-medium">Dropped by default</span> (env's fault, not the
        policy): <code className="text-rose-300">timeout</code>,{" "}
        <code className="text-rose-300">env_setup_failed</code>.{" "}
        <span className="text-emerald-300 font-medium">Kept</span> (real learning signal):{" "}
        <code className="text-emerald-300">agent_completed</code> (normal finish, scored),{" "}
        <code className="text-emerald-300">overlong</code> (hit length budget, truncated),{" "}
        <code className="text-emerald-300">max_turns_reached</code> (hit turn cap).
        <div className="mt-1.5">
          <span className="text-slate-300">invalid_ratio</span> = fraction of the batch dropped
          (reason ∈ drop-list) → wasted rollouts. High/spiking = environment trouble, not learning.
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel
          title="Dropped (invalid) ratio — fraction of batch excluded from loss"
          data={data}
          keys={["trajectory_filter/invalid_ratio"]}
          colors={["#d03b3b"]}
          showArea
        />
        <ChartPanel
          title="Dropped (invalid) count"
          data={data}
          keys={["trajectory_filter/invalid_count"]}
          colors={["#e07070"]}
          showArea
        />
        <ChartPanel
          title="Dropped reasons (count) — excluded from training"
          data={data}
          keys={[
            "trajectory_filter/reason/timeout",
            "trajectory_filter/reason/env_setup_failed",
          ]}
          colors={["#e0a01a", "#d03b3b"]}
        />
        <ChartPanel
          title="Kept reasons (count) — contribute to training"
          data={data}
          keys={[
            "trajectory_filter/reason/agent_completed",
            "trajectory_filter/reason/overlong",
            "trajectory_filter/reason/max_turns_reached",
          ]}
          colors={["#4a9440", "#7a6ddb", "#199e70"]}
        />
      </div>
    </div>
  );
}
