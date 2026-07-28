import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

export default function PerformancePanel({ data }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">Performance</h2>
      <p className="text-xs text-slate-500 mb-4">
        Throughput, per-step timing, MFU, and stage-level breakdown
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel
          title="Throughput (tokens/s/GPU)"
          data={data}
          keys={["perf/throughput"]}
          colors={["#10b981"]}
          showArea
        />
        <ChartPanel
          title="Time per Step"
          data={data}
          keys={["perf/time_per_step"]}
          colors={["#f59e0b"]}
          yAxisLabel="seconds"
          showArea
        />
        <ChartPanel
          title="Total Tokens per Step"
          data={data}
          keys={["perf/total_num_tokens"]}
          colors={["#6366f1"]}
        />
        <ChartPanel
          title="Model FLOPs Utilization (MFU)"
          data={data}
          keys={[
            "perf/mfu/actor",
            "perf/mfu/critic",
            "perf/mfu/actor_infer",
          ]}
          colors={["#6366f1", "#10b981", "#f59e0b"]}
        />
        <ChartPanel
          title="Stage Timing (seconds)"
          data={data}
          keys={[
            "timing_s/gen",
            "timing_s/ref",
            "timing_s/values",
            "timing_s/adv",
            "timing_s/update_actor",
            "timing_s/update_critic",
          ]}
          colors={[
            "#6366f1",
            "#10b981",
            "#f59e0b",
            "#f43f5e",
            "#8b5cf6",
            "#06b6d4",
          ]}
        />
        <ChartPanel
          title="Per-Token Timing (ms)"
          data={data}
          keys={[
            "timing_per_token_ms/gen",
            "timing_per_token_ms/ref",
            "timing_per_token_ms/update_actor",
          ]}
          colors={["#6366f1", "#10b981", "#8b5cf6"]}
        />
      </div>
    </div>
  );
}
