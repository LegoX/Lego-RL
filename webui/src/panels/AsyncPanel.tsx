import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

export default function AsyncPanel({ data }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">
        Async Training Health
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        Fully-async rollout/trainer pipeline: sample staleness, queue pressure,
        idle ratios, processing-time latency, and partial-rollout behaviour
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel
          title="Sample Generation & Staleness"
          data={data}
          keys={[
            "fully_async/count/total_generated_samples",
            "fully_async/count/staleness_samples",
            "fully_async/count/stale_trajectory_processed",
            "fully_async/count/dropped_stale_samples",
          ]}
          colors={["#4a9440", "#e0a01a", "#e07070", "#d03b3b"]}
        />
        <ChartPanel
          title="Param Version"
          data={data}
          keys={["fully_async/count/current_param_version"]}
          colors={["#c96a45"]}
          showArea
        />
        <ChartPanel
          title="Queue Sizes"
          data={data}
          keys={[
            "fully_async/monitor/active_tasks_size",
            "fully_async/monitor/queue/mq_queue_size",
            "fully_async/monitor/queue/pending_queue_size",
          ]}
          colors={["#199e70", "#147f5c", "#e0a01a"]}
        />
        <ChartPanel
          title="Processing Time (s)"
          data={data}
          keys={[
            "fully_async/processing_time/avg",
            "fully_async/processing_time/tp50",
            "fully_async/processing_time/tp95",
            "fully_async/processing_time/tp99",
            "fully_async/processing_time/max",
            "fully_async/processing_time/min",
          ]}
          colors={[
            "#7a6ddb",
            "#9085e9",
            "#b3aae8",
            "#5a4bc0",
            "#cfc8f0",
            "#4a3aa7",
          ]}
        />
        <ChartPanel
          title="Rollouter vs Trainer Idle Ratio"
          data={data}
          keys={[
            "fully_async/rollouter/idle_ratio",
            "fully_async/trainer/idle_ratio",
          ]}
          colors={["#d03b3b", "#199e70"]}
        />
        <ChartPanel
          title="Rollouter & Wait Time (s)"
          data={data}
          keys={[
            "fully_async/rollouter/active_time",
            "fully_async/rollouter/version_time",
            "fully_async/total_wait_time",
          ]}
          colors={["#4a9440", "#8aa02a", "#e0a01a"]}
        />
        <ChartPanel
          title="Partial Rollouts"
          data={data}
          keys={[
            "fully_async/partial/partial_ratio",
            "fully_async/partial/total_partial_num",
            "fully_async/partial/max_partial_span",
          ]}
          colors={["#7a6ddb", "#199e70", "#e0a01a"]}
        />
        <ChartPanel
          title="Static Config"
          data={data}
          keys={[
            "fully_async/static/required_samples",
            "fully_async/static/max_required_samples",
            "fully_async/static/max_concurrent_samples",
            "fully_async/static/max_queue_size",
            "fully_async/static/staleness_threshold",
          ]}
          colors={["#c96a45", "#dd8a63", "#4a9440", "#199e70", "#d03b3b"]}
        />
      </div>
    </div>
  );
}
