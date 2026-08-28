import MetricCard from "../components/MetricCard";
import ConfigSection from "./ConfigSection";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
  runId?: string | null;
}

export default function OverviewPanel({ data, runId }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-4">Overview</h2>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
        <MetricCard
          label="Step"
          metricKey="training/global_step"
          data={data}
          format="int"
          color="#c96a45"
        />
        <MetricCard
          label="Epoch"
          metricKey="training/epoch"
          data={data}
          format="int"
          color="#7a6ddb"
        />
        <MetricCard
          label="Reward (mean)"
          metricKey="critic/rewards/mean"
          data={data}
          color="#4a9440"
        />
        <MetricCard
          label="Score (mean)"
          metricKey="critic/score/mean"
          data={data}
          color="#199e70"
        />
        <MetricCard
          label="Actor Loss"
          metricKey="actor/pg_loss"
          data={data}
          color="#e0a01a"
        />
        <MetricCard
          label="Entropy"
          metricKey="actor/entropy"
          data={data}
          color="#b8506e"
        />
        <MetricCard
          label="PPO KL"
          metricKey="actor/ppo_kl"
          data={data}
          color="#d03b3b"
        />
        <MetricCard
          label="Clip Frac"
          metricKey="actor/pg_clipfrac"
          data={data}
          format="percent"
          color="#e0a01a"
        />
        <MetricCard
          label="Grad Norm"
          metricKey="actor/grad_norm"
          data={data}
          color="#8aa02a"
        />
        <MetricCard
          label="Response Len"
          metricKey="response_length/mean"
          data={data}
          format="int"
          color="#c96a45"
        />
        <MetricCard
          label="Throughput"
          metricKey="perf/throughput"
          data={data}
          format="number"
          color="#4a9440"
        />
        <MetricCard
          label="Step Time"
          metricKey="perf/time_per_step"
          data={data}
          format="duration"
          color="#e0a01a"
        />
      </div>
      <ConfigSection runId={runId ?? null} />
    </div>
  );
}
