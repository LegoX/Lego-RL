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
          color="#6366f1"
        />
        <MetricCard
          label="Epoch"
          metricKey="training/epoch"
          data={data}
          format="int"
          color="#8b5cf6"
        />
        <MetricCard
          label="Reward (mean)"
          metricKey="critic/rewards/mean"
          data={data}
          color="#10b981"
        />
        <MetricCard
          label="Score (mean)"
          metricKey="critic/score/mean"
          data={data}
          color="#06b6d4"
        />
        <MetricCard
          label="Actor Loss"
          metricKey="actor/pg_loss"
          data={data}
          color="#f59e0b"
        />
        <MetricCard
          label="Entropy"
          metricKey="actor/entropy"
          data={data}
          color="#ec4899"
        />
        <MetricCard
          label="PPO KL"
          metricKey="actor/ppo_kl"
          data={data}
          color="#f43f5e"
        />
        <MetricCard
          label="Clip Frac"
          metricKey="actor/pg_clipfrac"
          data={data}
          format="percent"
          color="#f59e0b"
        />
        <MetricCard
          label="Grad Norm"
          metricKey="actor/grad_norm"
          data={data}
          color="#84cc16"
        />
        <MetricCard
          label="Response Len"
          metricKey="response_length/mean"
          data={data}
          format="int"
          color="#6366f1"
        />
        <MetricCard
          label="Throughput"
          metricKey="perf/throughput"
          data={data}
          format="number"
          color="#10b981"
        />
        <MetricCard
          label="Step Time"
          metricKey="perf/time_per_step"
          data={data}
          format="duration"
          color="#f59e0b"
        />
      </div>
      <ConfigSection runId={runId ?? null} />
    </div>
  );
}
