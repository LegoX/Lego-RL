import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

export default function PolicyPanel({ data }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">
        Policy & Training
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        Actor loss, entropy, KL divergence, gradient norms, and learning rate
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel
          title="Policy Gradient Loss"
          data={data}
          keys={["actor/pg_loss", "actor/loss"]}
          colors={["#f59e0b", "#f97316"]}
          showArea
        />
        <ChartPanel
          title="Entropy"
          data={data}
          keys={["actor/entropy"]}
          colors={["#ec4899"]}
          showArea
        />
        <ChartPanel
          title="KL Divergence"
          data={data}
          keys={["actor/ppo_kl", "actor/kl_loss"]}
          colors={["#f43f5e", "#fb7185"]}
        />
        <ChartPanel
          title="Clip Fraction"
          data={data}
          keys={["actor/pg_clipfrac", "actor/pg_clipfrac_lower"]}
          colors={["#f59e0b", "#fbbf24"]}
        />
        <ChartPanel
          title="Gradient Norm"
          data={data}
          keys={["actor/grad_norm"]}
          colors={["#84cc16"]}
        />
        <ChartPanel
          title="Learning Rate"
          data={data}
          keys={["actor/lr"]}
          colors={["#6366f1"]}
        />
      </div>
    </div>
  );
}
