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
          colors={["#e0a01a", "#e07a3c"]}
          showArea
        />
        <ChartPanel
          title="Entropy"
          data={data}
          keys={["actor/entropy"]}
          colors={["#b8506e"]}
          showArea
        />
        <ChartPanel
          title="KL Divergence"
          data={data}
          keys={["actor/ppo_kl", "actor/kl_loss"]}
          colors={["#d03b3b", "#e07070"]}
        />
        <ChartPanel
          title="Clip Fraction"
          data={data}
          keys={["actor/pg_clipfrac", "actor/pg_clipfrac_lower"]}
          colors={["#e0a01a", "#fab219"]}
        />
        <ChartPanel
          title="Gradient Norm"
          data={data}
          keys={["actor/grad_norm"]}
          colors={["#8aa02a"]}
        />
        <ChartPanel
          title="Learning Rate"
          data={data}
          keys={["actor/lr"]}
          colors={["#c96a45"]}
        />
      </div>
    </div>
  );
}
