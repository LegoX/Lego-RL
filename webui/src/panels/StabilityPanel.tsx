import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

export default function StabilityPanel({ data }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">
        Training Stability
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        Off-policy correction, importance sampling, rollout–training mismatch,
        and variance diagnostics
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel
          title="Rollout Correction: KL & Perplexity"
          data={data}
          keys={[
            "rollout_corr/kl",
            "rollout_corr/k3_kl",
            "rollout_corr/training_ppl",
          ]}
          colors={["#f43f5e", "#fb7185", "#f59e0b"]}
        />
        <ChartPanel
          title="Log-Perplexity Diff"
          data={data}
          keys={[
            "rollout_corr/log_ppl_diff",
            "rollout_corr/log_ppl_diff_max",
            "rollout_corr/log_ppl_diff_min",
          ]}
          colors={["#8b5cf6", "#c4b5fd", "#7c3aed"]}
        />
        <ChartPanel
          title="Chi-Squared Test"
          data={data}
          keys={["rollout_corr/chi2_token", "rollout_corr/chi2_seq"]}
          colors={["#06b6d4", "#0891b2"]}
        />
        <ChartPanel
          title="Importance Sampling Weights"
          data={data}
          keys={[
            "rollout_is_mean",
            "rollout_is_max",
            "rollout_is_min",
            "rollout_is_std",
          ]}
          colors={["#6366f1", "#818cf8", "#4f46e5", "#a5b4fc"]}
        />
        <ChartPanel
          title="IS Per-Sequence Stats"
          data={data}
          keys={[
            "rollout_is_seq_mean",
            "rollout_is_seq_max",
            "rollout_is_seq_min",
          ]}
          colors={["#10b981", "#6ee7b7", "#059669"]}
        />
        <ChartPanel
          title="IS Diagnostics"
          data={data}
          keys={[
            "rollout_is_oob_ratio",
            "rollout_is_ratio_fraction_high",
            "rollout_is_ratio_fraction_low",
            "rollout_is_eff_sample_size",
          ]}
          colors={["#f43f5e", "#f59e0b", "#84cc16", "#06b6d4"]}
        />
        <ChartPanel
          title="Rollout Probs Diff"
          data={data}
          keys={[
            "training/rollout_probs_diff_mean",
            "training/rollout_probs_diff_max",
            "training/rollout_probs_diff_std",
          ]}
          colors={["#8b5cf6", "#c4b5fd", "#a78bfa"]}
        />
        <ChartPanel
          title="Variance Proxy"
          data={data}
          keys={[
            "variance_proxy/proxy1_signal_strength",
            "variance_proxy/proxy2_total_power",
            "variance_proxy/proxy3_pure_noise",
          ]}
          colors={["#10b981", "#f59e0b", "#f43f5e"]}
        />
      </div>
    </div>
  );
}
