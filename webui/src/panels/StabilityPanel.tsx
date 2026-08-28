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
          colors={["#d03b3b", "#e07070", "#e0a01a"]}
        />
        <ChartPanel
          title="Log-Perplexity Diff"
          data={data}
          keys={[
            "rollout_corr/log_ppl_diff",
            "rollout_corr/log_ppl_diff_max",
            "rollout_corr/log_ppl_diff_min",
          ]}
          colors={["#7a6ddb", "#b3aae8", "#5a4bc0"]}
        />
        <ChartPanel
          title="Chi-Squared Test"
          data={data}
          keys={["rollout_corr/chi2_token", "rollout_corr/chi2_seq"]}
          colors={["#199e70", "#147f5c"]}
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
          colors={["#c96a45", "#dd8a63", "#a83f1d", "#efa07c"]}
        />
        <ChartPanel
          title="IS Per-Sequence Stats"
          data={data}
          keys={[
            "rollout_is_seq_mean",
            "rollout_is_seq_max",
            "rollout_is_seq_min",
          ]}
          colors={["#4a9440", "#9ccb8e", "#3f8f2f"]}
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
          colors={["#d03b3b", "#e0a01a", "#8aa02a", "#199e70"]}
        />
        <ChartPanel
          title="Rollout Probs Diff"
          data={data}
          keys={[
            "training/rollout_probs_diff_mean",
            "training/rollout_probs_diff_max",
            "training/rollout_probs_diff_std",
          ]}
          colors={["#7a6ddb", "#b3aae8", "#9085e9"]}
        />
        <ChartPanel
          title="Variance Proxy"
          data={data}
          keys={[
            "variance_proxy/proxy1_signal_strength",
            "variance_proxy/proxy2_total_power",
            "variance_proxy/proxy3_pure_noise",
          ]}
          colors={["#4a9440", "#e0a01a", "#d03b3b"]}
        />
      </div>
    </div>
  );
}
