import type { MetricPoint } from "../types";

// Derived display fields only: preserve every original logged metric.
export function withRewardMetrics(point: MetricPoint): MetricPoint {
  const result = { ...point };
  const finite = (value: unknown): value is number =>
    typeof value === "number" && Number.isFinite(value);
  const version = point["reward_stats/schema_version"];
  const segment = point["reward_stats/segment_mean"];
  if (version === 2 && finite(segment)) {
    result["dashboard/reward/segment_mean"] = segment;
  } else if (version === undefined) {
    // Legacy harness means are usable only when the log proves one segment
    // per trial and no compaction; otherwise their weighting is ambiguous.
    const harnessMeans = Object.keys(point).filter(key =>
      /^timing_s\/agent_loop\/harness_reward\/[^/]+\/mean$/.test(key));
    const selected = harnessMeans.map(key => key.replace("/harness_reward/", "/harness_selected/"));
    // Each harness reward is zero-filled for rows handled by other harnesses.
    // Their means share the same denominator, so SUM (not mean) combines them.
    const completeHarnesses = harnessMeans.length > 0 && harnessMeans.every(key => finite(point[key]))
      && selected.every(key => finite(point[key]) && point[key] >= 0 && point[key] <= 1)
      && Math.abs(selected.reduce((sum, key) => sum + point[key], 0) - 1) < 1e-6;
    if (completeHarnesses
        && point["timing_s/agent_loop/proxy_num_segments/min"] === 1
        && point["timing_s/agent_loop/proxy_num_segments/max"] === 1
        && point["timing_s/agent_loop/proxy_num_compactions/max"] === 0) {
      result["dashboard/reward/segment_mean"] = harnessMeans.reduce((sum, key) => sum + point[key], 0);
      result["dashboard/reward/trial_mean"] = result["dashboard/reward/segment_mean"];
    }
  }
  const trial = point["reward_stats/trial_mean"];
  if (version === 2 && finite(trial)) {
    result["dashboard/reward/trial_mean"] = trial;
  }
  const real = point["fully_async/compaction/real_rows"];
  const padding = point["fully_async/compaction/padding_rows"];
  const ratio = point["reward_stats/padding_ratio"];
  if (version === 2 && finite(ratio) && ratio >= 0 && ratio <= 1) {
    result["dashboard/reward/padding_ratio"] = ratio;
  } else if (finite(real) && finite(padding) && real >= 0 && padding >= 0 && real + padding > 0) {
    result["dashboard/reward/padding_ratio"] = padding / (real + padding);
  }
  return result;
}
