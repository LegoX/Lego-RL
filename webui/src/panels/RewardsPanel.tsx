import { useMemo } from "react";
import ChartPanel, { MinMaxChart } from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

export default function RewardsPanel({ data }: Props) {
  // `score` is the raw verifier output; `rewards` is score after the
  // KL-in-reward penalty. They are identical unless algorithm.use_kl_in_reward
  // is enabled, so collapse the duplicate charts when they match.
  const scoreEqualsReward = useMemo(() => {
    let compared = 0;
    for (const p of data) {
      const s = p["critic/score/mean"];
      const r = p["critic/rewards/mean"];
      if (s === undefined || r === undefined) continue;
      compared++;
      if (Math.abs(s - r) > 1e-9) return false;
    }
    return compared > 0;
  }, [data]);

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">Rewards & Scores</h2>
      <p className="text-xs text-slate-500 mb-4">
        Reward signal from the verifier, advantage estimates, and return values
        {scoreEqualsReward && (
          <span className="ml-1 text-slate-400">
            · rewards ≡ scores (KL not applied in reward)
          </span>
        )}
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {scoreEqualsReward ? (
          <MinMaxChart
            title="Score / Reward"
            data={data}
            meanKey="critic/score/mean"
            maxKey="critic/score/max"
            minKey="critic/score/min"
            color="#4a9440"
          />
        ) : (
          <>
            <MinMaxChart
              title="Score"
              data={data}
              meanKey="critic/score/mean"
              maxKey="critic/score/max"
              minKey="critic/score/min"
              color="#4a9440"
            />
            <MinMaxChart
              title="Rewards"
              data={data}
              meanKey="critic/rewards/mean"
              maxKey="critic/rewards/max"
              minKey="critic/rewards/min"
              color="#199e70"
            />
          </>
        )}
        <MinMaxChart
          title="Advantages"
          data={data}
          meanKey="critic/advantages/mean"
          maxKey="critic/advantages/max"
          minKey="critic/advantages/min"
          color="#7a6ddb"
        />
        <MinMaxChart
          title="Returns"
          data={data}
          meanKey="critic/returns/mean"
          maxKey="critic/returns/max"
          minKey="critic/returns/min"
          color="#e0a01a"
        />
        {!scoreEqualsReward && (
          <ChartPanel
            title="Score & Reward Trend"
            data={data}
            keys={["critic/score/mean", "critic/rewards/mean"]}
            colors={["#4a9440", "#199e70"]}
            showArea
          />
        )}
        <ChartPanel
          title="Advantage Distribution"
          data={data}
          keys={[
            "critic/advantages/mean",
            "critic/advantages/max",
            "critic/advantages/min",
          ]}
          colors={["#7a6ddb", "#b3aae8", "#5a4bc0"]}
        />
      </div>
    </div>
  );
}
