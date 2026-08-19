import { useMemo } from "react";
import ChartPanel, { MinMaxChart } from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

// Response length alone cannot tell "the agent explored more" from "the agent
// got wordier" — both push it up, and they have opposite fixes. Dividing by the
// turn count separates them: tokens-per-turn flat while length rises = more
// turns (exploration); tokens-per-turn rising while turns are flat or falling =
// verbosity drift, the classic no-KL-anchor failure where the policy writes
// more per turn without acting more. Derived from two keys already on the page.
const PER_TURN = "derived/tokens_per_turn";

export default function SequencePanel({ data }: Props) {
  const derived = useMemo(
    () =>
      data.map((p) => {
        const rl = p["response_length/mean"];
        const tn = p["num_turns/mean"];
        if (typeof rl !== "number" || typeof tn !== "number" || !tn) return p;
        return { ...p, [PER_TURN]: rl / tn };
      }),
    [data],
  );
  const hasPerTurn = useMemo(
    () => derived.some((p) => typeof p[PER_TURN] === "number"),
    [derived],
  );

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">
        Sequence Analysis
      </h2>
      <p className="text-xs text-slate-500 mb-4">
        Prompt and response lengths, truncation rates, and abort ratios
      </p>
      {hasPerTurn && (
        <div className="mb-4">
          <ChartPanel
            title="Output per TURN (verbosity drift) — response_length/mean ÷ num_turns/mean"
            data={derived}
            keys={[PER_TURN, "num_turns/mean"]}
            colors={["#e0a01a", "#199e70"]}
            showArea
          />
          <p className="text-[11px] text-slate-500 mt-1.5 leading-relaxed">
            Read the two lines together.{" "}
            <span className="text-slate-300">Per-turn flat, turns rising</span> = the
            agent is exploring more.{" "}
            <span className="text-amber-400">Per-turn rising, turns flat or falling</span>{" "}
            = verbosity drift: the policy is writing more per turn without acting
            more. The second is not fixed by raising the turn cap — it wants a KL
            anchor or a length penalty.
          </p>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MinMaxChart
          title="Response Length"
          data={data}
          meanKey="response_length/mean"
          maxKey="response_length/max"
          minKey="response_length/min"
          color="#c96a45"
        />
        <MinMaxChart
          title="Prompt Length"
          data={data}
          meanKey="prompt_length/mean"
          maxKey="prompt_length/max"
          minKey="prompt_length/min"
          color="#199e70"
        />
        <ChartPanel
          title="Clip Ratio (Truncation Rate)"
          data={data}
          keys={["response_length/clip_ratio", "prompt_length/clip_ratio"]}
          colors={["#d03b3b", "#e0a01a"]}
          yAxisLabel="ratio"
        />
        <ChartPanel
          title="Aborted Responses"
          data={data}
          keys={["response/aborted_ratio"]}
          colors={["#c73434"]}
          yAxisLabel="ratio"
          showArea
        />
        <MinMaxChart
          title="Response Length (non-aborted)"
          data={data}
          meanKey="response_length_non_aborted/mean"
          maxKey="response_length_non_aborted/max"
          minKey="response_length_non_aborted/min"
          color="#7a6ddb"
        />
        <ChartPanel
          title="Sequence Balance (global_seqlen)"
          data={data}
          keys={[
            "global_seqlen/mean",
            "global_seqlen/balanced_min",
            "global_seqlen/balanced_max",
          ]}
          colors={["#4a9440", "#9ccb8e", "#3f8f2f"]}
        />
      </div>
    </div>
  );
}
