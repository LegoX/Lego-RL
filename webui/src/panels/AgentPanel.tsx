import { useMemo } from "react";
import ChartPanel, { MinMaxChart } from "../components/Chart";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

// ---------------------------------------------------------------------------
// Several agent-loop timers are structurally dead depending on the scaffold,
// and a permanently-flat zero line reads as "the agent did no work" rather than
// "verl cannot see this".
//
// With the harbor scaffolds the agent runs INSIDE the sandbox pod and executes
// its own tools, so verl never observes an individual tool call:
//   * `tool_calls` / `compute_score` — only ticked by verl's own
//     tool_agent_loop; builtin_cc_agent_loop.py and builtin_swe_agent_loop.py
//     initialize them to 0.0 and never write them again.
//   * `harbor_agent_tool_call` / `harbor_agent_model_inference` — only the
//     OpenHands agents populate these (installed_/image_mounted_openhands_ai);
//     the CC loop forwards them from agent_result.metadata, and the claude-code
//     agent never writes them, so the forward resolves to 0.
// Model-inference and in-pod tool time are fused into `harbor_agent_execute`.
// The closest verl-side view of per-round tool work is `proxy_tool_gap_*`.
//
// So: hide any chart whose series are all zero, and say which ones were hidden
// instead of silently dropping them. On an OpenHands run the same charts light
// up again on their own.
// ---------------------------------------------------------------------------

function allZero(data: MetricPoint[], keys: string[]): boolean {
  let seen = false;
  for (const p of data) {
    for (const k of keys) {
      const v = p[k];
      if (typeof v === "number" && !Number.isNaN(v)) {
        seen = true;
        if (v !== 0) return false;
      }
    }
  }
  return seen; // all-zero only counts if we actually saw values
}

function hasAny(data: MetricPoint[], keys: string[]): boolean {
  return data.some((p) => keys.some((k) => p[k] !== undefined));
}

const MM = (base: string) => [`${base}/mean`, `${base}/max`, `${base}/min`];

// proxy_tool_gap_* was added to the agent loop later than the other proxy
// counters, so runs from before it exists should not render an empty card.
const GAP_KEYS = [
  "timing_s/agent_loop/proxy_tool_gap_mean/mean",
  "timing_s/agent_loop/proxy_tool_gap_p90/mean",
  "timing_s/agent_loop/proxy_tool_gap_max/max",
];

export default function AgentPanel({ data }: Props) {
  // title -> the keys it draws, for the charts that can go structurally dead
  const dead = useMemo(() => {
    const candidates: [string, string[]][] = [
      ["Model Inference (s)", MM("timing_s/agent_loop/harbor_agent_model_inference")],
      ["Tool Calls (s)", MM("timing_s/agent_loop/harbor_agent_tool_call")],
      ["Compute Score (s)", MM("timing_s/agent_loop/compute_score")],
      ["Tool Calls — verl builtin loop (s)", MM("timing_s/agent_loop/tool_calls")],
    ];
    return new Set(candidates.filter(([, ks]) => allZero(data, ks)).map(([t]) => t));
  }, [data]);

  const show = (title: string) => !dead.has(title);

  // Slowest-trial breakdown: same problem, one chart, so filter its series.
  const slowestKeys = useMemo(
    () =>
      [
        "timing_s/agent_loop/slowest/harbor_total",
        "timing_s/agent_loop/slowest/harbor_agent_execute",
        "timing_s/agent_loop/slowest/harbor_agent_model_inference",
        "timing_s/agent_loop/slowest/harbor_agent_tool_call",
        "timing_s/agent_loop/slowest/harbor_env_setup",
        "timing_s/agent_loop/slowest/harbor_verifier",
      ].filter((k) => !allZero(data, [k])),
    [data],
  );
  const SLOWEST_COLORS: Record<string, string> = {
    "timing_s/agent_loop/slowest/harbor_total": "#6366f1",
    "timing_s/agent_loop/slowest/harbor_agent_execute": "#10b981",
    "timing_s/agent_loop/slowest/harbor_agent_model_inference": "#f59e0b",
    "timing_s/agent_loop/slowest/harbor_agent_tool_call": "#ec4899",
    "timing_s/agent_loop/slowest/harbor_env_setup": "#06b6d4",
    "timing_s/agent_loop/slowest/harbor_verifier": "#f43f5e",
  };

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">Agent Loop</h2>
      <p className="text-xs text-slate-500 mb-4">
        Harbor sandbox execution, model inference, tool calls, and per-trial
        timing breakdown
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MinMaxChart
          title="Harbor Total Time (s)"
          data={data}
          meanKey="timing_s/agent_loop/harbor_total/mean"
          maxKey="timing_s/agent_loop/harbor_total/max"
          minKey="timing_s/agent_loop/harbor_total/min"
          color="#6366f1"
        />
        <MinMaxChart
          title="Agent Execute (s) — inference + in-pod tools, fused"
          data={data}
          meanKey="timing_s/agent_loop/harbor_agent_execute/mean"
          maxKey="timing_s/agent_loop/harbor_agent_execute/max"
          minKey="timing_s/agent_loop/harbor_agent_execute/min"
          color="#10b981"
        />
        {show("Model Inference (s)") && (
          <MinMaxChart
            title="Model Inference (s)"
            data={data}
            meanKey="timing_s/agent_loop/harbor_agent_model_inference/mean"
            maxKey="timing_s/agent_loop/harbor_agent_model_inference/max"
            minKey="timing_s/agent_loop/harbor_agent_model_inference/min"
            color="#f59e0b"
          />
        )}
        {show("Tool Calls (s)") && (
          <MinMaxChart
            title="Tool Calls (s)"
            data={data}
            meanKey="timing_s/agent_loop/harbor_agent_tool_call/mean"
            maxKey="timing_s/agent_loop/harbor_agent_tool_call/max"
            minKey="timing_s/agent_loop/harbor_agent_tool_call/min"
            color="#ec4899"
          />
        )}
        <MinMaxChart
          title="Env Setup (s)"
          data={data}
          meanKey="timing_s/agent_loop/harbor_env_setup/mean"
          maxKey="timing_s/agent_loop/harbor_env_setup/max"
          minKey="timing_s/agent_loop/harbor_env_setup/min"
          color="#06b6d4"
        />
        <MinMaxChart
          title="Agent Setup (s)"
          data={data}
          meanKey="timing_s/agent_loop/harbor_agent_setup/mean"
          maxKey="timing_s/agent_loop/harbor_agent_setup/max"
          minKey="timing_s/agent_loop/harbor_agent_setup/min"
          color="#8b5cf6"
        />
        <MinMaxChart
          title="Verifier (s)"
          data={data}
          meanKey="timing_s/agent_loop/harbor_verifier/mean"
          maxKey="timing_s/agent_loop/harbor_verifier/max"
          minKey="timing_s/agent_loop/harbor_verifier/min"
          color="#f43f5e"
        />
        {show("Compute Score (s)") && (
          <MinMaxChart
            title="Compute Score (s)"
            data={data}
            meanKey="timing_s/agent_loop/compute_score/mean"
            maxKey="timing_s/agent_loop/compute_score/max"
            minKey="timing_s/agent_loop/compute_score/min"
            color="#84cc16"
          />
        )}
        <MinMaxChart
          title="Generate Sequences (s)"
          data={data}
          meanKey="timing_s/agent_loop/generate_sequences/mean"
          maxKey="timing_s/agent_loop/generate_sequences/max"
          minKey="timing_s/agent_loop/generate_sequences/min"
          color="#06b6d4"
        />
        <MinMaxChart
          title="Num Turns"
          data={data}
          meanKey="num_turns/mean"
          maxKey="num_turns/max"
          minKey="num_turns/min"
          color="#f59e0b"
        />
        {/* The real per-trial call count. The old "Tool Calls Count" charted
            timing_s/.../tool_calls, which is a TIMER in seconds and is dead on
            harbor scaffolds — this is the metric people actually wanted. */}
        <MinMaxChart
          title="LLM Calls per Trial (≈ tool-call rounds)"
          data={data}
          meanKey="timing_s/agent_loop/proxy_num_calls/mean"
          maxKey="timing_s/agent_loop/proxy_num_calls/max"
          minKey="timing_s/agent_loop/proxy_num_calls/min"
          color="#10b981"
        />
        {/* Gap between consecutive LLM calls = one round of in-pod tool
            execution + reasoning. The closest verl-side view of tool time when
            the agent runs its own tools inside the sandbox. */}
        {hasAny(data, GAP_KEYS) && (
          <ChartPanel
            title="Tool Gap per Round (s) — in-pod tool + reasoning between LLM calls"
            data={data}
            keys={GAP_KEYS}
            colors={["#10b981", "#f59e0b", "#f43f5e"]}
          />
        )}
        {show("Tool Calls — verl builtin loop (s)") && (
          <MinMaxChart
            title="Tool Calls — verl builtin loop (s)"
            data={data}
            meanKey="timing_s/agent_loop/tool_calls/mean"
            maxKey="timing_s/agent_loop/tool_calls/max"
            minKey="timing_s/agent_loop/tool_calls/min"
            color="#6366f1"
          />
        )}
        <ChartPanel
          title="Proxy Tokens (completion)"
          data={data}
          keys={[
            "timing_s/agent_loop/proxy_total_completion_tokens/mean",
            "timing_s/agent_loop/proxy_total_completion_tokens/max",
            "timing_s/agent_loop/proxy_total_completion_tokens/min",
          ]}
          colors={["#8b5cf6", "#c4b5fd", "#7c3aed"]}
        />
        <ChartPanel
          title="Proxy Tokens (prompt)"
          data={data}
          keys={[
            "timing_s/agent_loop/proxy_total_prompt_tokens/mean",
            "timing_s/agent_loop/proxy_total_prompt_tokens/max",
            "timing_s/agent_loop/proxy_total_prompt_tokens/min",
          ]}
          colors={["#06b6d4", "#67e8f9", "#0891b2"]}
        />
        <ChartPanel
          title="Preempted & Aborted"
          data={data}
          keys={[
            "timing_s/agent_loop/num_preempted/mean",
            "timing_s/agent_loop/proxy_num_aborts/mean",
            "timing_s/agent_loop/proxy_num_preempted/mean",
          ]}
          colors={["#f43f5e", "#f59e0b", "#ec4899"]}
        />
        {slowestKeys.length > 0 && (
          <ChartPanel
            title="Slowest Trial Breakdown (s)"
            data={data}
            keys={slowestKeys}
            colors={slowestKeys.map((k) => SLOWEST_COLORS[k])}
          />
        )}
      </div>

      {dead.size > 0 && (
        <div className="mt-4 rounded-lg bg-slate-800/40 border border-slate-700/50 p-3 text-[11px] text-slate-400 leading-relaxed">
          <span className="text-slate-300 font-medium">
            Hid {dead.size} metrics that are always 0
          </span>{" "}
          ({[...dead].join(", ")}). This does not mean the agent did nothing — under
          this scaffold the agent runs its own tools inside the sandbox pod, so verl
          never sees an individual tool call:{" "}
          <code className="text-slate-300">tool_calls</code> /{" "}
          <code className="text-slate-300">compute_score</code> are only timed inside
          verl's own tool_agent_loop, and{" "}
          <code className="text-slate-300">harbor_agent_tool_call</code> /{" "}
          <code className="text-slate-300">harbor_agent_model_inference</code> are only
          populated by OpenHands-family agents, not by claude-code.
          <div className="mt-1.5">
            Inference time and in-pod tool time are folded into{" "}
            <span className="text-emerald-300">Agent Execute</span>. For per-round tool
            cost see <span className="text-emerald-300">Tool Gap per Round</span>, and for
            call counts see{" "}
            <span className="text-emerald-300">LLM Calls per Trial</span>. These charts
            appear on their own for a run using an OpenHands scaffold.
          </div>
        </div>
      )}
    </div>
  );
}
