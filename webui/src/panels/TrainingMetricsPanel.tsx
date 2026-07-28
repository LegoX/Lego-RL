import RewardsPanel from "./RewardsPanel";
import PolicyPanel from "./PolicyPanel";
import StabilityPanel from "./StabilityPanel";
import SequencePanel from "./SequencePanel";
import AgentPanel from "./AgentPanel";
import FilterPanel from "./FilterPanel";
import PerformancePanel from "./PerformancePanel";
import AsyncPanel from "./AsyncPanel";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
}

// All per-step metric-trend panels in one place — these are real-time training
// curves (no analysis), so they live together. Analysis views (Rollout Dist,
// Task Grid, Trajectory, …) stay as separate panels.
const SECTIONS: { id: string; label: string; render: (d: MetricPoint[]) => React.ReactNode }[] = [
  { id: "rewards", label: "Rewards & Scores", render: (d) => <RewardsPanel data={d} /> },
  { id: "agent", label: "Agent Loop", render: (d) => <AgentPanel data={d} /> },
  { id: "policy", label: "Policy & Training", render: (d) => <PolicyPanel data={d} /> },
  { id: "stability", label: "Training Stability", render: (d) => <StabilityPanel data={d} /> },
  { id: "sequences", label: "Sequences", render: (d) => <SequencePanel data={d} /> },
  { id: "filter", label: "Trajectory Filter", render: (d) => <FilterPanel data={d} /> },
  { id: "performance", label: "Performance", render: (d) => <PerformancePanel data={d} /> },
  { id: "async", label: "Async Training Health", render: (d) => <AsyncPanel data={d} /> },
];

export default function TrainingMetricsPanel({ data }: Props) {
  const jump = (id: string) => {
    document
      .getElementById(`sec-${id}`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div>
      {/* sticky in-page jump nav */}
      <div className="sticky top-0 z-10 -mx-4 md:-mx-6 px-4 md:px-6 py-2 mb-4 bg-slate-950/85 backdrop-blur border-b border-slate-800/60 flex flex-wrap gap-1.5">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => jump(s.id)}
            className="text-[11px] px-2.5 py-1 rounded-full border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="space-y-10">
        {SECTIONS.map((s) => (
          <section key={s.id} id={`sec-${s.id}`} className="scroll-mt-16">
            {s.render(data)}
          </section>
        ))}
      </div>
    </div>
  );
}
