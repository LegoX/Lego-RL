import ChartPanel from "../components/Chart";
import MetricCard from "../components/MetricCard";
import ValAnalysis from "./ValAnalysis";
import ValCotTrend from "./ValCotTrend";
import ValFailureModes from "./ValFailureModes";
import type { MetricPoint } from "../types";

interface Props {
  data: MetricPoint[];
  runId?: string | null;
  onOpenTrajectory?: (step: string, task: string) => void;
}

// These charts plot ONLY what the trainer logged. This panel used to swap in
// (and later, overlay) solve rates recomputed from the val dumps on disk, which
// meant the curve you were looking at silently changed meaning depending on
// whether trial dirs happened to still be around. The on-disk recomputation
// still exists — in the ValAnalysis panel below, where it is labelled as such.
// Offline evals injected as a run's val are given `val-core` points in their
// snapshot metrics, so they show up here without needing the disk series.
const VAL_LOG_KEY = "val-core/unknown/reward/mean@1";

export default function ValidationPanel({ data, runId, onOpenTrajectory }: Props) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-100 mb-1">Validation</h2>
      <p className="text-xs text-slate-500 mb-4">
        Validation scores and auxiliary metrics evaluated periodically during training
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard
          label="Val Score (mean@1)"
          metricKey={VAL_LOG_KEY}
          data={data}
          format="percent"
          color="#4a9440"
        />
        <MetricCard
          label="Val Turns (mean)"
          metricKey="val-aux/num_turns/mean"
          data={data}
          format="number"
          color="#199e70"
        />
        <MetricCard
          label="Val Turns (min)"
          metricKey="val-aux/num_turns/min"
          data={data}
          format="int"
          color="#7a6ddb"
        />
        <MetricCard
          label="Val Turns (max)"
          metricKey="val-aux/num_turns/max"
          data={data}
          format="int"
          color="#e0a01a"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel
          title="Validation Reward"
          data={data}
          keys={[VAL_LOG_KEY]}
          colors={["#4a9440"]}
          showArea
        />
        <ChartPanel
          title="Validation Turns"
          data={data}
          keys={[
            "val-aux/num_turns/mean",
            "val-aux/num_turns/max",
            "val-aux/num_turns/min",
          ]}
          colors={["#199e70", "#f5c9b4", "#10684b"]}
        />
        <ChartPanel
          title="Validation Time"
          data={data}
          keys={["rollouter/validate_time"]}
          colors={["#d03b3b"]}
          yAxisLabel="seconds"
        />
      </div>

      <ValCotTrend runId={runId ?? null} />

      <ValAnalysis runId={runId ?? null} onOpenTrajectory={onOpenTrajectory} />

      <ValFailureModes runId={runId ?? null} onOpenTrajectory={onOpenTrajectory} />
    </div>
  );
}
