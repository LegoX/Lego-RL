import type { MetricPoint } from "../types";

export function withTrainingStep(point: MetricPoint): MetricPoint {
  const step = point["training/global_step"];
  return Number.isInteger(step) && step >= 0 ? { ...point, step } : point;
}

export function trainingStepCount(points: MetricPoint[]): number {
  const steps = points.map(p => p["training/global_step"]).filter(s => Number.isInteger(s) && s >= 0);
  return steps.length ? steps.reduce((a, b) => Math.max(a, b), 0) : new Set(points.map(p => p.step)).size;
}
