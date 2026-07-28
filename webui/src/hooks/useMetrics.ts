import { useState, useEffect, useCallback, useRef } from "react";
import type { RunInfo, MetricsData } from "../types";
import { fetchRuns, fetchMetrics } from "../api/client";

export function useRuns(refreshInterval: number) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchRuns();
      setRuns(data);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, refreshInterval * 1000);
    return () => clearInterval(id);
  }, [load, refreshInterval]);

  return { runs, error, reload: load };
}

export function useMetrics(runId: string | null, refreshInterval: number) {
  const [data, setData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prevRunId = useRef(runId);

  const load = useCallback(async () => {
    if (!runId) return;
    // Capture the run this request is for. A poll for the previous run can still
    // be in flight when you switch, and runs differ by an order of magnitude in
    // how long /metrics takes (a warm live run answers in ~0.15s, a cold
    // snapshot in ~2s). Without this guard the older, faster response lands last
    // and repaints the chart with the run you just navigated away from — which
    // reads as "the chart didn't switch".
    const reqRun = runId;
    if (prevRunId.current !== runId) {
      setData(null);
      prevRunId.current = runId;
    }
    try {
      setLoading(true);
      const result = await fetchMetrics(reqRun);
      if (prevRunId.current !== reqRun) return; // superseded; drop it
      setData(result);
      setError(null);
    } catch (e: unknown) {
      if (prevRunId.current !== reqRun) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (prevRunId.current === reqRun) setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    load();
    const id = setInterval(load, refreshInterval * 1000);
    return () => clearInterval(id);
  }, [load, refreshInterval]);

  return { data, loading, error, reload: load };
}
