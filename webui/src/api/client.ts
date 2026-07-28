import type { RunInfo, MetricPoint, MetricsData, IngestJob } from "../types";

const BASE = "/api";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as { error?: string }).error || `${res.status} ${res.statusText}`);
  return data as T;
}

export async function ingestSnapshot(
  path: string,
  name: string,
  logPath?: string,
): Promise<{ job_id: string }> {
  return postJson(`${BASE}/snapshots/ingest`, {
    path,
    name,
    ...(logPath ? { log_path: logPath } : {}),
  });
}

export async function fetchIngestJob(jobId: string): Promise<IngestJob> {
  return fetchJson(`${BASE}/snapshots/jobs/${jobId}`);
}

export async function deleteSnapshot(runId: string): Promise<{ ok: boolean }> {
  return postJson(`${BASE}/snapshots/delete`, { run_id: runId });
}

export async function fetchRuns(): Promise<RunInfo[]> {
  return fetchJson(`${BASE}/runs`);
}

export async function fetchMetrics(
  runId: string,
  keys?: string[],
): Promise<MetricsData> {
  const params = new URLSearchParams();
  if (keys?.length) params.set("keys", keys.join(","));
  const qs = params.toString();
  return fetchJson(`${BASE}/runs/${runId}/metrics${qs ? `?${qs}` : ""}`);
}

export async function fetchLatest(runId: string): Promise<MetricPoint | null> {
  return fetchJson(`${BASE}/runs/${runId}/latest`);
}

export async function fetchConfig(): Promise<{
  log_dir: string;
  wandb_entity: string;
  wandb_project: string;
  data_source: string;
}> {
  return fetchJson(`${BASE}/config`);
}

// These are generated per-config by the backend, so unlike the UI's own strings
// they cannot be translated by key in src/i18n.ts — the server sends both and
// the client picks. The *_en fields are absent when the label is already
// language-neutral (max_model_len, ppo_epochs, ...), so callers fall back.
export interface RunConfigItem {
  label: string;
  label_en?: string;
  value: string | number;
  value_en?: string | number;
  hint?: string;
  wide?: boolean;
}
export interface RunConfigSection {
  title: string;
  title_en?: string;
  note?: string;
  note_en?: string;
  items: RunConfigItem[];
}
export interface RunConfigNode {
  ip: string;
  roles: string[];
}
export interface RunConfig {
  available: boolean;
  error?: string;
  sections?: RunConfigSection[];
  nodes?: RunConfigNode[];
  raw?: Record<string, unknown>;
}

export async function fetchRunConfig(runId: string): Promise<RunConfig> {
  return fetchJson(`${BASE}/runs/${runId}/config`);
}
