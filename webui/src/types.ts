export interface RunInfo {
  id: string;
  name: string;
  state: "running" | "finished" | "crashed" | "unknown" | "snapshot";
  created_at: string;
  source: "log" | "wandb" | "snapshot";
  summary?: SnapshotSummary;
  // true when the run's on-disk trial dirs are gone, so its analysis panels
  // (Rollout/Task Grid/Trajectory/Validation) render empty. Hidden by default.
  analysis_empty?: boolean;
}

export interface SnapshotSummary {
  // "kind" distinguishes the two import sources: "run_dir" = referenced real
  // harbor run dir (subset of fields); otherwise a materialized 3-file bundle.
  kind?: string;
  format?: string;
  display_name?: string;
  num_samples: number;
  num_groups: number;
  expected_n: number;
  mean_reward: number;
  pass_rate?: number;
  solved?: number;
  num_repos: number;
  correct_out_of_n_hist?: number[];
  zero_advantage_groups: number;
  resp_len: { min?: number; p50: number | null; p90?: number; max: number | null; mean?: number };
  mean_logprob: number | null;
  termination_reasons?: Record<string, number>;
  reward_source: string;
  has_token_file: boolean;
  token_file?: string | null;
  files?: { jsonl: string | null; trajectories: string; rollout_output: string | null };
}

export interface IngestJob {
  status: "queued" | "running" | "done" | "error" | "unknown";
  name?: string;
  done?: number;
  total?: number;
  message?: string;
  run_id?: string;
  error?: string;
  summary?: SnapshotSummary;
}

export interface MetricPoint {
  step: number;
  [key: string]: number;
}

export interface MetricsData {
  run: RunInfo;
  metrics: MetricPoint[];
  available_keys: string[];
}

export interface PanelConfig {
  id: string;
  title: string;
  icon: string;
  metrics: ChartConfig[];
}

export interface ChartConfig {
  title: string;
  keys: string[];
  colors?: string[];
  yAxisLabel?: string;
  format?: "number" | "percent" | "duration" | "int";
}

export type DataSource = "auto" | "log" | "wandb";

export interface DashboardSettings {
  dataSource: DataSource;
  refreshInterval: number;
  wandbEntity: string;
  wandbProject: string;
  wandbApiKey: string;
  logDir: string;
  selectedRun: string;
}
