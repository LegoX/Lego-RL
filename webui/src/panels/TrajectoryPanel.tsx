import { useState, useCallback, useEffect, useRef } from "react";
import {
  Shuffle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Trophy,
  XCircle,
  MessageSquare,
  Wrench,
  Eye,
  ChevronsUpDown,
  ChevronsDownUp,
  Download,
  Terminal,
  FileText,
  Package,
} from "lucide-react";
import { useT } from "../i18n";

interface Props {
  runId: string | null;
  // deep-link target set when a task is clicked elsewhere (e.g. Task Grid)
  target?: { step: string; task: string; nonce: number } | null;
}

interface TurnAction {
  name: string;
  arguments: string;
}

interface Turn {
  turn: number;
  thought: string;
  actions: TurnAction[];
  observations: string[];
  usage: { prompt: number; completion: number };
  duration_ms: number | null;
}

interface TrajectoryData {
  run_id: string;
  step: string;
  task: string;
  reward: number | null;
  num_turns: number;
  system_prompt: string;
  problem_statement: string;
  turns: Turn[];
}

interface TaskInfo {
  task: string;
  reward: number | null;
}

function TurnCard({
  turn,
  defaultExpanded,
}: {
  turn: Turn;
  defaultExpanded: boolean;
}) {
  const { t } = useT();
  const [expanded, setExpanded] = useState(defaultExpanded);

  useEffect(() => {
    setExpanded(defaultExpanded);
  }, [defaultExpanded]);

  return (
    <div className="rounded-lg border border-slate-800/60 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-800/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          {expanded ? (
            <ChevronDown size={13} className="text-slate-500" />
          ) : (
            <ChevronRight size={13} className="text-slate-500" />
          )}
          <span className="text-xs font-medium text-slate-300">
            {t("traj.turn")} {turn.turn + 1}
          </span>
          {turn.actions.length > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-amber-400/80">
              <Wrench size={10} />
              {turn.actions.map((a) => a.name).join(", ")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-[10px] text-slate-500">
          {turn.duration_ms != null && (
            <span>{(turn.duration_ms / 1000).toFixed(1)}s</span>
          )}
          <span>
            {turn.usage.completion} {t("traj.tokens")}
          </span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-3 space-y-2.5">
          {turn.thought && (
            <div className="rounded-lg bg-blue-500/[0.06] border border-blue-500/20 p-3">
              <div className="flex items-center gap-1.5 mb-1.5">
                <MessageSquare size={11} className="text-blue-400" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-400">
                  {t("traj.thought")}
                </span>
              </div>
              <pre className="text-xs text-slate-300 whitespace-pre-wrap break-words font-sans leading-relaxed">
                {turn.thought}
              </pre>
            </div>
          )}

          {turn.actions.map((action, ai) => (
            <div key={ai} className="space-y-2">
              <div className="rounded-lg bg-amber-500/[0.06] border border-amber-500/20 p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Wrench size={11} className="text-amber-400" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-400">
                    {t("traj.action")}
                  </span>
                  <span className="text-[10px] font-mono text-amber-300/80 ml-1">
                    {action.name}
                  </span>
                </div>
                <pre className="text-xs text-slate-300 whitespace-pre-wrap break-words font-mono leading-relaxed bg-slate-900/50 rounded p-2 max-h-48 overflow-y-auto">
                  {action.arguments}
                </pre>
              </div>

              {turn.observations[ai] && (
                <div className="rounded-lg bg-emerald-500/[0.06] border border-emerald-500/20 p-3">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Eye size={11} className="text-emerald-400" />
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
                      {t("traj.observation")}
                    </span>
                  </div>
                  <pre className="text-xs text-slate-400 whitespace-pre-wrap break-words font-mono leading-relaxed bg-slate-900/50 rounded p-2 max-h-48 overflow-y-auto">
                    {turn.observations[ai]}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CollapsibleBlock({
  icon: Icon,
  iconClass,
  title,
  content,
  borderClass,
  bgClass,
}: {
  icon: typeof Terminal;
  iconClass: string;
  title: string;
  content: string;
  borderClass: string;
  bgClass: string;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!content) return null;

  const preview =
    content.length > 300 ? content.slice(0, 300) + "..." : content;

  return (
    <div
      className={`rounded-xl border ${borderClass} ${bgClass} overflow-hidden`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-800/20 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon size={14} className={iconClass} />
          <span className={`text-xs font-semibold ${iconClass}`}>{title}</span>
          <span className="text-[10px] text-slate-500">
            ({content.length.toLocaleString()} chars)
          </span>
        </div>
        {expanded ? (
          <ChevronDown size={14} className="text-slate-500" />
        ) : (
          <ChevronRight size={14} className="text-slate-500" />
        )}
      </button>
      {expanded ? (
        <div className="px-4 pb-3">
          <pre className="text-xs text-slate-300 whitespace-pre-wrap break-words font-mono leading-relaxed bg-slate-900/60 rounded-lg p-3 max-h-96 overflow-y-auto">
            {content}
          </pre>
        </div>
      ) : (
        <div className="px-4 pb-3">
          <pre className="text-xs text-slate-500 whitespace-pre-wrap break-words font-mono leading-relaxed">
            {preview}
          </pre>
        </div>
      )}
    </div>
  );
}

function buildMarkdownExport(data: TrajectoryData): string {
  const lines: string[] = [];
  lines.push(`# Trajectory: ${data.task}`);
  lines.push("");
  lines.push(`- **Run**: ${data.run_id}`);
  lines.push(`- **Step**: ${data.step}`);
  lines.push(`- **Reward**: ${data.reward ?? "unknown"}`);
  lines.push(`- **Turns**: ${data.num_turns}`);
  lines.push("");

  if (data.system_prompt) {
    lines.push("## System Prompt");
    lines.push("");
    lines.push("```");
    lines.push(data.system_prompt);
    lines.push("```");
    lines.push("");
  }

  if (data.problem_statement) {
    lines.push("## Problem Statement");
    lines.push("");
    lines.push(data.problem_statement);
    lines.push("");
  }

  lines.push("## Trajectory");
  lines.push("");

  for (const turn of data.turns) {
    lines.push(`### Turn ${turn.turn + 1}`);
    lines.push("");
    if (turn.thought) {
      lines.push("**Thought:**");
      lines.push("");
      lines.push(turn.thought);
      lines.push("");
    }
    for (let ai = 0; ai < turn.actions.length; ai++) {
      const action = turn.actions[ai];
      lines.push(`**Action:** \`${action.name}\``);
      lines.push("");
      lines.push("```json");
      lines.push(action.arguments);
      lines.push("```");
      lines.push("");
      if (turn.observations[ai]) {
        lines.push("**Observation:**");
        lines.push("");
        lines.push("```");
        lines.push(turn.observations[ai]);
        lines.push("```");
        lines.push("");
      }
    }
  }

  return lines.join("\n");
}

export default function TrajectoryPanel({ runId, target }: Props) {
  const { t } = useT();
  const [data, setData] = useState<TrajectoryData | null>(null);
  const [steps, setSteps] = useState<string[]>([]);
  const [tasksInStep, setTasksInStep] = useState<TaskInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingTasks, setLoadingTasks] = useState(false);
  const [downloadingAll, setDownloadingAll] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [allExpanded, setAllExpanded] = useState(false);
  const [selectedStep, setSelectedStep] = useState<string>("");
  const [selectedTask, setSelectedTask] = useState<string>("");
  const [expandKey, setExpandKey] = useState(0);

  // pending deep-link nonce — so async loaders below don't clobber a target
  const deepLinkNonce = useRef<number | null>(null);

  // Load steps when run changes
  useEffect(() => {
    if (!runId) return;
    setSteps([]);
    setTasksInStep([]);
    setData(null);
    // Ignore a reply that arrives after the run changed, otherwise the step list
    // belongs to the run you just navigated away from.
    let stale = false;
    fetch(`/api/runs/${runId}/trials`)
      .then((r) => r.json())
      .then((d) => {
        if (stale) return;
        const s: string[] = d.steps || [];
        setSteps(s);
        // don't auto-pick the first step if a deep-link target is driving us
        if (s.length > 0 && deepLinkNonce.current == null) setSelectedStep(s[0]);
      })
      .catch(() => {
        if (!stale) setSteps([]);
      });
    return () => {
      stale = true;
    };
  }, [runId]);

  // Load tasks when step changes (do NOT clear selectedTask here — that would
  // wipe a deep-linked task; manual step changes clear it in the select below)
  useEffect(() => {
    if (!runId || !selectedStep) return;
    setTasksInStep([]);
    setLoadingTasks(true);
    let stale = false;
    fetch(`/api/runs/${runId}/step-tasks?step=${selectedStep}`)
      .then((r) => r.json())
      .then((d) => {
        if (!stale) setTasksInStep(d.tasks || []);
      })
      .catch(() => {
        if (!stale) setTasksInStep([]);
      })
      .finally(() => {
        if (!stale) setLoadingTasks(false);
      });
    return () => {
      stale = true;
    };
  }, [runId, selectedStep]);

  const loadTrajectory = useCallback(
    async (step?: string, task?: string) => {
      if (!runId) return;
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        if (step) params.set("step", step);
        if (task) params.set("task", task);
        const res = await fetch(
          `/api/runs/${runId}/trajectory?${params.toString()}`
        );
        const d = await res.json();
        if (!res.ok || d.error)
          throw new Error(d.error || `HTTP ${res.status}`);
        setData(d);
        // keep the selectors in sync with what actually resolved (e.g. a
        // deep-linked task_id prefix resolves to a full trial name)
        if (d.step) setSelectedStep(d.step);
        if (d.task) setSelectedTask(d.task);
        setAllExpanded(false);
        setExpandKey((k) => k + 1);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
        setData(null);
      } finally {
        setLoading(false);
      }
    },
    [runId]
  );

  // react to a deep-link target (task clicked in another panel)
  useEffect(() => {
    if (!target || !runId) return;
    deepLinkNonce.current = target.nonce;
    setSelectedStep(target.step);
    setSelectedTask(target.task);
    loadTrajectory(target.step, target.task);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.nonce]);

  const handleLoad = () => {
    loadTrajectory(selectedStep, selectedTask || undefined);
  };

  const handleRandom = () => {
    loadTrajectory(selectedStep, undefined);
  };

  const toggleAll = () => {
    setAllExpanded((v) => !v);
    setExpandKey((k) => k + 1);
  };

  const downloadTrajectory = useCallback(() => {
    if (!data) return;
    const md = buildMarkdownExport(data);
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trajectory-${data.task}-${data.step}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data]);

  const downloadAllInStep = useCallback(async () => {
    if (!runId || !selectedStep) return;
    setDownloadingAll(true);
    try {
      const res = await fetch(
        `/api/runs/${runId}/step-export?step=${selectedStep}`
      );
      const d = await res.json();
      if (!res.ok || d.error) throw new Error(d.error || `HTTP ${res.status}`);

      const lines: string[] = [];
      lines.push(
        `# All Trajectories: ${runId} / ${selectedStep}`
      );
      lines.push("");
      lines.push(`Total: ${d.total} tasks`);
      lines.push("");
      lines.push("---");
      lines.push("");

      for (const traj of d.trajectories) {
        lines.push(`# ${traj.task}`);
        lines.push("");
        lines.push(`- **Reward**: ${traj.reward ?? "unknown"}`);
        lines.push(`- **Turns**: ${traj.num_turns}`);
        lines.push("");

        if (traj.problem_statement) {
          lines.push("## Problem Statement");
          lines.push("");
          lines.push(traj.problem_statement);
          lines.push("");
        }

        lines.push("## Trajectory");
        lines.push("");
        for (const turn of traj.turns) {
          lines.push(`### Turn ${turn.turn + 1}`);
          lines.push("");
          if (turn.thought) {
            lines.push("**Thought:**");
            lines.push("");
            lines.push(turn.thought);
            lines.push("");
          }
          for (let ai = 0; ai < turn.actions.length; ai++) {
            const action = turn.actions[ai];
            lines.push(`**Action:** \`${action.name}\``);
            lines.push("");
            lines.push("```json");
            lines.push(action.arguments);
            lines.push("```");
            lines.push("");
            if (turn.observations[ai]) {
              lines.push("**Observation:**");
              lines.push("");
              lines.push("```");
              lines.push(turn.observations[ai]);
              lines.push("```");
              lines.push("");
            }
          }
        }
        lines.push("---");
        lines.push("");
      }

      const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `trajectories-${runId}-${selectedStep}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // silent — bulk download is best-effort
    } finally {
      setDownloadingAll(false);
    }
  }, [runId, selectedStep]);

  const solvedCount = tasksInStep.filter(
    (t) => t.reward != null && t.reward > 0
  ).length;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            {t("traj.title")}
          </h2>
          <p className="text-xs text-slate-500">{t("traj.desc")}</p>
        </div>
      </div>

      {/* Controls bar */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3 mb-4 flex items-center gap-2 flex-wrap">
        {/* Step selector */}
        <select
          value={selectedStep}
          onChange={(e) => {
            // manual step change: drop the deep-link and reset task selection
            deepLinkNonce.current = null;
            setSelectedTask("");
            setSelectedStep(e.target.value);
          }}
          disabled={steps.length === 0}
          className="bg-slate-800 text-slate-300 text-xs border border-slate-700 rounded-lg px-2 py-1.5 focus:outline-none focus:border-indigo-500 font-mono"
        >
          {steps.length === 0 && <option value="">--</option>}
          {steps.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {/* Task selector */}
        <select
          value={selectedTask}
          onChange={(e) => setSelectedTask(e.target.value)}
          disabled={tasksInStep.length === 0}
          className="bg-slate-800 text-slate-300 text-xs border border-slate-700 rounded-lg px-2 py-1.5 focus:outline-none focus:border-indigo-500 font-mono max-w-[280px] truncate"
        >
          <option value="">
            {t("traj.allTasks")} ({tasksInStep.length} {t("traj.taskCount")})
          </option>
          {tasksInStep.map((ti) => (
            <option key={ti.task} value={ti.task}>
              {ti.reward != null && ti.reward > 0 ? "✓ " : "✗ "}
              {ti.task}
            </option>
          ))}
        </select>

        {loadingTasks && (
          <Loader2 size={14} className="text-slate-500 animate-spin" />
        )}

        {/* Task stats badge */}
        {tasksInStep.length > 0 && !loadingTasks && (
          <span className="text-[10px] text-slate-500">
            <span className="text-emerald-400">{solvedCount}</span>
            <span className="text-slate-600">/</span>
            {tasksInStep.length}
          </span>
        )}

        <div className="flex-1" />

        {/* Load specific task */}
        <button
          onClick={handleLoad}
          disabled={loading || !runId || !selectedStep || (!selectedTask && tasksInStep.length === 0)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500 text-white text-xs font-medium hover:bg-indigo-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? (
            <Loader2 size={13} className="animate-spin" />
          ) : (
            <FileText size={13} />
          )}
          {selectedTask ? t("traj.load") : t("traj.random")}
        </button>

        {/* Random button (always random) */}
        {selectedTask && (
          <button
            onClick={handleRandom}
            disabled={loading || !runId}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 transition-colors disabled:opacity-40"
          >
            <Shuffle size={13} />
            {t("traj.random")}
          </button>
        )}

        {/* Download all in step */}
        <button
          onClick={downloadAllInStep}
          disabled={downloadingAll || tasksInStep.length === 0}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 text-xs hover:bg-slate-800 transition-colors disabled:opacity-40"
          title={t("traj.downloadAll")}
        >
          {downloadingAll ? (
            <Loader2 size={13} className="animate-spin" />
          ) : (
            <Package size={13} />
          )}
          {downloadingAll ? t("traj.downloadingAll") : t("traj.downloadAll")}
        </button>
      </div>

      {/* Async step-naming hint */}
      <p className="text-[10px] text-slate-600 mb-4 -mt-2">
        {t("traj.stepHint")}
      </p>

      {/* Error */}
      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4 text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}

      {/* No data */}
      {!data && !loading && !error && (
        <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-8 text-center text-sm text-slate-500">
          {t("traj.noData")}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-6 flex items-center justify-center gap-3">
          <Loader2 size={18} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">{t("traj.loading")}</span>
        </div>
      )}

      {/* Trajectory display */}
      {data && !loading && (
        <div>
          {/* Meta bar */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3 mb-4 flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-4 text-xs flex-wrap">
              <div>
                <span className="text-slate-500">{t("traj.step")}: </span>
                <span className="font-mono text-slate-300">{data.step}</span>
              </div>
              <div className="max-w-[400px]">
                <span className="text-slate-500">{t("traj.task")}: </span>
                <span className="font-mono text-slate-300 break-all">
                  {data.task}
                </span>
              </div>
              <div className="flex items-center gap-1">
                {data.reward != null && data.reward > 0 ? (
                  <Trophy size={13} className="text-emerald-400" />
                ) : (
                  <XCircle size={13} className="text-rose-400" />
                )}
                <span className="text-slate-500">{t("traj.reward")}: </span>
                <span
                  className={`font-mono font-semibold ${data.reward != null && data.reward > 0 ? "text-emerald-400" : "text-rose-400"}`}
                >
                  {data.reward ?? "?"}
                </span>
              </div>
              <div>
                <span className="text-slate-500">
                  {data.num_turns} {t("traj.turns")}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={downloadTrajectory}
                className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-indigo-400 transition-colors"
                title={t("traj.download")}
              >
                <Download size={13} />
                {t("traj.download")}
              </button>
              <span className="text-slate-700">|</span>
              <button
                onClick={toggleAll}
                className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-200 transition-colors"
              >
                {allExpanded ? (
                  <ChevronsDownUp size={13} />
                ) : (
                  <ChevronsUpDown size={13} />
                )}
                {allExpanded ? t("traj.collapseAll") : t("traj.expandAll")}
              </button>
            </div>
          </div>

          {/* System Prompt & Problem Statement */}
          <div className="space-y-2 mb-4">
            <CollapsibleBlock
              icon={Terminal}
              iconClass="text-violet-400"
              title={t("traj.systemPrompt")}
              content={data.system_prompt}
              borderClass="border-violet-500/20"
              bgClass="bg-violet-500/[0.03]"
            />
            <CollapsibleBlock
              icon={FileText}
              iconClass="text-cyan-400"
              title={t("traj.problemStatement")}
              content={data.problem_statement}
              borderClass="border-cyan-500/20"
              bgClass="bg-cyan-500/[0.03]"
            />
          </div>

          {/* Turns */}
          <div className="space-y-2 max-h-[calc(100vh-280px)] overflow-y-auto pr-1">
            {data.turns.map((turn) => (
              <TurnCard
                key={`${expandKey}-${turn.turn}`}
                turn={turn}
                defaultExpanded={
                  allExpanded ||
                  turn.turn === 0 ||
                  turn.turn === data.turns.length - 1
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
