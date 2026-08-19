import { useState, useEffect, useCallback, useMemo } from "react";
import {
  Activity,
  LineChart,
  RefreshCw,
  ChevronRight,
  Radio,
  Server,
  ScrollText,
  FlaskConical,
  Sparkles,
  Settings,
  ClipboardCheck,
  Route,
  GitCompareArrows,
  BarChart3,
  Grid3x3,
  ListChecks,
  Sun,
  Moon,
  Languages,
  Upload,
  Sigma,
} from "lucide-react";
import { useRuns, useMetrics } from "./hooks/useMetrics";
import type { RunInfo } from "./types";
import OverviewPanel from "./panels/OverviewPanel";
import TrainingMetricsPanel from "./panels/TrainingMetricsPanel";
import LogsPanel from "./panels/LogsPanel";
import ExplorerPanel from "./panels/ExplorerPanel";
import ValidationPanel from "./panels/ValidationPanel";
import AnalysisPanel from "./panels/AnalysisPanel";
import TrajectoryPanel from "./panels/TrajectoryPanel";
import ComparePanel from "./panels/ComparePanel";
import RolloutDistPanel from "./panels/RolloutDistPanel";
import TaskGridPanel from "./panels/TaskGridPanel";
import EpisodePanel from "./panels/EpisodePanel";
import ImportPanel from "./panels/ImportPanel";
import TokenISPanel from "./panels/TokenISPanel";
import SettingsPanel from "./panels/SettingsPanel";
import {
  I18nContext,
  getSavedLang,
  saveLang,
  createT,
  type Lang,
} from "./i18n";

const PANELS = [
  { id: "overview", labelKey: "nav.overview", icon: Activity },
  { id: "metrics", labelKey: "nav.metrics", icon: LineChart },
  { id: "episode", labelKey: "nav.episode", icon: ListChecks },
  { id: "rollout-dist", labelKey: "nav.rolloutDist", icon: BarChart3 },
  { id: "token-is", labelKey: "nav.tokenIS", icon: Sigma },
  { id: "task-grid", labelKey: "nav.taskGrid", icon: Grid3x3 },
  { id: "validation", labelKey: "nav.validation", icon: ClipboardCheck },
  { id: "compare", labelKey: "nav.compare", icon: GitCompareArrows },
  { id: "analysis", labelKey: "nav.analysis", icon: Sparkles },
  { id: "trajectory", labelKey: "nav.trajectory", icon: Route },
  { id: "logs", labelKey: "nav.logs", icon: ScrollText },
  { id: "explorer", labelKey: "nav.explorer", icon: FlaskConical },
  { id: "import", labelKey: "nav.import", icon: Upload },
  { id: "settings", labelKey: "nav.settings", icon: Settings },
] as const;

function RunSelector({
  runs,
  selected,
  onSelect,
}: {
  runs: RunInfo[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-1">
      {runs.map((r) => (
        <button
          key={r.id}
          onClick={() => onSelect(r.id)}
          className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors ${
            selected === r.id
              ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
              : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-300 border border-transparent"
          }`}
        >
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                r.state === "running"
                  ? "bg-emerald-400 animate-pulse"
                  : r.state === "snapshot"
                    ? "bg-indigo-400"
                    : r.state === "finished"
                      ? "bg-slate-500"
                      : "bg-amber-500"
              }`}
            />
            <span
              className="font-mono break-all whitespace-normal leading-snug"
              title={r.name}
            >
              {r.name}
            </span>
          </div>
          <div className="flex items-center gap-2 mt-1 ml-4">
            <span className="text-[10px] text-slate-500 uppercase">
              {r.source}
            </span>
            <span className="text-[10px] text-slate-600">{r.state}</span>
          </div>
        </button>
      ))}
      {runs.length === 0 && (
        <p className="text-xs text-slate-600 px-3 py-4 text-center">
          No runs found
        </p>
      )}
    </div>
  );
}

type Theme = "dark" | "light";

// Deliberately not the old "harbor-theme" key. Every previous visitor has "dark"
// written there (the effect below persists on first paint, so nobody has an
// unset value), which would silently pin them to the old default forever. A new
// key resets everyone to light once, and still remembers their choice after.
const THEME_KEY = "lego-rl-theme";

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem(THEME_KEY);
    return saved === "dark" ? "dark" : "light";
  });

  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggle };
}

export default function App() {
  const [activePanel, setActivePanel] = useState("overview");
  const [selectedRun, setSelectedRun] = useState("");
  // deep-link target for the Trajectory panel (set when a task is clicked in
  // the Task Grid)
  const [trajTarget, setTrajTarget] = useState<{
    step: string;
    task: string;
    nonce: number;
  } | null>(null);
  const openTrajectory = useCallback((step: string, task: string) => {
    setTrajTarget({ step, task, nonce: Date.now() });
    setActivePanel("trajectory");
  }, []);
  const [refreshInterval, setRefreshInterval] = useState(15);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // Draggable sidebar width (px), persisted. Long run names wrap, so the user
  // can widen the panel to read full names instead of squinting at a truncation.
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const s = Number(localStorage.getItem("harbor-sidebar-w"));
    return s >= 200 && s <= 640 ? s : 256;
  });
  useEffect(() => {
    localStorage.setItem("harbor-sidebar-w", String(sidebarWidth));
  }, [sidebarWidth]);
  const startResize = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = sidebarWidth;
      const onMove = (ev: MouseEvent) =>
        setSidebarWidth(Math.min(640, Math.max(200, startW + ev.clientX - startX)));
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        document.body.style.userSelect = "";
      };
      document.body.style.userSelect = "none";
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [sidebarWidth],
  );
  const { theme, toggle: toggleTheme } = useTheme();
  const [lang, setLang] = useState<Lang>(getSavedLang);
  const t = useMemo(() => createT(lang), [lang]);
  const toggleLang = useCallback(() => {
    const next = lang === "en" ? "zh" : "en";
    setLang(next);
    saveLang(next);
  }, [lang]);

  // Old runs whose trial dirs were cleaned up (analysis panels empty) are
  // hidden by default; this toggle brings them back — nothing is deleted.
  const [showEmpty, setShowEmpty] = useState(false);

  const { runs, error: runsError, reload: reloadRuns } = useRuns(refreshInterval);
  const visibleRuns = useMemo(
    () => (showEmpty ? runs : runs.filter((r) => !r.analysis_empty)),
    [runs, showEmpty],
  );
  const hiddenCount = useMemo(
    () => runs.filter((r) => r.analysis_empty).length,
    [runs],
  );
  const { data, loading, error: metricsError } = useMetrics(
    selectedRun || null,
    refreshInterval,
  );

  useEffect(() => {
    if (!selectedRun && runs.length > 0) {
      // Prefer a live run, then any real (non-snapshot) run, so importing a
      // snapshot never changes the default selection for normal usage.
      const running = visibleRuns.find((r) => r.state === "running");
      const realRun = visibleRuns.find((r) => r.source !== "snapshot");
      setSelectedRun(running?.id || realRun?.id || visibleRuns[0]?.id || runs[0].id);
    }
  }, [runs, visibleRuns, selectedRun]);

  const metrics = data?.metrics || [];
  const availableKeys = data?.available_keys || [];

  const handleRefresh = useCallback(() => {
    window.location.reload();
  }, []);

  return (
    <I18nContext.Provider value={{ lang, t }}>
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside
        className="flex-shrink-0 overflow-hidden relative"
        style={{ width: sidebarOpen ? sidebarWidth : 0 }}
      >
        <div
          className="h-full flex flex-col bg-slate-900/50 border-r border-slate-800/60"
          style={{ width: sidebarWidth }}
        >
          {/* Drag-to-resize handle */}
          <div
            onMouseDown={startResize}
            className="absolute top-0 right-0 h-full w-1.5 cursor-col-resize hover:bg-indigo-500/40 z-10"
            title="Drag to resize"
          />
          {/* Logo */}
          <div className="px-4 py-4 border-b border-slate-800/60">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center">
                <span className="text-white text-xs font-bold font-mono">
                  RL
                </span>
              </div>
              <div>
                <h1 className="text-sm font-semibold text-slate-100">
                  {t("app.title")}
                </h1>
                <p className="text-[10px] text-slate-500">{t("app.subtitle")}</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="px-2 py-3 space-y-0.5">
            {PANELS.map(({ id, labelKey, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActivePanel(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activePanel === id
                    ? "bg-slate-800/80 text-slate-100"
                    : "text-slate-400 hover:text-slate-300 hover:bg-slate-800/40"
                }`}
              >
                <Icon size={15} />
                {t(labelKey)}
              </button>
            ))}
          </nav>

          {/* Runs */}
          <div className="flex-1 overflow-y-auto border-t border-slate-800/60 px-2 py-3">
            <div className="flex items-center gap-1.5 px-3 mb-2">
              <Server size={12} className="text-slate-500" />
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                {t("app.runs")}
              </span>
            </div>
            <RunSelector
              runs={visibleRuns}
              selected={selectedRun}
              onSelect={setSelectedRun}
            />
            {hiddenCount > 0 && (
              <button
                onClick={() => setShowEmpty((v) => !v)}
                className="w-full text-left text-[10px] text-slate-500 hover:text-slate-300 px-3 mt-2 transition-colors"
              >
                {showEmpty
                  ? `Hide ${hiddenCount} stale tasks with no data`
                  : `Show ${hiddenCount} stale tasks with no data`}
              </button>
            )}
            {runsError && (
              <p className="text-[10px] text-rose-400 px-3 mt-2">
                {runsError}
              </p>
            )}
          </div>

          {/* Settings */}
          <div className="px-4 py-3 border-t border-slate-800/60">
            <div className="flex items-center justify-between">
              <label className="text-[10px] text-slate-500 uppercase tracking-wider">
                {t("app.refresh")}
              </label>
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="bg-slate-800 text-slate-300 text-xs border border-slate-700 rounded px-1.5 py-0.5 focus:outline-none focus:border-indigo-500"
              >
                <option value={5}>5s</option>
                <option value={10}>10s</option>
                <option value={15}>15s</option>
                <option value={30}>30s</option>
                <option value={60}>60s</option>
              </select>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-12 flex-shrink-0 flex items-center justify-between px-4 border-b border-slate-800/60 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-slate-400 hover:text-slate-200 transition-colors"
            >
              <ChevronRight
                size={16}
                className={`transform transition-transform ${sidebarOpen ? "rotate-180" : ""}`}
              />
            </button>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-slate-400">
                {t(PANELS.find((p) => p.id === activePanel)?.labelKey ?? "")}
              </span>
              {selectedRun && (
                <>
                  <span className="text-slate-600">/</span>
                  <span className="text-slate-300 font-mono text-xs">
                    {selectedRun}
                  </span>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {loading && (
              <RefreshCw size={14} className="text-indigo-400 animate-spin" />
            )}
            {data?.run?.state === "running" && (
              <span className="flex items-center gap-1.5 text-xs text-emerald-400">
                <Radio size={12} className="animate-pulse" />
                {t("app.live")}
              </span>
            )}
            {metricsError && (
              <span className="text-xs text-rose-400">{metricsError}</span>
            )}
            <span className="text-xs text-slate-500 font-mono">
              {metrics.length} {t("app.steps")}
            </span>
            <button
              onClick={toggleLang}
              className="text-slate-400 hover:text-slate-200 transition-colors text-xs font-medium"
              title={t("app.langToggle")}
            >
              <Languages size={14} />
            </button>
            <button
              onClick={toggleTheme}
              className="text-slate-400 hover:text-slate-200 transition-colors"
              title={theme === "dark" ? t("app.lightMode") : t("app.darkMode")}
            >
              {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <button
              onClick={handleRefresh}
              className="text-slate-400 hover:text-slate-200 transition-colors"
              title={t("app.hardRefresh")}
            >
              <RefreshCw size={14} />
            </button>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {activePanel === "overview" && (
            <OverviewPanel data={metrics} runId={selectedRun || null} />
          )}
          {activePanel === "metrics" && <TrainingMetricsPanel data={metrics} />}
          {activePanel === "episode" && <EpisodePanel data={metrics} />}
          {activePanel === "rollout-dist" && (
            <RolloutDistPanel runId={selectedRun || null} data={metrics} />
          )}
          {activePanel === "token-is" && (
            <TokenISPanel
              runId={selectedRun || null}
              data={metrics}
              availableKeys={availableKeys}
            />
          )}
          {activePanel === "task-grid" && (
            <TaskGridPanel
              runId={selectedRun || null}
              onOpenTrajectory={openTrajectory}
            />
          )}
          {activePanel === "validation" && (
            <ValidationPanel
              data={metrics}
              runId={selectedRun || null}
              onOpenTrajectory={openTrajectory}
            />
          )}
          {activePanel === "compare" && <ComparePanel runs={runs} />}
          {activePanel === "analysis" && (
            <AnalysisPanel runId={selectedRun || null} />
          )}
          {activePanel === "trajectory" && (
            <TrajectoryPanel runId={selectedRun || null} target={trajTarget} />
          )}
          {activePanel === "logs" && (
            <LogsPanel
              runId={selectedRun || null}
              refreshInterval={refreshInterval}
            />
          )}
          {activePanel === "explorer" && (
            <ExplorerPanel data={metrics} availableKeys={availableKeys} />
          )}
          {activePanel === "import" && (
            <ImportPanel
              runs={runs}
              onSelectRun={(id) => {
                reloadRuns();
                setSelectedRun(id);
                setActivePanel("rollout-dist");
              }}
              onChanged={reloadRuns}
            />
          )}
          {activePanel === "settings" && <SettingsPanel />}
        </div>
      </main>
    </div>
    </I18nContext.Provider>
  );
}
