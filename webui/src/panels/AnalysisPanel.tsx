import { useState, useEffect, useCallback, useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Sparkles,
  Loader2,
  Download,
  Trash2,
  Clock,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  FileText,
  Lightbulb,
  Compass,
  X,
} from "lucide-react";
import { getActiveProfile } from "./SettingsPanel";
import { useT } from "../i18n";

interface Props {
  runId: string | null;
}

interface Report {
  id: string;
  runId: string;
  model: string;
  createdAt: string;
  content: string;
  isDemo?: boolean;
}

const REPORTS_KEY = "lego-rl-reports";

const RECO_PATTERNS = [
  /^#{1,3}\s*\d*\.?\s*Top[- ]Priority Recommendations/im,
  /^#{1,3}\s*\d+\.\s*Top[- ]Priority Recommendations/im,
  /^#{1,3}\s*Recommendations/im,
  /^#{1,3}\s*\d+\.\s*Recommendations/im,
];

const USER_DIRECTED_PATTERNS = [
  /^#{1,4}\s*\d*\.?\s*User[- ]Directed Analysis/im,
  /^#{1,4}\s*\d+\.\s*User[- ]Directed Analysis/im,
];

function _extractSection(
  content: string,
  patterns: RegExp[],
): { body: string; rest: string } | null {
  for (const pat of patterns) {
    const match = pat.exec(content);
    if (!match) continue;
    const start = match.index;
    const afterHeader = content.indexOf("\n", start);
    const headerLine = content.slice(start, afterHeader);
    const headerLevel = (headerLine.match(/^(#{1,4})/) || ["##"])[0].length;

    const sameOrHigher = new RegExp(`^#{1,${headerLevel}}\\s`, "m");
    const bodyAfterHeader = content.slice(afterHeader + 1);
    const nextSection = sameOrHigher.exec(bodyAfterHeader);

    let sectionEnd: number;
    if (nextSection) {
      sectionEnd = afterHeader + 1 + nextSection.index;
    } else {
      sectionEnd = content.length;
    }

    let body = content.slice(afterHeader + 1, sectionEnd).trim();
    const tailMatch = /\n---\s*\n\*.*\*\s*$/.exec(body);
    if (tailMatch) {
      body = body.slice(0, tailMatch.index).trim();
    }

    const rest =
      content.slice(0, start).trim() +
      "\n\n" +
      content.slice(sectionEnd).trim();
    return { body, rest: rest.trim() };
  }
  return null;
}

function splitReport(content: string): {
  recommendations: string;
  userDirected: string;
  rest: string;
} {
  let remaining = content;
  let recommendations = "";
  let userDirected = "";

  const ud = _extractSection(remaining, USER_DIRECTED_PATTERNS);
  if (ud) {
    userDirected = ud.body;
    remaining = ud.rest;
  }

  const reco = _extractSection(remaining, RECO_PATTERNS);
  if (reco) {
    recommendations = reco.body;
    remaining = reco.rest;
  }

  return { recommendations, userDirected, rest: remaining.trim() };
}

function loadReports(): Report[] {
  try {
    const raw = localStorage.getItem(REPORTS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveReports(reports: Report[]) {
  localStorage.setItem(
    REPORTS_KEY,
    JSON.stringify(reports.filter((r) => !r.isDemo)),
  );
}

// GFM is what makes the report's tables render at all — the diagnostic template
// is table-driven, and without it every table falls back to a wall of pipes.
// Module-level constant so the plugin array is not rebuilt on every render.
const GFM = [remarkGfm];

const PROSE = "prose prose-invert prose-sm max-w-none prose-headings:text-slate-200 prose-p:text-slate-300 prose-strong:text-slate-200 prose-code:text-indigo-300 prose-code:bg-slate-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-li:text-slate-300 prose-a:text-indigo-400 prose-table:text-slate-300 prose-th:text-slate-200 prose-td:text-slate-300 prose-th:border-slate-700 prose-td:border-slate-700";

function ReportBody({ content }: { content: string }) {
  const { t } = useT();
  const { recommendations, userDirected, rest } = useMemo(
    () => splitReport(content),
    [content],
  );

  if (!recommendations && !userDirected) {
    return (
      <div className={`border-t border-slate-800/60 px-6 py-5 ${PROSE}`}>
        <Markdown remarkPlugins={GFM}>{content}</Markdown>
      </div>
    );
  }

  return (
    <div className="border-t border-slate-800/60">
      {/* Highlighted recommendations box */}
      {recommendations && (
        <div className="mx-4 mt-4 rounded-xl border border-indigo-500/30 bg-indigo-500/[0.04] overflow-hidden">
          <div className="flex items-center gap-2 px-5 pt-4 pb-2">
            <Lightbulb size={15} className="text-indigo-400" />
            <span className="text-sm font-semibold text-indigo-300">
              {t("analysis.recoTitle")}
            </span>
          </div>
          <div className={`px-5 pb-4 ${PROSE} prose-headings:text-indigo-200`}>
            <Markdown remarkPlugins={GFM}>{recommendations}</Markdown>
          </div>
          <div className="px-5 pb-3">
            <p className="text-[11px] text-slate-500 italic">
              {t("analysis.recoHint")}
            </p>
          </div>
        </div>
      )}

      {/* User-directed analysis box */}
      {userDirected && (
        <div className="mx-4 mt-3 rounded-xl border border-teal-500/30 bg-teal-500/[0.04] overflow-hidden">
          <div className="flex items-center gap-2 px-5 pt-4 pb-2">
            <Compass size={15} className="text-teal-400" />
            <span className="text-sm font-semibold text-teal-300">
              {t("analysis.userDirectedTitle")}
            </span>
          </div>
          <div className={`px-5 pb-4 ${PROSE} prose-headings:text-teal-200 prose-strong:text-teal-200`}>
            <Markdown remarkPlugins={GFM}>{userDirected}</Markdown>
          </div>
          <div className="px-5 pb-3">
            <p className="text-[11px] text-slate-500 italic">
              {t("analysis.userDirectedHint")}
            </p>
          </div>
        </div>
      )}

      {/* Full report */}
      <div className={`px-6 py-5 ${PROSE}`}>
        <Markdown remarkPlugins={GFM}>{rest}</Markdown>
      </div>
    </div>
  );
}

export default function AnalysisPanel({ runId }: Props) {
  const { t } = useT();
  const [userReports, setUserReports] = useState<Report[]>(loadReports);
  const [demoReports, setDemoReports] = useState<Report[]>([]);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [progress, setProgress] = useState("");
  const [showConfirm, setShowConfirm] = useState(false);

  const profile = getActiveProfile();

  useEffect(() => {
    fetch("/api/analysis/demo-reports")
      .then((r) => r.json())
      .then((data: Report[]) => {
        const demos = data.map((d) => ({ ...d, isDemo: true }));
        setDemoReports(demos);
        if (demos.length > 0 && userReports.length === 0 && !expandedId) {
          setExpandedId(demos[0].id);
        }
      })
      .catch(() => {});
  }, []);

  const allReports = [...userReports, ...demoReports];

  const handleGenerateClick = useCallback(() => {
    if (!runId || !profile) return;
    setShowConfirm(true);
  }, [runId, profile]);

  const generate = useCallback(async () => {
    if (!runId || !profile) return;
    setShowConfirm(false);
    setGenerating(true);
    setError(null);
    setProgress(t("analysis.buildingCtx"));

    try {
      setProgress(t("analysis.callingLLM"));
      const res = await fetch("/api/analysis/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId,
          api_key: profile.apiKey,
          base_url: profile.baseUrl,
          model: profile.model,
          custom_prompt: profile.customPrompt || "",
        }),
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const report: Report = {
        id: Date.now().toString(36),
        runId,
        model: profile.model,
        createdAt: new Date().toISOString(),
        content: data.report,
      };

      const updated = [report, ...userReports].slice(0, 20);
      setUserReports(updated);
      saveReports(updated);
      setExpandedId(report.id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setGenerating(false);
      setProgress("");
    }
  }, [runId, profile, userReports, t]);

  const deleteReport = (id: string) => {
    const updated = userReports.filter((r) => r.id !== id);
    setUserReports(updated);
    saveReports(updated);
    if (expandedId === id) setExpandedId(null);
  };

  const downloadReport = (report: Report) => {
    const blob = new Blob([report.content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-${report.runId}-${report.id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderReportCard = (report: Report) => {
    const expanded = expandedId === report.id;
    return (
      <div
        key={report.id}
        className="rounded-xl bg-slate-900/80 border border-slate-800/60 overflow-hidden"
      >
        <button
          onClick={() => setExpandedId(expanded ? null : report.id)}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-800/30 transition-colors"
        >
          <div className="flex items-center gap-3 text-left">
            {expanded ? (
              <ChevronDown size={14} className="text-slate-500" />
            ) : (
              <ChevronRight size={14} className="text-slate-500" />
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-200 font-medium">
                  {report.runId}
                </span>
                {report.isDemo && (
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20">
                    DEMO
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-[10px] text-slate-500">
                <Clock size={10} />
                {new Date(report.createdAt).toLocaleString()}
                <span className="font-mono">{report.model}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={(e) => {
                e.stopPropagation();
                downloadReport(report);
              }}
              className="p-1.5 text-slate-500 hover:text-indigo-400 transition-colors"
              title={t("analysis.download")}
            >
              <Download size={13} />
            </button>
            {!report.isDemo && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteReport(report.id);
                }}
                className="p-1.5 text-slate-500 hover:text-rose-400 transition-colors"
                title={t("analysis.delete")}
              >
                <Trash2 size={13} />
              </button>
            )}
          </div>
        </button>

        {expanded && <ReportBody content={report.content} />}
      </div>
    );
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            {t("analysis.title")}
          </h2>
          <p className="text-xs text-slate-500">
            {t("analysis.desc")}
          </p>
        </div>
        <button
          onClick={handleGenerateClick}
          disabled={generating || !runId || !profile}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-500 text-white text-xs font-medium hover:bg-indigo-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {generating ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Sparkles size={14} />
          )}
          {generating ? t("analysis.generating") : t("analysis.generate")}
        </button>
      </div>

      {/* Confirmation modal */}
      {showConfirm && profile && runId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md mx-4 rounded-2xl bg-slate-900 border border-slate-700/60 shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800/60">
              <h3 className="text-sm font-semibold text-slate-100">
                {t("analysis.confirmTitle")}
              </h3>
              <button
                onClick={() => setShowConfirm(false)}
                className="text-slate-500 hover:text-slate-300 transition-colors"
              >
                <X size={16} />
              </button>
            </div>
            <div className="px-5 py-4 space-y-3">
              <p className="text-xs text-slate-400">
                {t("analysis.confirmMsg")}
              </p>
              <div className="rounded-lg bg-slate-800/60 border border-slate-700/40 p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wider text-slate-500">{t("analysis.confirmRun")}</span>
                  <span className="text-xs font-mono text-slate-200">{runId}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wider text-slate-500">{t("analysis.confirmProfile")}</span>
                  <span className="text-xs text-slate-300">{profile.name}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wider text-slate-500">{t("analysis.confirmModel")}</span>
                  <span className="text-xs font-mono text-slate-300">{profile.model}</span>
                </div>
                <div className="flex items-start justify-between gap-4">
                  <span className="text-[10px] uppercase tracking-wider text-slate-500 flex-shrink-0 pt-0.5">{t("analysis.confirmCustom")}</span>
                  <span className="text-xs text-slate-400 text-right truncate max-w-[250px]">
                    {profile.customPrompt?.trim() || t("analysis.confirmNone")}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-slate-800/60 bg-slate-900/50">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-4 py-2 rounded-lg border border-slate-700 text-slate-400 text-xs hover:text-slate-200 transition-colors"
              >
                {t("analysis.cancelBtn")}
              </button>
              <button
                onClick={generate}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-500 text-white text-xs font-medium hover:bg-indigo-400 transition-colors"
              >
                <Sparkles size={13} />
                {t("analysis.confirmBtn")}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status/warnings */}
      {!profile && (
        <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 p-4 mb-4 flex items-start gap-2">
          <AlertCircle
            size={14}
            className="text-amber-400 mt-0.5 flex-shrink-0"
          />
          <div className="text-xs text-amber-300">
            {t("analysis.noProfile")}
          </div>
        </div>
      )}

      {!runId && !allReports.length && (
        <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-8 text-center text-sm text-slate-500 mb-4">
          {t("analysis.selectRun")}
        </div>
      )}

      {profile && runId && (
        <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-3 mb-4 flex items-center justify-between">
          <div className="text-xs text-slate-400">
            <span className="text-slate-500">Profile:</span>{" "}
            <span className="text-slate-300">{profile.name}</span>
            <span className="text-slate-600 mx-2">&middot;</span>
            <span className="font-mono text-slate-400">{profile.model}</span>
            <span className="text-slate-600 mx-2">&middot;</span>
            <span className="font-mono text-slate-500">
              {new URL(profile.baseUrl).host}
            </span>
          </div>
          <div className="text-xs text-slate-500">
            Run: <span className="font-mono text-slate-400">{runId}</span>
          </div>
        </div>
      )}

      {generating && (
        <div className="rounded-xl bg-indigo-500/5 border border-indigo-500/20 p-4 mb-4 flex items-center gap-3">
          <Loader2 size={16} className="text-indigo-400 animate-spin" />
          <span className="text-sm text-indigo-300">{progress}</span>
        </div>
      )}

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/20 p-4 mb-4">
          <div className="text-xs text-rose-400 font-medium mb-1">
            {t("analysis.failed")}
          </div>
          <div className="text-xs text-rose-300 font-mono">{error}</div>
        </div>
      )}

      {/* Reports */}
      <div className="space-y-2">
        {/* User-generated reports */}
        {userReports.length > 0 && (
          <>
            {demoReports.length > 0 && (
              <div className="flex items-center gap-2 mb-1 mt-2">
                <Sparkles size={11} className="text-slate-500" />
                <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                  {t("analysis.yourReports")}
                </span>
              </div>
            )}
            {userReports.map(renderReportCard)}
          </>
        )}

        {/* Demo reports */}
        {demoReports.length > 0 && (
          <>
            <div className="flex items-center gap-2 mb-1 mt-4">
              <FileText size={11} className="text-slate-500" />
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                {t("analysis.demoReports")}
              </span>
            </div>
            {demoReports.map(renderReportCard)}
          </>
        )}

        {allReports.length === 0 && !generating && (
          <div className="rounded-xl border border-dashed border-slate-700 p-8 text-center">
            <Sparkles size={24} className="mx-auto text-slate-600 mb-2" />
            <p className="text-sm text-slate-500">
              {t("analysis.noReports")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
