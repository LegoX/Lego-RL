import { useState, useEffect, useRef, useCallback } from "react";
import { Search, ArrowDown, Pause, Play } from "lucide-react";

interface Props {
  runId: string | null;
  refreshInterval: number;
}

interface LogChunk {
  lines: string[];
  total_lines: number;
  offset: number;
}

export default function LogsPanel({ runId, refreshInterval }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [totalLines, setTotalLines] = useState(0);
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [paused, setPaused] = useState(false);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastOffsetRef = useRef(0);

  const fetchLogs = useCallback(async () => {
    if (!runId || paused) return;
    try {
      setLoading(true);
      const tail = 500;
      const res = await fetch(
        `/api/runs/${runId}/logs?tail=${tail}&offset=${lastOffsetRef.current}`,
      );
      if (!res.ok) return;
      const data: LogChunk = await res.json();
      if (data.lines.length > 0) {
        if (lastOffsetRef.current === 0) {
          setLines(data.lines);
        } else {
          setLines((prev) => [...prev, ...data.lines].slice(-2000));
        }
        lastOffsetRef.current = data.offset + data.lines.length;
      }
      setTotalLines(data.total_lines);
    } catch {
      // ignore fetch errors
    } finally {
      setLoading(false);
    }
  }, [runId, paused]);

  useEffect(() => {
    lastOffsetRef.current = 0;
    setLines([]);
    fetchLogs();
  }, [runId]);

  useEffect(() => {
    if (paused) return;
    const id = setInterval(fetchLogs, refreshInterval * 1000);
    return () => clearInterval(id);
  }, [fetchLogs, refreshInterval, paused]);

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 40;
    setAutoScroll(atBottom);
  };

  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
      setAutoScroll(true);
    }
  };

  const filtered = search
    ? lines.filter((l) => l.toLowerCase().includes(search.toLowerCase()))
    : lines;

  const highlightLine = (line: string): string => {
    if (line.includes("ERROR") || line.includes("error"))
      return "text-rose-400";
    if (line.includes("WARNING") || line.includes("warning"))
      return "text-amber-400";
    if (line.match(/^.*step:\d+/)) return "text-emerald-300";
    return "text-slate-400";
  };

  if (!runId) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        Select a run to view logs
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Logs</h2>
          <p className="text-xs text-slate-500">
            {totalLines.toLocaleString()} lines total
            {search && ` / ${filtered.length} matched`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
            />
            <input
              type="text"
              placeholder="Filter logs..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-slate-800 text-slate-300 text-xs border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 w-56 focus:outline-none focus:border-indigo-500"
            />
          </div>
          <button
            onClick={() => setPaused(!paused)}
            className={`p-1.5 rounded-lg border transition-colors ${
              paused
                ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
                : "border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title={paused ? "Resume" : "Pause"}
          >
            {paused ? <Play size={14} /> : <Pause size={14} />}
          </button>
          <button
            onClick={scrollToBottom}
            className="p-1.5 rounded-lg border border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Scroll to bottom"
          >
            <ArrowDown size={14} />
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 min-h-0 overflow-y-auto rounded-xl bg-slate-950 border border-slate-800/60 font-mono text-xs leading-5"
        style={{ maxHeight: "calc(100vh - 200px)" }}
      >
        {loading && lines.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-slate-500">
            Loading logs...
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-slate-500">
            {search ? "No matching lines" : "No log output yet"}
          </div>
        ) : (
          <table className="w-full">
            <tbody>
              {filtered.map((line, i) => (
                <tr
                  key={i}
                  className="hover:bg-slate-900/50 group"
                >
                  <td className="px-3 py-0 text-right text-slate-600 select-none w-12 align-top group-hover:text-slate-500">
                    {i + 1}
                  </td>
                  <td
                    className={`px-2 py-0 whitespace-pre-wrap break-all ${highlightLine(line)}`}
                  >
                    {line}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {!autoScroll && (
        <button
          onClick={scrollToBottom}
          className="fixed bottom-6 right-6 px-3 py-1.5 rounded-lg bg-indigo-500 text-white text-xs shadow-lg hover:bg-indigo-400 transition-colors flex items-center gap-1.5"
        >
          <ArrowDown size={12} />
          New output
        </button>
      )}
    </div>
  );
}
