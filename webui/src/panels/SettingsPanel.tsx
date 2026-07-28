import { useState, useEffect } from "react";
import { Save, Eye, EyeOff, Trash2, Plus } from "lucide-react";
import { useT } from "../i18n";

export interface LLMProfile {
  id: string;
  name: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  customPrompt: string;
}

const STORAGE_KEY = "swe-lego-rl-llm-profiles";
const ACTIVE_KEY = "swe-lego-rl-active-profile";

export function loadProfiles(): LLMProfile[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveProfiles(profiles: LLMProfile[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles));
}

export function getActiveProfile(): LLMProfile | null {
  const profiles = loadProfiles();
  const activeId = localStorage.getItem(ACTIVE_KEY);
  return profiles.find((p) => p.id === activeId) || profiles[0] || null;
}

export function setActiveProfileId(id: string) {
  localStorage.setItem(ACTIVE_KEY, id);
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

export default function SettingsPanel() {
  const { t } = useT();
  const [profiles, setProfiles] = useState<LLMProfile[]>(loadProfiles);
  const [activeId, setActiveId] = useState(
    localStorage.getItem(ACTIVE_KEY) || "",
  );
  const [editing, setEditing] = useState<LLMProfile | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    saveProfiles(profiles);
  }, [profiles]);

  const handleSave = () => {
    if (!editing) return;
    setProfiles((prev) => {
      const idx = prev.findIndex((p) => p.id === editing.id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = editing;
        return next;
      }
      return [...prev, editing];
    });
    if (!activeId || profiles.length === 0) {
      setActiveId(editing.id);
      setActiveProfileId(editing.id);
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleAdd = () => {
    setEditing({
      id: generateId(),
      name: `Profile ${profiles.length + 1}`,
      apiKey: "",
      baseUrl: "https://api.openai.com/v1",
      model: "gpt-4o",
      customPrompt: "",
    });
    setShowKey(false);
  };

  const handleDelete = (id: string) => {
    setProfiles((prev) => prev.filter((p) => p.id !== id));
    if (editing?.id === id) setEditing(null);
    if (activeId === id) {
      const remaining = profiles.filter((p) => p.id !== id);
      const newActive = remaining[0]?.id || "";
      setActiveId(newActive);
      setActiveProfileId(newActive);
    }
  };

  const handleActivate = (id: string) => {
    setActiveId(id);
    setActiveProfileId(id);
  };

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-100 mb-1">{t("settings.title")}</h2>
      <p className="text-xs text-slate-500 mb-4">
        {t("settings.desc")}
      </p>

      {/* Profile list */}
      <div className="space-y-2 mb-4">
        {profiles.map((p) => (
          <div
            key={p.id}
            className={`flex items-center justify-between px-4 py-3 rounded-xl border transition-colors cursor-pointer ${
              activeId === p.id
                ? "bg-indigo-500/10 border-indigo-500/30"
                : "bg-slate-900/80 border-slate-800/60 hover:border-slate-700/60"
            }`}
            onClick={() => {
              setEditing({ ...p });
              setShowKey(false);
            }}
          >
            <div className="flex items-center gap-3">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleActivate(p.id);
                }}
                className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${
                  activeId === p.id
                    ? "bg-indigo-500 border-indigo-400"
                    : "border-slate-600 hover:border-slate-400"
                }`}
              />
              <div>
                <span className="text-sm text-slate-200 font-medium">
                  {p.name}
                </span>
                <div className="text-[10px] text-slate-500 font-mono">
                  {p.model} &middot; {new URL(p.baseUrl).host}
                </div>
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(p.id);
              }}
              className="text-slate-600 hover:text-rose-400 transition-colors"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
        {profiles.length === 0 && (
          <p className="text-xs text-slate-500 text-center py-6">
            {t("settings.noProfiles")}
          </p>
        )}
      </div>

      <button
        onClick={handleAdd}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-dashed border-slate-700 text-slate-400 text-xs hover:border-indigo-500 hover:text-indigo-400 transition-colors mb-6"
      >
        <Plus size={12} />
        {t("settings.addProfile")}
      </button>

      {/* Edit form */}
      {editing && (
        <div className="rounded-xl bg-slate-900/80 border border-slate-800/60 p-4 space-y-4">
          <h3 className="text-sm font-medium text-slate-200">
            {profiles.find((p) => p.id === editing.id) ? t("settings.editProfile") : t("settings.newProfile")}
          </h3>

          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              {t("settings.name")}
            </label>
            <input
              type="text"
              value={editing.name}
              onChange={(e) =>
                setEditing({ ...editing, name: e.target.value })
              }
              className="w-full bg-slate-800 text-slate-200 text-sm border border-slate-700 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              {t("settings.baseUrl")}
            </label>
            <input
              type="text"
              value={editing.baseUrl}
              onChange={(e) =>
                setEditing({ ...editing, baseUrl: e.target.value })
              }
              placeholder="https://api.openai.com/v1"
              className="w-full bg-slate-800 text-slate-200 text-sm border border-slate-700 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              {t("settings.model")}
            </label>
            <input
              type="text"
              value={editing.model}
              onChange={(e) =>
                setEditing({ ...editing, model: e.target.value })
              }
              placeholder="gpt-4o"
              className="w-full bg-slate-800 text-slate-200 text-sm border border-slate-700 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              {t("settings.apiKey")}
            </label>
            <div className="relative">
              <input
                type={showKey ? "text" : "password"}
                value={editing.apiKey}
                onChange={(e) =>
                  setEditing({ ...editing, apiKey: e.target.value })
                }
                placeholder="sk-..."
                className="w-full bg-slate-800 text-slate-200 text-sm border border-slate-700 rounded-lg px-3 py-2 pr-10 focus:outline-none focus:border-indigo-500 font-mono"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              >
                {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            <p className="text-[10px] text-slate-600 mt-1">
              {t("settings.apiKeyHint")}
            </p>
          </div>

          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              {t("settings.customPrompt")}
            </label>
            <textarea
              value={editing.customPrompt}
              onChange={(e) =>
                setEditing({ ...editing, customPrompt: e.target.value })
              }
              placeholder={t("settings.customPromptPlaceholder")}
              rows={4}
              className="w-full bg-slate-800 text-slate-200 text-sm border border-slate-700 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500 resize-y"
            />
            <p className="text-[10px] text-slate-600 mt-1">
              {t("settings.customPromptHint")}
            </p>
          </div>

          <div className="flex items-center gap-2 pt-2">
            <button
              onClick={handleSave}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-500 text-white text-xs font-medium hover:bg-indigo-400 transition-colors"
            >
              <Save size={12} />
              {t("settings.save")}
            </button>
            <button
              onClick={() => setEditing(null)}
              className="px-4 py-2 rounded-lg border border-slate-700 text-slate-400 text-xs hover:text-slate-200 transition-colors"
            >
              {t("settings.cancel")}
            </button>
            {saved && (
              <span className="text-xs text-emerald-400">{t("settings.saved")}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
