import { createContext, useContext } from "react";

export type Lang = "en" | "zh";

const STORAGE_KEY = "lego-rl-lang";

export function getSavedLang(): Lang {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "zh" || saved === "en") return saved;
  return navigator.language.startsWith("zh") ? "zh" : "en";
}

export function saveLang(lang: Lang) {
  localStorage.setItem(STORAGE_KEY, lang);
}

const dict: Record<string, Record<Lang, string>> = {
  // App — sidebar
  "app.title": { en: "Lego-RL", zh: "Lego-RL" },
  "app.subtitle": { en: "Training Dashboard", zh: "训练看板" },
  "app.runs": { en: "Runs", zh: "运行记录" },
  "app.refresh": { en: "Refresh", zh: "刷新频率" },
  "app.noRuns": { en: "No runs found", zh: "暂无运行记录" },
  "app.steps": { en: "steps", zh: "步" },
  "app.live": { en: "Live", zh: "运行中" },
  "app.hardRefresh": { en: "Hard refresh", zh: "强制刷新" },
  "app.lightMode": { en: "Switch to light mode", zh: "切换亮色模式" },
  "app.darkMode": { en: "Switch to dark mode", zh: "切换暗色模式" },
  "app.langToggle": { en: "切换中文", zh: "Switch to EN" },

  // Nav panels
  "nav.overview": { en: "Overview", zh: "总览" },
  "nav.rewards": { en: "Rewards & Scores", zh: "奖励与得分" },
  "nav.optimization": { en: "Optimization", zh: "优化(奖励+策略)" },
  "nav.perfSystem": { en: "Perf & System", zh: "性能与系统" },
  "nav.metrics": { en: "Training Metrics", zh: "训练曲线" },
  "nav.policy": { en: "Policy & Training", zh: "策略与训练" },
  "nav.agent": { en: "Agent Loop", zh: "智能体循环" },
  "nav.sequences": { en: "Sequences", zh: "序列分析" },
  "nav.performance": { en: "Performance", zh: "性能" },
  "nav.episode": { en: "Episode Summary", zh: "回合统计" },
  "nav.rolloutDist": { en: "Rollout Outcomes", zh: "Rollout 结果分析" },
  "nav.tokenIS": { en: "Token & IS", zh: "Token / IS" },
  "nav.taskGrid": { en: "Task Grid", zh: "任务网格" },
  "nav.async": { en: "Async Health", zh: "异步健康度" },
  "nav.filter": { en: "Traj Filter", zh: "轨迹过滤" },
  "nav.validation": { en: "Validation", zh: "验证" },
  "nav.stability": { en: "Stability", zh: "训练稳定性" },
  "nav.analysis": { en: "AI Analysis", zh: "AI 分析" },
  "nav.logs": { en: "Logs", zh: "日志" },
  "nav.explorer": { en: "Explorer", zh: "指标浏览" },
  "nav.import": { en: "Import", zh: "导入快照" },
  "nav.settings": { en: "Settings", zh: "设置" },

  // Import / snapshot panel
  "import.title": { en: "Import Result Snapshot", zh: "导入结果快照" },
  "import.desc": {
    en: "Register a run by a server-accessible path. Two path types are auto-detected: (1) a real harbor run directory (an exp-dir with step_NNNN/<task>/, or a project dir of several) — referenced in place, no copy; (2) a single-step rollout bundle (brief jsonl + full trajectories + token tensors) — materialized into the trial layout. Either shows up as a run usable by Rollout Dist / Task Grid / Trajectory / Episode panels.",
    zh: "按服务器可访问的路径注册一个 run。系统会自动识别两类路径：(1) 真实的 harbor 运行目录（含 step_NNNN/<task>/ 的 exp-dir，或包含多个 exp-dir 的 project 目录）——原地引用、不复制数据；(2) 单步 rollout 结果包（简要 jsonl + 完整轨迹 + token 张量）——物化成 trial 目录结构。两者都会作为一个 run 出现，可直接用 Rollout 分布 / 任务网格 / 轨迹 / 回合统计等面板分析。",
  },
  "import.pathLabel": { en: "Run directory or bundle path", zh: "运行目录 或 结果包路径" },
  "import.pathHint": {
    en: "Path on the server filesystem. Either a real run dir (exp-dir with step_NNNN/, referenced in place — best for shared storage) OR a bundle dir/zip containing the 3 files (*.jsonl, rollout_trajectories_*.json, rollout_output_*.json).",
    zh: "服务器文件系统上的路径。可以是真实运行目录（含 step_NNNN/ 的 exp-dir，原地引用、最适合共享存储），或包含三个文件（*.jsonl、rollout_trajectories_*.json、rollout_output_*.json）的结果包目录/zip。",
  },
  "import.logPathLabel": { en: "Training log path (optional)", zh: "训练日志路径（可选）" },
  "import.logPathHint": {
    en: "Optional. Only needed for training-curve panels (reward/loss/lr/grad_norm vs step) — those live in the console log, not the trajectory dir. Leave empty for trajectory-only analysis.",
    zh: "可选。仅当需要训练指标曲线（reward/loss/lr/grad_norm 随 step 变化）时才填 —— 这些数据在训练 console 日志里，不在轨迹目录中。只看轨迹分析可留空。",
  },
  "import.nameLabel": { en: "Display name (optional)", zh: "显示名称（可选）" },
  "import.namePlaceholder": { en: "e.g. 0617 swerebench ohsdk step1", zh: "例如 0617 swerebench ohsdk step1" },
  "import.button": { en: "Import", zh: "导入" },
  "import.done": { en: "Imported successfully", zh: "导入成功" },
  "import.samples": { en: "Groups × N", zh: "分组 × N" },
  "import.meanReward": { en: "Mean reward", zh: "平均奖励" },
  "import.zeroAdv": { en: "Zero-advantage groups", zh: "零优势分组" },
  "import.repos": { en: "Repos", zh: "仓库数" },
  "import.respLen": { en: "Resp len p50 / max", zh: "回复长度 中位/最大" },
  "import.meanLogprob": { en: "Mean logprob", zh: "平均 logprob" },
  "import.tokenFile": { en: "Token file", zh: "Token 文件" },
  "import.rewardSource": { en: "Reward source", zh: "奖励来源" },
  "import.openInPanels": { en: "Open in analysis panels", zh: "在分析面板中打开" },
  "import.existing": { en: "Imported snapshots", zh: "已导入快照" },
  "import.none": { en: "No snapshots imported yet", zh: "尚未导入任何快照" },
  "import.open": { en: "Open", zh: "打开" },
  "import.delete": { en: "Delete", zh: "删除" },
  "import.confirmDelete": { en: "Delete this snapshot?", zh: "确定删除该快照？" },

  // Overview panel
  "config.title": { en: "Task Configuration", zh: "任务配置" },
  "config.subtitle": {
    en: "Key knobs of this run, parsed from its console log.",
    zh: "本次训练的关键配置，从运行日志解析得到。",
  },
  "config.nodes": { en: "Nodes", zh: "节点" },
  "config.rawToggle": { en: "Full resolved config (raw)", zh: "完整原始配置" },
  "config.unavailable": {
    en: "Config not available for this run.",
    zh: "该 run 暂无可解析的配置。",
  },
  "config.loadFailed": { en: "Failed to load", zh: "加载失败" },
  "overview.step": { en: "Step", zh: "步数" },
  "overview.epoch": { en: "Epoch", zh: "轮次" },
  "overview.rewardMean": { en: "Reward (mean)", zh: "奖励（均值）" },
  "overview.scoreMean": { en: "Score (mean)", zh: "得分（均值）" },
  "overview.actorLoss": { en: "Actor Loss", zh: "Actor 损失" },
  "overview.entropy": { en: "Entropy", zh: "熵" },
  "overview.clipFrac": { en: "Clip Frac", zh: "截断比例" },
  "overview.gradNorm": { en: "Grad Norm", zh: "梯度范数" },
  "overview.responseLen": { en: "Response Len", zh: "响应长度" },
  "overview.throughput": { en: "Throughput", zh: "吞吐量" },
  "overview.stepTime": { en: "Step Time", zh: "步耗时" },

  // Rewards panel
  "rewards.score": { en: "Score", zh: "得分" },
  "rewards.rewards": { en: "Rewards", zh: "奖励" },
  "rewards.advantages": { en: "Advantages", zh: "优势函数" },
  "rewards.returns": { en: "Returns", zh: "回报" },
  "rewards.scoreTrend": { en: "Score & Reward Trend", zh: "得分与奖励趋势" },
  "rewards.advDist": { en: "Advantage Distribution", zh: "优势函数分布" },

  // Policy panel
  "policy.pgLoss": { en: "Policy Gradient Loss", zh: "策略梯度损失" },
  "policy.entropy": { en: "Entropy", zh: "熵" },
  "policy.clipFrac": { en: "Clip Fraction", zh: "截断比例" },
  "policy.gradNorm": { en: "Gradient Norm", zh: "梯度范数" },
  "policy.lr": { en: "Learning Rate", zh: "学习率" },

  // Agent panel
  "agent.harborTotal": { en: "Harbor Total Time (s)", zh: "Harbor 总耗时 (s)" },
  "agent.agentExec": { en: "Agent Execute (s)", zh: "智能体执行 (s)" },
  "agent.modelInfer": { en: "Model Inference (s)", zh: "模型推理 (s)" },
  "agent.toolCalls": { en: "Tool Calls (s)", zh: "工具调用 (s)" },
  "agent.envSetup": { en: "Env Setup (s)", zh: "环境搭建 (s)" },
  "agent.agentSetup": { en: "Agent Setup (s)", zh: "智能体初始化 (s)" },
  "agent.verifier": { en: "Verifier (s)", zh: "验证器 (s)" },
  "agent.computeScore": { en: "Compute Score (s)", zh: "计算得分 (s)" },
  "agent.genSeq": { en: "Generate Sequences (s)", zh: "生成序列 (s)" },
  "agent.numTurns": { en: "Num Turns", zh: "对话轮数" },
  "agent.proxyCalls": { en: "Proxy Calls", zh: "代理调用数" },
  "agent.toolCallsCount": { en: "Tool Calls Count", zh: "工具调用次数" },
  "agent.proxyCompletion": { en: "Proxy Tokens (completion)", zh: "代理 Token（生成）" },
  "agent.proxyPrompt": { en: "Proxy Tokens (prompt)", zh: "代理 Token（提示）" },
  "agent.preempted": { en: "Preempted & Aborted", zh: "抢占与中止" },

  // Sequences panel
  "seq.responseLen": { en: "Response Length", zh: "响应长度" },
  "seq.promptLen": { en: "Prompt Length", zh: "提示长度" },
  "seq.clipRatio": { en: "Clip Ratio (Truncation Rate)", zh: "截断率" },
  "seq.aborted": { en: "Aborted Responses", zh: "中止的响应" },
  "seq.nonAborted": { en: "Response Length (non-aborted)", zh: "响应长度（未中止）" },
  "seq.balance": { en: "Sequence Balance (global_seqlen)", zh: "序列均衡 (global_seqlen)" },

  // Performance panel
  "perf.throughput": { en: "Throughput (tokens/s/GPU)", zh: "吞吐量 (tokens/s/GPU)" },
  "perf.stepTime": { en: "Time per Step", zh: "每步耗时" },
  "perf.totalTokens": { en: "Total Tokens per Step", zh: "每步总 Token 数" },
  "perf.mfu": { en: "Model FLOPs Utilization (MFU)", zh: "模型算力利用率 (MFU)" },
  "perf.staging": { en: "Stage Timing (seconds)", zh: "阶段耗时（秒）" },
  "perf.perToken": { en: "Per-Token Timing (ms)", zh: "每 Token 耗时 (ms)" },

  // Validation panel
  "val.rewardMean": { en: "Val Score (mean@1)", zh: "验证得分（mean@1）" },
  "val.turnsMean": { en: "Val Turns (mean)", zh: "验证轮数（均值）" },
  "val.turnsMin": { en: "Val Turns (min)", zh: "验证轮数（最小）" },
  "val.turnsMax": { en: "Val Turns (max)", zh: "验证轮数（最大）" },
  "val.reward": { en: "Validation Score", zh: "验证得分" },
  "val.turns": { en: "Validation Turns", zh: "验证轮数" },
  "val.vsTrainScore": { en: "Val Score vs Train Score", zh: "验证得分 vs 训练得分" },
  "val.time": { en: "Validation Time", zh: "验证耗时" },

  // Stability panel
  "stab.klPpl": { en: "Rollout Correction: KL & Perplexity", zh: "Rollout 校正：KL & 困惑度" },
  "stab.logPplDiff": { en: "Log-Perplexity Diff", zh: "对数困惑度差" },
  "stab.chi2": { en: "Chi-Squared Test", zh: "卡方检验" },
  "stab.isWeights": { en: "Importance Sampling Weights", zh: "重要性采样权重" },
  "stab.probsDiff": { en: "Rollout Probs Diff", zh: "Rollout 概率差" },
  "stab.variance": { en: "Variance Proxy", zh: "方差代理" },

  // Explorer panel
  "explorer.search": { en: "Search metrics...", zh: "搜索指标..." },
  "explorer.noMetrics": { en: "No metrics available yet", zh: "暂无指标数据" },
  "explorer.noMatch": { en: "No metrics match your search", zh: "无匹配的指标" },

  // Logs panel
  "logs.filter": { en: "Filter logs...", zh: "过滤日志..." },
  "logs.resume": { en: "Resume", zh: "继续" },
  "logs.pause": { en: "Pause", zh: "暂停" },
  "logs.scrollBottom": { en: "Scroll to bottom", zh: "滚动到底部" },
  "logs.noMatch": { en: "No matching lines", zh: "无匹配行" },
  "logs.noOutput": { en: "No log output yet", zh: "暂无日志输出" },

  // Settings panel
  "settings.title": { en: "Settings", zh: "设置" },
  "settings.desc": {
    en: "Configure LLM API profiles for AI-powered analysis reports. Keys are stored in your browser only.",
    zh: "配置 LLM API 档案以生成 AI 分析报告。密钥仅存储在浏览器本地。",
  },
  "settings.noProfiles": { en: "No profiles yet. Add one to enable AI analysis.", zh: "暂无档案，添加一个以启用 AI 分析。" },
  "settings.addProfile": { en: "Add Profile", zh: "添加档案" },
  "settings.editProfile": { en: "Edit Profile", zh: "编辑档案" },
  "settings.newProfile": { en: "New Profile", zh: "新建档案" },
  "settings.name": { en: "Name", zh: "名称" },
  "settings.baseUrl": { en: "API Base URL", zh: "API 基础地址" },
  "settings.model": { en: "Model", zh: "模型" },
  "settings.apiKey": { en: "API Key", zh: "API 密钥" },
  "settings.apiKeyHint": {
    en: "Stored in localStorage only. Never sent to our server — goes directly to the LLM API endpoint you configured.",
    zh: "仅存储在 localStorage 中，不会发送到我们的服务器 — 直接发往你配置的 LLM API。",
  },
  "settings.customPrompt": { en: "Custom Analysis Directions", zh: "自定义分析方向" },
  "settings.customPromptPlaceholder": {
    en: "e.g. Focus on reward sparsity and whether the agent is learning to use tools more efficiently across steps. Also analyze if the response length is trending shorter as training progresses.",
    zh: "例如：关注奖励稀疏性，以及智能体是否在学习更高效地使用工具。同时分析训练过程中响应长度是否有缩短趋势。",
  },
  "settings.customPromptHint": {
    en: "Optional. Describe your research focus or improvement directions. The report will include a dedicated section addressing your custom analysis goals.",
    zh: "可选。描述你的研究重点或改进方向，报告中会专门增加一个章节来回应你的自定义分析目标。",
  },
  "settings.save": { en: "Save", zh: "保存" },
  "settings.cancel": { en: "Cancel", zh: "取消" },
  "settings.saved": { en: "Saved!", zh: "已保存！" },

  // Analysis panel
  "analysis.title": { en: "AI Analysis", zh: "AI 分析" },
  "analysis.desc": {
    en: "Generate professional diagnostic reports using LLM analysis",
    zh: "使用 LLM 生成专业的训练诊断报告",
  },
  "analysis.generate": { en: "Generate Report", zh: "生成报告" },
  "analysis.generating": { en: "Generating...", zh: "生成中..." },
  "analysis.noProfile": {
    en: "No LLM profile configured. Go to Settings to add your API key and model to generate new reports. Demo reports are available below.",
    zh: "未配置 LLM 档案。请前往「设置」添加 API 密钥和模型以生成新报告。下方可查看示例报告。",
  },
  "analysis.selectRun": {
    en: "Select a run from the sidebar to generate an analysis report.",
    zh: "从左侧栏选择一个运行记录来生成分析报告。",
  },
  "analysis.buildingCtx": { en: "Building analysis context...", zh: "正在构建分析上下文..." },
  "analysis.callingLLM": {
    en: "Calling LLM API (this may take 1-2 minutes)...",
    zh: "正在调用 LLM API（可能需要 1-2 分钟）...",
  },
  "analysis.failed": { en: "Generation failed", zh: "生成失败" },
  "analysis.download": { en: "Download as markdown", zh: "下载为 Markdown" },
  "analysis.delete": { en: "Delete report", zh: "删除报告" },
  "analysis.yourReports": { en: "Your Reports", zh: "我的报告" },
  "analysis.demoReports": { en: "Demo Reports", zh: "示例报告" },
  "analysis.noReports": {
    en: 'No reports yet. Select a run and click "Generate Report" to get an AI-powered diagnostic analysis.',
    zh: '暂无报告。选择一个运行记录并点击「生成报告」以获取 AI 诊断分析。',
  },
  // Trajectory panel
  "nav.trajectory": { en: "Trajectory", zh: "轨迹查看" },
  "traj.title": { en: "Trajectory Viewer", zh: "轨迹查看器" },
  "traj.desc": { en: "Inspect agent ReAct trajectories from training rollouts", zh: "查看训练 rollout 中的智能体 ReAct 轨迹" },
  "traj.step": { en: "Rollout sample", zh: "Rollout 采样" },
  "traj.stepHint": {
    en: "step_NNNN is the rollout-sample folder (a global sample index), not the training step — async writes many folders per training step.",
    zh: "step_NNNN 是 rollout 采样文件夹（全局采样索引），不是训练步——异步训练每个训练步会写很多个文件夹。",
  },
  "traj.task": { en: "Task", zh: "任务" },
  "traj.reward": { en: "Reward", zh: "奖励" },
  "traj.turns": { en: "turns", zh: "轮" },
  "traj.tokens": { en: "tokens", zh: "tokens" },
  "traj.thought": { en: "Thought", zh: "思考" },
  "traj.action": { en: "Action", zh: "动作" },
  "traj.observation": { en: "Observation", zh: "观察" },
  "traj.random": { en: "Random Trial", zh: "随机选取" },
  "traj.loading": { en: "Loading trajectory...", zh: "加载轨迹中..." },
  "traj.noData": { en: "No trajectory data found for this run. Select a run with harbor_trials data.", zh: "未找到该运行的轨迹数据。请选择一个有 harbor_trials 数据的运行记录。" },
  "traj.turn": { en: "Turn", zh: "轮次" },
  "traj.expandAll": { en: "Expand All", zh: "展开全部" },
  "traj.collapseAll": { en: "折叠全部", zh: "折叠全部" },
  "traj.duration": { en: "Duration", zh: "耗时" },
  "traj.systemPrompt": { en: "System Prompt", zh: "系统提示词" },
  "traj.problemStatement": { en: "Problem Statement", zh: "问题描述" },
  "traj.download": { en: "Download", zh: "下载" },
  "traj.downloadAll": { en: "Download All in Step", zh: "下载该步全部轨迹" },
  "traj.downloadingAll": { en: "Downloading...", zh: "下载中..." },
  "traj.allTasks": { en: "All Tasks (random)", zh: "全部任务（随机）" },
  "traj.taskCount": { en: "tasks", zh: "个任务" },
  "traj.load": { en: "Load", zh: "加载" },

  // Compare panel
  "nav.compare": { en: "Compare", zh: "对比" },
  "compare.title": { en: "Run Comparison", zh: "运行对比" },
  "compare.desc": { en: "Overlay metrics from multiple runs on the same charts", zh: "将多个运行的指标叠加在同一图表上对比" },
  "compare.selectRuns": { en: "Select runs to compare", zh: "选择要对比的运行" },
  "compare.selectMetrics": { en: "Metrics", zh: "指标" },
  "compare.noRuns": { en: "Select at least 2 runs from the list above to compare.", zh: "从上方列表中选择至少 2 个运行进行对比。" },
  "compare.loading": { en: "Loading metrics...", zh: "加载指标中..." },
  "compare.clear": { en: "Clear", zh: "清除" },
  "compare.downloadPNG": { en: "PNG", zh: "PNG" },
  "compare.downloadCSV": { en: "CSV", zh: "CSV" },

  "analysis.confirmTitle": { en: "Confirm Report Generation", zh: "确认生成报告" },
  "analysis.confirmMsg": { en: "Generate an AI analysis report for the following run?", zh: "确认为以下运行记录生成 AI 分析报告？" },
  "analysis.confirmRun": { en: "Run", zh: "运行记录" },
  "analysis.confirmModel": { en: "Model", zh: "模型" },
  "analysis.confirmProfile": { en: "Profile", zh: "档案" },
  "analysis.confirmCustom": { en: "Custom directions", zh: "自定义方向" },
  "analysis.confirmNone": { en: "(none)", zh: "（无）" },
  "analysis.confirmBtn": { en: "Confirm & Generate", zh: "确认并生成" },
  "analysis.cancelBtn": { en: "Cancel", zh: "取消" },
  "analysis.recoTitle": { en: "Top-Priority Recommendations", zh: "最优先建议" },
  "analysis.recoHint": {
    en: "These are the highest-impact suggestions distilled from the full analysis. Scroll down for detailed observations across all dimensions.",
    zh: "这些是从完整分析中提炼的最高影响力建议。向下滚动查看各维度的详细分析。",
  },
  "analysis.userDirectedTitle": { en: "Your Research Directions", zh: "你的研究方向" },
  "analysis.userDirectedHint": {
    en: "Focused analysis based on your custom directions configured in Settings. Related metrics in the standard sections are cross-referenced.",
    zh: "基于你在「设置」中配置的自定义方向进行的定向分析，已与标准章节中的相关指标交叉引用。",
  },
};

export type TFunc = (key: string) => string;

export function createT(lang: Lang): TFunc {
  return (key: string) => dict[key]?.[lang] ?? dict[key]?.en ?? key;
}

export const I18nContext = createContext<{ lang: Lang; t: TFunc }>({
  lang: "en",
  t: createT("en"),
});

export function useT() {
  return useContext(I18nContext);
}
