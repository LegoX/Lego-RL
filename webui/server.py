"""
Lightweight API server for the RL training dashboard.

Data sources:
  1. Local log files — parses verl console output (step:N - key:val - ...)
  2. wandb API — proxied to avoid exposing API keys in the browser

Usage:
  python server.py                          # auto-detect log dir
  python server.py --log-dir /path/to/logs  # explicit
  python server.py --wandb-entity X --wandb-project Y  # enable wandb
  python server.py --static-dir dist        # serve built frontend
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError
from urllib.parse import urlparse, parse_qs
import threading
import uuid
import shutil
import zipfile

# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"^step:(\d+)\s*-\s*(.+)$")
KV_RE = re.compile(r"([^:]+):(.*)")
NP_RE = re.compile(r"np\.(?:float|int)(?:32|64)\(([^)]+)\)")


def _clean_value(raw: str) -> float | None:
    raw = raw.strip()
    m = NP_RE.match(raw)
    if m:
        raw = m.group(1)
    if raw in ("nan", "None", "inf", "-inf", ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_log_file(path: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            # strip Ray worker prefix like "(TaskRunner pid=123456) "
            paren = line.find(")")
            if paren > 0 and line.lstrip().startswith("("):
                line = line[paren + 1 :].strip()
            m = STEP_RE.match(line.strip())
            if not m:
                continue
            step_num = int(m.group(1))
            rest = m.group(2)
            point: dict[str, Any] = {"step": step_num}
            for part in rest.split(" - "):
                kv = KV_RE.match(part.strip())
                if kv:
                    key = kv.group(1).strip()
                    val = _clean_value(kv.group(2))
                    if val is not None:
                        point[key] = val
            if len(point) > 1:
                steps.append(point)
    return steps


LOG_EXTENSIONS = (".log", ".out")
# Sidecar/wrapper logs that are never a run on their own: the vLLM throughput
# stream, the per-node GPU wandb keepalive, this server's own stdout, and the
# `launch_*` nohup wrappers (whose content duplicates the canonical exp log via
# `tee`, so surfacing them would double every runner-launched run).
SKIP_SUFFIXES = ("_vllm.log", "_train_gpu_wandb.log", "_server.out")
SKIP_PREFIXES = ("launch_",)

# Number of TRAINING steps a run reached = max of `training/global_step` in its
# log (float in the log, e.g. "1.0" -> 1). Cached by (mtime, size) so a running
# run only re-scans when its log actually grows. Used to hide never-really-ran
# jobs from the run list (see _handle_runs): show only running runs or steps > 5.
_GSTEP_RE = re.compile(r"training/global_step:(\d+)")
_STEP_COUNT_CACHE: dict[str, tuple[float, int, int]] = {}


def _training_step_count(path: str, size: int, mtime: float) -> int:
    ent = _STEP_COUNT_CACHE.get(path)
    if ent and ent[0] == mtime and ent[1] == size:
        return ent[2]
    max_step = 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if "training/global_step:" not in line:
                    continue
                m = _GSTEP_RE.search(line)
                if m:
                    v = int(m.group(1))
                    if v > max_step:
                        max_step = v
    except OSError:
        return 0
    _STEP_COUNT_CACHE[path] = (mtime, size, max_step)
    return max_step


def _strip_ext(name: str) -> str:
    for ext in LOG_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def discover_runs_in_dir(log_dir: str) -> list[dict[str, Any]]:
    runs = []
    for ext in LOG_EXTENSIONS:
        for path in sorted(glob.glob(os.path.join(log_dir, f"*{ext}"))):
            name = os.path.basename(path)
            if any(name.endswith(s) for s in SKIP_SUFFIXES) or name.startswith(SKIP_PREFIXES):
                continue
            try:
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
            except OSError:
                # Dangling symlink or file removed mid-scan — skip it instead of
                # letting the whole run list (this API) crash.
                continue
            age = time.time() - mtime
            state = "running" if age < 120 else ("finished" if size > 1000 else "unknown")
            run_id = _strip_ext(name)
            runs.append(
                {
                    "id": run_id,
                    "name": run_id,
                    "state": state,
                    "steps": _training_step_count(path, size, mtime),
                    "created_at": datetime.fromtimestamp(
                        os.path.getctime(path), tz=timezone.utc
                    ).isoformat(),
                    "source": "log",
                    "path": path,
                    "size": size,
                }
            )
    return runs


def discover_runs(log_dirs: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in log_dirs:
        if not d:
            continue
        for r in discover_runs_in_dir(d):
            if r["id"] not in seen:
                seen.add(r["id"])
                runs.append(r)
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


# ---------------------------------------------------------------------------
# wandb proxy
# ---------------------------------------------------------------------------


def wandb_api(
    entity: str, project: str, api_key: str, path: str
) -> dict[str, Any] | list[Any]:
    url = f"https://api.wandb.ai/api/v1/{entity}/{project}/{path}"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wandb_runs(entity: str, project: str, api_key: str) -> list[dict[str, str]]:
    data = wandb_api(entity, project, api_key, "runs?per_page=50")
    runs = []
    for r in data if isinstance(data, list) else data.get("runs", data.get("data", [])):
        runs.append(
            {
                "id": r.get("id", r.get("name", "")),
                "name": r.get("displayName", r.get("name", "")),
                "state": r.get("state", "unknown"),
                "created_at": r.get("createdAt", ""),
                "source": "wandb",
            }
        )
    return runs


def wandb_history(
    entity: str, project: str, api_key: str, run_id: str
) -> list[dict[str, Any]]:
    data = wandb_api(
        entity, project, api_key, f"runs/{run_id}/history?samples=1500"
    )
    rows = data if isinstance(data, list) else data.get("history", data.get("data", []))
    cleaned = []
    for row in rows:
        point: dict[str, Any] = {}
        for k, v in row.items():
            if k.startswith("_") and k != "_step":
                continue
            if isinstance(v, (int, float)) and v == v:  # filter NaN
                point[k.replace("_step", "step")] = v
        if point:
            cleaned.append(point)
    return cleaned


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class MetricsCache:
    def __init__(self, ttl: float = 10.0):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry[0]) < self._ttl:
                return entry[1]
        return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (time.time(), value)


_cache = MetricsCache(ttl=10)

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Analysis report generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "analysis_prompt.md")


def _load_prompt_template() -> str:
    """Template text with its leading `#`-comment header removed.

    That header documents the available placeholders by NAME, so leaving it in
    means every `{{placeholder}}` gets substituted twice — once where it belongs
    and once inside the doc comment. With the big payloads that is ~70k chars of
    duplicated metrics table and step JSON prepended to every request, which the
    model reads as content. Strip lines matching `#` or `# ...` off the front;
    markdown headings (`## `, `### `) never match, so the body is untouched.
    """
    try:
        with open(PROMPT_TEMPLATE_PATH, "r") as f:
            raw = f.read()
    except FileNotFoundError:
        return "Analyze the following RL training metrics and provide recommendations:\n\n{{metrics_summary}}"

    lines = raw.split("\n")
    i = 0
    while i < len(lines) and (lines[i] == "#" or lines[i].startswith("# ")):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def _build_metrics_summary(metrics: list[dict]) -> str:
    full = [m for m in metrics if len(m) > 10]
    if not full:
        full = metrics
    all_keys: set[str] = set()
    for m in full:
        all_keys.update(k for k in m if k != "step")

    lines = [f"{'Metric':<65} {'First':>12} {'Last':>12} {'Min':>12} {'Max':>12} {'Trend':>8}"]
    lines.append("-" * 125)
    for key in sorted(all_keys):
        vals = [m[key] for m in full if key in m and m[key] is not None]
        if not vals:
            continue
        first, last = vals[0], vals[-1]
        vmin, vmax = min(vals), max(vals)
        if len(vals) >= 2 and first != 0:
            pct = ((last - first) / abs(first)) * 100
            trend = f"{pct:+.1f}%"
        else:
            trend = "—"
        lines.append(
            f"{key:<65} {first:>12.6g} {last:>12.6g} {vmin:>12.6g} {vmax:>12.6g} {trend:>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis evidence builders.
#
# The metrics table alone can only support "entropy is rising" -- it cannot
# support "26% of trajectories were cut off and still entered the gradient".
# These turn the per-step metrics (and the trial-dir aggregates the analysis
# panels already compute) into the specific claims a diagnostic report has to
# make, so the model argues from numbers instead of from vibes.
# ---------------------------------------------------------------------------

# Series worth a significance test: the ones a reader will claim a trend on.
_TREND_KEYS = (
    ("critic/score/mean", "reward (score)"),
    ("critic/rewards/mean", "reward"),
    ("num_turns/mean", "turns/trajectory"),
    ("response_length/mean", "response length"),
    ("actor/entropy", "entropy"),
    ("actor/grad_norm", "grad norm"),
    ("actor/kl_loss", "kl loss"),
    ("training/rollout_actor_probs_pearson_corr", "rollout/actor pearson"),
)


def _linfit(xs: list[float], ys: list[float]) -> dict | None:
    """Least-squares slope with its standard error, so a trend can be reported
    against its own noise instead of eyeballed off a chart. t = slope/se; |t| < 2
    means the run has not moved by more than step-to-step scatter."""
    n = len(xs)
    if n < 4:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = (s2 / sxx) ** 0.5
    sd = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    k = max(1, min(5, n // 4))
    return {
        "n": n,
        "slope": slope,
        "se": se,
        "t": (slope / se) if se > 0 else None,
        "sd": sd,
        "first_k": sum(ys[:k]) / k,
        "last_k": sum(ys[-k:]) / k,
        "k": k,
        "span": xs[-1] - xs[0],
    }


def _build_trend_stats(metrics: list[dict]) -> str:
    full = [m for m in metrics if len(m) > 10] or metrics
    if not full:
        return "(no metrics)"
    lines = [
        "Least-squares trend per series. t = slope / stderr(slope): |t| >= 2 is a",
        "real trend, |t| < 2 means step-to-step noise dominates and you must NOT",
        "claim the run is improving. noise/trend = sd(series) / |slope * span|.",
        "",
        f"{'series':<28}{'first→last':>22}{'slope/step':>14}{'t':>8}{'verdict':>14}{'noise/trend':>13}",
        "-" * 99,
    ]
    seen = 0
    for key, label in _TREND_KEYS:
        pts = [
            (float(m.get("step", i)), float(m[key]))
            for i, m in enumerate(full)
            if m.get(key) is not None and m[key] == m[key]
        ]
        if len(pts) < 4:
            continue
        fit = _linfit([p[0] for p in pts], [p[1] for p in pts])
        if not fit:
            continue
        seen += 1
        t = fit["t"]
        if t is None:
            verdict = "—"
        elif t >= 2:
            verdict = "rising"
        elif t <= -2:
            verdict = "falling"
        else:
            verdict = "FLAT (noise)"
        drift = abs(fit["slope"] * fit["span"])
        ratio = (fit["sd"] / drift) if drift > 0 else float("inf")
        if ratio == float("inf"):
            ratio_s = "inf"
        else:
            ratio_s = f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"
        lines.append(
            f"{label:<28}{fit['first_k']:>10.4g} → {fit['last_k']:<9.4g}"
            f"{fit['slope']:>14.4g}{(f'{t:+.2f}' if t is not None else '—'):>8}"
            f"{verdict:>14}{ratio_s:>13}"
        )
    if not seen:
        return "(not enough steps for a trend test)"
    lines.append("")
    lines.append(
        f"(first→last = mean of the first vs last {max(1, min(5, len(full) // 4))} logged steps)"
    )
    return "\n".join(lines)


def _build_verbosity_stats(metrics: list[dict]) -> str:
    """Output per TURN, which separates "explored more" from "got wordier".
    Trajectory length rising while turn count is flat/falling is verbosity
    drift, not more exploration -- and the two have opposite fixes."""
    full = [m for m in metrics if len(m) > 10] or metrics
    rows = []
    for i, m in enumerate(full):
        rl, tn = m.get("response_length/mean"), m.get("num_turns/mean")
        if rl is None or tn is None or not tn or tn != tn or rl != rl:
            continue
        rows.append((float(m.get("step", i)), float(rl), float(tn), float(rl) / float(tn)))
    if len(rows) < 4:
        return "(needs both response_length/mean and num_turns/mean — not available for this run)"
    k = max(1, min(5, len(rows) // 4))

    def avg(idx, sl):
        return sum(r[idx] for r in sl) / len(sl)

    a, b = rows[:k], rows[-k:]
    out = [
        f"{'':<22}{'first ' + str(k) + ' steps':>18}{'last ' + str(k) + ' steps':>18}{'change':>12}",
        "-" * 70,
    ]
    for idx, label in ((1, "response len (tok)"), (2, "turns"), (3, "tok per TURN")):
        f, l = avg(idx, a), avg(idx, b)
        pct = ((l - f) / f * 100) if f else 0.0
        out.append(f"{label:<22}{f:>18.4g}{l:>18.4g}{pct:>11.1f}%")
    out.append("")
    out.append(
        "Reading: tok-per-turn flat = length grew because the agent took more turns "
        "(more exploration). tok-per-turn rising while turns are flat or falling = "
        "verbosity drift — the policy is writing more per turn without acting more."
    )
    return "\n".join(out)


def _build_term_reasons(metrics: list[dict]) -> str:
    """Termination-reason mix, early vs late. A rising dropped/timeout band is
    environment trouble; a rising max_turns band is the agent failing to finish."""
    full = [m for m in metrics if len(m) > 10] or metrics
    reasons = [
        "agent_completed",
        "overlong",
        "max_turns_reached",
        "timeout",
        "env_setup_failed",
    ]
    rows = []
    for m in full:
        vals = {}
        tot = 0.0
        for r in reasons:
            v = m.get(f"trajectory_filter/reason/{r}")
            if isinstance(v, (int, float)) and v == v:
                vals[r] = float(v)
                tot += float(v)
        if tot > 0:
            rows.append((vals, tot))
    if len(rows) < 2:
        return "(no trajectory_filter/reason/* metrics for this run)"
    k = max(1, min(5, len(rows) // 4))

    def share(sl, r):
        num = sum(v.get(r, 0.0) for v, _t in sl)
        den = sum(t for _v, t in sl)
        return (num / den) if den else 0.0

    a, b = rows[:k], rows[-k:]
    out = [
        f"batch size (tagged rollouts/step) ≈ {rows[-1][1]:.0f}",
        "",
        f"{'reason':<22}{'first ' + str(k):>12}{'last ' + str(k):>12}   {'class'}",
        "-" * 62,
    ]
    for r in reasons:
        fa, fb = share(a, r), share(b, r)
        if fa == 0 and fb == 0:
            continue
        cls = "DROPPED (env noise)" if r in ("timeout", "env_setup_failed") else "kept (in loss)"
        out.append(f"{r:<22}{fa * 100:>11.1f}%{fb * 100:>11.1f}%   {cls}")
    inv = [m.get("trajectory_filter/invalid_ratio") for m in full]
    inv = [float(x) for x in inv if isinstance(x, (int, float)) and x == x]
    if inv:
        out.append("")
        out.append(
            f"invalid_ratio (share neutralized out of the loss): "
            f"{inv[0] * 100:.1f}% → {inv[-1] * 100:.1f}%"
        )
    return "\n".join(out)


def _fmt_rollout_dist(dist: dict | None, series: dict | None, batch_prompts: float | None) -> str:
    """Zero-advantage accounting: a GRPO group whose n rollouts all score the
    same contributes nothing, so the batch size that matters is the number of
    groups with spread, not the nominal one."""
    if not dist:
        return (
            "(not computed — the Rollout Distribution panel populates this cache; "
            "open it once for this run to give the next report this evidence)"
        )
    n = dist.get("expected_n") or 8
    tot = dist.get("train_groups") or 0
    if not tot:
        return "(no complete prompt-groups found in the trial dirs)"
    aw, ar = dist.get("all_wrong") or 0, dist.get("all_correct") or 0
    mixed = max(0, tot - aw - ar)
    p = lambda x: f"{x / tot * 100:.1f}%"  # noqa: E731
    out = [
        f"prompt-groups scanned: {tot}  (n={n} rollouts each)",
        f"  all-wrong  (0/{n}) : {aw:>6}  {p(aw):>7}   ZERO advantage — task too hard for the policy",
        f"  mixed  (1..{n - 1}/{n}) : {mixed:>6}  {p(mixed):>7}   HAS gradient",
        f"  all-right  ({n}/{n}) : {ar:>6}  {p(ar):>7}   ZERO advantage — task too easy",
        f"  no-signal total    : {aw + ar:>6}  {p(aw + ar):>7}",
    ]
    eff = mixed / tot
    if batch_prompts:
        out.append("")
        out.append(
            f"EFFECTIVE gradient groups per step ≈ {batch_prompts:.0f} prompts "
            f"x {eff * 100:.1f}% = {batch_prompts * eff:.1f} "
            f"(nominal batch is {batch_prompts:.0f})"
        )
    if aw > ar:
        out.append("")
        out.append(
            "NOTE: no-signal is dominated by ALL-WRONG — the task set is too hard "
            "for the current policy. That is worse than the all-right case: those "
            "prompts yield nothing no matter how many times they are sampled."
        )
    elif ar > aw:
        out.append("")
        out.append(
            "NOTE: no-signal is dominated by ALL-RIGHT — the task set skews easy "
            "for the current policy; headroom is shrinking."
        )
    bs = (series or {}).get("buckets") or []
    if len(bs) >= 4:
        q = max(1, len(bs) // 4)
        f = sum(b["no_signal_frac"] for b in bs[:q]) / q
        l = sum(b["no_signal_frac"] for b in bs[-q:]) / q
        fs = sum(b["mean_solve_rate"] for b in bs[:q]) / q
        ls = sum(b["mean_solve_rate"] for b in bs[-q:]) / q
        out.append("")
        out.append(
            f"over training (generation order): no-signal {f * 100:.1f}% → {l * 100:.1f}%, "
            f"mean solve rate {fs * 100:.1f}% → {ls * 100:.1f}%"
        )
    return "\n".join(out)


def _fmt_traj_shape(ts: dict | None) -> str:
    """First-epoch vs last-epoch trajectory shape, per task then averaged. Same
    task, same prompt — so a change here is the policy changing, not the data."""
    if not ts:
        return (
            "(not computed — the Task Grid panel's trajectory-shape view populates "
            "this cache; open it once for this run to give the next report this evidence)"
        )
    s = ts.get("summary") or {}
    if not ts.get("num_tasks"):
        return (
            "(no multi-epoch tasks with readable trajectories — this run's trial dirs "
            "may not carry proxy_trajectory.json)"
        )
    out = [
        f"{ts['num_tasks']} tasks rolled out in >=2 epochs ({ts.get('trials_read', 0)} trials read)",
        "",
        f"{'':<20}{'first epoch':>14}{'last epoch':>14}{'change':>11}",
        "-" * 59,
    ]
    labels = {
        "turns": "turns",
        "out_tok": "output tokens",
        "cot_chars": "CoT chars",
        "resp_chars": "response chars",
        "cot_ratio": "CoT share",
    }
    for k, label in labels.items():
        d = s.get(k) or {}
        f, l, pct = d.get("first"), d.get("last"), d.get("pct")
        if f is None or l is None:
            continue
        out.append(
            f"{label:<20}{f:>14.4g}{l:>14.4g}"
            f"{(f'{pct * 100:+.1f}%' if pct is not None else '—'):>11}"
        )
    return "\n".join(out)


def _fmt_task_solve(ts: dict | None) -> str:
    if not ts or not ts.get("num_tasks"):
        return (
            "(not computed — the Task Grid panel populates this cache; open it once "
            "for this run to give the next report this evidence)"
        )
    n = ts["num_tasks"]
    p = lambda x: f"{x} ({x / n * 100:.1f}%)"  # noqa: E731
    return "\n".join(
        [
            f"tasks seen: {n}  (n={ts.get('expected_n')} rollouts per epoch)",
            f"  never solved in any epoch : {p(ts.get('never_solved', 0))}  <- permanently zero signal",
            f"  always solved             : {p(ts.get('always_solved', 0))}",
            f"  rolled out in >=2 epochs  : {p(ts.get('multi_epoch', 0))}",
            f"    of those, improved      : {ts.get('improved', 0)}",
            f"    of those, regressed     : {ts.get('regressed', 0)}",
        ]
    )


def _extract_training_config(run_id: str, log_dirs: list[str]) -> str:
    for d in log_dirs:
        if not d:
            continue
        for ext in LOG_EXTENSIONS + (".out",):
            p = os.path.join(d, f"{run_id}{ext}")
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", errors="replace") as f:
                    content = f.read(200_000)
                config_start = content.find("{'actor_rollout_ref':")
                if config_start == -1:
                    continue
                depth = 0
                end = config_start
                for i, ch in enumerate(content[config_start:], config_start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                raw = content[config_start:end]
                raw = ANSI_RE.sub("", raw)
                raw = re.sub(r"\([A-Za-z]+ pid=\d+\)\s*", "", raw)
                return raw[:8000]
            except Exception:
                continue
    return "(not available — config dump not found in log)"


def _extract_val_results(metrics: list[dict]) -> str:
    val_steps = [m for m in metrics if any(k.startswith("val-") for k in m)]
    if not val_steps:
        return "(no validation results found)"
    lines = []
    for m in val_steps:
        step = m.get("step", "?")
        vals = {k: v for k, v in m.items() if k.startswith("val-") or k == "rollouter/validate_time"}
        parts = [f"{k}={v}" for k, v in sorted(vals.items())]
        lines.append(f"Step {step}: {', '.join(parts)}")
    return "\n".join(lines)


def _build_analysis_prompt(
    run_id: str, metrics: list[dict], log_dirs: list[str],
    custom_prompt: str = "", evidence: dict | None = None,
) -> str:
    template = _load_prompt_template()
    full_steps = [m for m in metrics if len(m) > 10]
    summary = _build_metrics_summary(metrics)
    config = _extract_training_config(run_id, log_dirs)
    val = _extract_val_results(metrics)
    step_json = json.dumps(full_steps[-10:] if len(full_steps) > 10 else full_steps, indent=1, default=str)
    ev = evidence or {}

    if custom_prompt.strip():
        custom_section = (
            "## 6. User-Directed Analysis\n\n"
            "Add this as a SIXTH section, after Data provenance. It is an extra "
            "lens, not a replacement — sections 0–5 stay complete and keep their "
            "format. Address each point below with specific evidence, the same "
            "hard rules, and concrete recommendations; cross-reference the earlier "
            "sections where they bear on it.\n\n"
            f"**User directions:**\n\n{custom_prompt.strip()}\n"
        )
    else:
        custom_section = ""

    prompt = template
    prompt = prompt.replace("{{run_id}}", run_id)
    prompt = prompt.replace("{{num_steps}}", str(len(full_steps)))
    prompt = prompt.replace("{{metrics_summary}}", summary)
    prompt = prompt.replace("{{training_config}}", config)
    prompt = prompt.replace("{{val_results}}", val)
    prompt = prompt.replace("{{step_data}}", step_json[:30000])
    prompt = prompt.replace("{{custom_directions}}", custom_section)
    prompt = prompt.replace("{{trend_stats}}", _build_trend_stats(metrics))
    prompt = prompt.replace("{{verbosity_stats}}", _build_verbosity_stats(metrics))
    prompt = prompt.replace("{{term_reasons}}", _build_term_reasons(metrics))
    prompt = prompt.replace(
        "{{rollout_dist}}",
        _fmt_rollout_dist(ev.get("dist"), ev.get("dist_series"), ev.get("batch_prompts")),
    )
    prompt = prompt.replace("{{traj_shape}}", _fmt_traj_shape(ev.get("traj_shape")))
    prompt = prompt.replace("{{task_solve}}", _fmt_task_solve(ev.get("task_solve")))
    return prompt


def call_llm_api(
    api_key: str, base_url: str, model: str, prompt: str
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.3,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = Request(url, data=payload, headers=headers, method="POST")
    with urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


TRIALS_DIRS: list[str] = []


def _discover_trials_dirs() -> list[str]:
    dirs: list[str] = []
    base = os.path.join(os.path.dirname(__file__), "..", "harbor_trials")
    if os.path.isdir(base):
        dirs.append(os.path.abspath(base))
    return dirs


def _find_exp_dir(run_id: str) -> str | None:
    for base in TRIALS_DIRS:
        for project in os.listdir(base):
            exp_dir = os.path.join(base, project, run_id)
            if os.path.isdir(exp_dir):
                return exp_dir
    return None


def _list_steps(run_id: str) -> list[str]:
    exp_dir = _find_exp_dir(run_id)
    if not exp_dir:
        return []
    return sorted(
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    )


def _read_reward(task_dir: str) -> float | None:
    result_file = os.path.join(task_dir, "result.json")
    if not os.path.isfile(result_file):
        return None
    try:
        with open(result_file) as f:
            rdata = json.load(f)
        vr = rdata.get("verifier_result", {})
        if isinstance(vr, dict):
            return vr.get("rewards", {}).get("reward")
    except Exception:
        pass
    return None


def _list_tasks_in_step(run_id: str, step: str) -> list[dict]:
    exp_dir = _find_exp_dir(run_id)
    if not exp_dir:
        return []
    step_dir = os.path.join(exp_dir, step)
    if not os.path.isdir(step_dir):
        return []
    tasks = []
    for task_name in sorted(os.listdir(step_dir)):
        task_dir = os.path.join(step_dir, task_name)
        traj_file = os.path.join(task_dir, "agent", "litellm-trajectory.jsonl")
        if not os.path.isfile(traj_file):
            continue
        reward = _read_reward(task_dir)
        tasks.append({"task": task_name, "reward": reward})
    return tasks


def _pick_trial(run_id: str, step: str | None = None, task: str | None = None) -> dict | None:
    import random
    exp_dir = _find_exp_dir(run_id)
    if not exp_dir:
        return None

    steps = _list_steps(run_id)
    if not steps:
        return None
    chosen_step = step if step and step in steps else random.choice(steps)

    step_dir = os.path.join(exp_dir, chosen_step)

    if task:
        task_dir = os.path.join(step_dir, task)
        traj_file = os.path.join(task_dir, "agent", "litellm-trajectory.jsonl")
        if os.path.isfile(traj_file):
            return {
                "step": chosen_step, "task": task,
                "reward": _read_reward(task_dir), "traj_path": traj_file,
            }
        return None

    task_names = [
        d for d in os.listdir(step_dir)
        if os.path.isdir(os.path.join(step_dir, d))
    ]
    random.shuffle(task_names)
    for task_name in task_names:
        task_dir = os.path.join(step_dir, task_name)
        traj_file = os.path.join(task_dir, "agent", "litellm-trajectory.jsonl")
        if os.path.isfile(traj_file):
            return {
                "step": chosen_step, "task": task_name,
                "reward": _read_reward(task_dir), "traj_path": traj_file,
            }
    return None


def _extract_msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")
        )
    return str(content)


def _parse_trajectory(traj_path: str, max_content: int = 800) -> tuple[str, str, list[dict]]:
    """Returns (system_prompt, problem_statement, turns)."""
    turns: list[dict] = []
    system_prompt = ""
    problem_statement = ""
    try:
        with open(traj_path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                req = entry.get("request_body", {})
                resp = entry.get("response_body", {})
                usage = entry.get("usage", {})
                duration = entry.get("duration_ms")

                if i == 0:
                    first_msgs = req.get("messages", [])
                    for m in first_msgs:
                        role = m.get("role", "")
                        if role == "system" and not system_prompt:
                            system_prompt = _extract_msg_text(m.get("content", ""))
                        elif role == "user" and not problem_statement:
                            problem_statement = _extract_msg_text(m.get("content", ""))

                choices = resp.get("choices", [])
                msg = choices[0].get("message", {}) if choices else {}
                content = _strip_think(msg.get("content") or "")
                tool_calls = msg.get("tool_calls", [])

                actions = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "")
                    if len(args_str) > max_content:
                        args_str = args_str[:max_content] + "..."
                    actions.append({"name": fn.get("name", ""), "arguments": args_str})

                tool_results = []
                msgs = req.get("messages", [])
                for m in msgs:
                    if m.get("role") == "tool":
                        tr_content = str(m.get("content", ""))
                        if len(tr_content) > max_content:
                            tr_content = tr_content[:max_content] + "..."
                        tool_results.append(tr_content)

                if len(content) > max_content:
                    content = content[:max_content] + "..."

                turns.append({
                    "turn": i,
                    "thought": content,
                    "actions": actions,
                    "observations": tool_results[-len(actions):] if tool_results else [],
                    "usage": {
                        "prompt": usage.get("prompt_tokens", 0),
                        "completion": usage.get("completion_tokens", 0),
                    },
                    "duration_ms": duration,
                })
    except Exception:
        pass
    return system_prompt, problem_statement, turns


# ---------------------------------------------------------------------------
# Async-aware multi-experiment-dir trajectory resolution
#
# For fully-async runs the on-disk layout is NOT one exp-dir per run and the
# `step_NNNN` folder name is a global rollout/sample index, NOT the training
# step. A single logical run is spread across several exp-dirs (one per rollout
# worker / node), e.g.
#   harbor_trials/<project>/<run>-120418/step_0001/<task_id>-<hash>/
#   harbor_trials/<project>/<run>-120425/step_0001/<task_id>-<hash>/
#   harbor_trials/<project>/<run>-120429/step_0001/<task_id>-<hash>/
# The 8 rollouts of one prompt share the same (step, task_id-prefix) but are
# split across those exp-dirs. We therefore aggregate across all exp-dirs and
# group trials by (step, task_id) where task_id strips the trailing -<hash>.
# ---------------------------------------------------------------------------

TRIAL_HASH_RE = re.compile(r"-[0-9a-f]{6,}$")
EXP_DIR_RE = re.compile(
    r"(/[^\s\"'=()]*?/harbor_trials/[^\s\"'=()]+?/[^\s\"'=()/]+)/step_\d+"
)


def _strip_trial_hash(trial_name: str) -> str:
    return TRIAL_HASH_RE.sub("", trial_name)


def _exp_dirs_from_log(log_path: str) -> list[str]:
    """Parse a run's console log for the harbor_trials exp-dirs it wrote to."""
    found: list[str] = []
    seen: set[str] = set()
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "harbor_trials" not in line:
                    continue
                for m in EXP_DIR_RE.finditer(line):
                    d = m.group(1)
                    if d not in seen and os.path.isdir(d):
                        seen.add(d)
                        found.append(d)
    except Exception:
        pass
    return found


# A multi-node fully-async run launches the SAME script on every node, and each
# node stamps its own exp_name with its local `date +%H%M%S` — so one logical run
# writes to SEVERAL sibling exp-dirs that differ ONLY in the trailing -HHMMSS
# (e.g. ...-veomni-20260629-102114 / -102118 / -102122, seconds apart). The N
# rollouts of one prompt are split across these dirs. But the console log usually
# references only the head node's dir with a /step_ path (the others appear only
# as env echoes that EXP_DIR_RE misses), so reward-grouping sees ~1/Nnodes of each
# group and every group looks the wrong size. Recover the siblings from disk.
EXP_TS_RE = re.compile(r"^(.*)-(\d{8})-(\d{6})$")


def _exp_dir_stem_ts(name: str) -> tuple[str | None, float | None]:
    m = EXP_TS_RE.match(name)
    if not m:
        return None, None
    try:
        ts = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None, None
    return m.group(1), ts


def _has_step_dirs(d: str) -> bool:
    try:
        with os.scandir(d) as it:
            for de in it:
                if de.name.startswith("step_"):
                    return True
    except (FileNotFoundError, NotADirectoryError):
        pass
    return False


def _expand_sibling_exp_dirs(seed_dirs: list[str], window_sec: int = 600) -> list[str]:
    """Add sibling exp-dirs (same base name, launch time within window_sec,
    containing step_ dirs) that the log-parse missed — the other nodes of a
    multi-node async run. No-op when names lack the -YYYYMMDD-HHMMSS suffix."""
    out = list(seed_dirs)
    seen = set(seed_dirs)
    for d in seed_dirs:
        stem, ts = _exp_dir_stem_ts(os.path.basename(d.rstrip("/")))
        if stem is None or ts is None:
            continue
        parent = os.path.dirname(d.rstrip("/"))
        try:
            with os.scandir(parent) as it:
                for de in it:
                    if de.path in seen or not de.name.startswith(stem):
                        continue
                    s2, ts2 = _exp_dir_stem_ts(de.name)
                    if s2 != stem or ts2 is None or abs(ts2 - ts) > window_sec:
                        continue
                    if de.is_dir() and _has_step_dirs(de.path):
                        seen.add(de.path)
                        out.append(de.path)
        except (FileNotFoundError, NotADirectoryError):
            continue
    return out


# ---------------------------------------------------------------------------
# Task config extraction.
#
# At startup the trainer pretty-prints its FULLY-RESOLVED OmegaConf as a Python
# dict (pprint) into the console log — one giant literal that carries model /
# data / rollout / algorithm / trainer settings. Ray interleaves other actors'
# stdout between those lines, but every config line shares the SAME actor
# prefix `(<Name> pid=<pid>) `, so we can recover the block by keeping only
# lines with that exact prefix (plus a shape filter for the rare same-pid
# warning that sneaks in) and stopping when the outer brace closes.
#
# A few operational knobs (agent max turns, k8s/agent timeouts, retries) are NOT
# in the trainer config — they come from HARBOR_*/K8S_* env vars read by the
# agent-loop YAML (path is in the config). We surface that YAML's DEFAULTS
# (labelled as such) plus any values actually echoed by the workers in the log.
# ---------------------------------------------------------------------------
import ast as _ast

_CFG_START = "{'actor_rollout_ref'"
_CFG_PREFIX_RE = re.compile(r"^\(([A-Za-z_][A-Za-z0-9_]*) pid=(\d+)(?:, ip=[\d.]+)?\)\s?")
# a genuine pprint line, after lstrip, begins with one of these tokens
_CFG_LINE_OK = re.compile(r"^(['\"{\[(}\])0-9\-]|True\b|False\b|None\b)")


def _cfg_scan_depth(s: str, d: int) -> int:
    """Brace-depth delta of a string, ignoring braces inside quoted literals."""
    instr = False
    esc = False
    q = ""
    for c in s:
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == q:
                instr = False
        else:
            if c in "'\"":
                instr = True
                q = c
            elif c == "{":
                d += 1
            elif c == "}":
                d -= 1
    return d


def _extract_config_dict(log_path: str) -> dict[str, Any] | None:
    """Recover the trainer's resolved-config dict from a run's console log."""
    buf: list[str] = []
    started = False
    depth = 0
    want_prefix: str | None = None  # exact "(<Name> pid=<pid>) " of the config block
    try:
        with open(log_path, "r", errors="replace") as f:
            for raw in f:
                s = ANSI_RE.sub("", raw.rstrip("\n"))
                if not started:
                    idx = s.find(_CFG_START)
                    if idx < 0:
                        continue
                    m = _CFG_PREFIX_RE.match(s)
                    want_prefix = s[: m.end()] if m and m.start() == 0 else ""
                    body = s[idx:]
                    buf.append(body)
                    depth = _cfg_scan_depth(body, 0)
                    started = True
                    if depth <= 0:
                        break
                    continue
                # inside the block: keep only same-prefix lines
                if want_prefix:
                    if not s.startswith(want_prefix):
                        continue
                    body = s[len(want_prefix):]
                else:
                    body = s
                if not _CFG_LINE_OK.match(body.lstrip()):
                    continue  # interleaved same-pid warning, not config
                buf.append(body)
                depth = _cfg_scan_depth(body, depth)
                if depth <= 0:
                    break
    except OSError:
        return None
    if not buf:
        return None
    try:
        cfg = _ast.literal_eval("\n".join(buf))
        return cfg if isinstance(cfg, dict) else None
    except (ValueError, SyntaxError):
        return None


def _extract_nodes(log_path: str) -> list[dict[str, Any]]:
    """Collect distinct node IPs seen in the log and the roles they played."""
    roles: dict[str, set[str]] = {}
    head: str | None = None
    worker_re = re.compile(r"\((WorkerDict|AgentLoopWorker|FullyAsyncTrainer) pid=\d+, ip=([\d.]+)\)")
    head_re = re.compile(r"Ray cluster at address:\s*([\d.]+):")
    role_names = {
        "WorkerDict": "Train worker",
        "AgentLoopWorker": "Agent Loop (rollout)",
        "FullyAsyncTrainer": "Trainer",
    }
    try:
        with open(log_path, "r", errors="replace") as f:
            for raw in f:
                s = ANSI_RE.sub("", raw)
                if head is None:
                    hm = head_re.search(s)
                    if hm:
                        head = hm.group(1)
                for cls, ip in worker_re.findall(s):
                    roles.setdefault(ip, set()).add(role_names.get(cls, cls))
    except OSError:
        pass
    if head:
        roles.setdefault(head, set()).add("Ray head (scheduler)")
    out = [{"ip": ip, "roles": sorted(rs)} for ip, rs in roles.items()]
    out.sort(key=lambda d: tuple(int(x) for x in d["ip"].split(".")))
    return out


_YAML_ENV_RE = re.compile(r"oc\.env:([A-Z0-9_]+),([^}]*)\}")


def _extract_agent_loop_defaults(yaml_path: str) -> dict[str, str]:
    """Pull `${oc.env:VAR,DEFAULT}` defaults out of the agent-loop YAML so we can
    show the operational knobs (turns/timeouts/retries) the trainer config omits."""
    out: dict[str, str] = {}
    try:
        with open(yaml_path, "r", errors="replace") as f:
            for line in f:
                for var, default in _YAML_ENV_RE.findall(line):
                    out.setdefault(var, default.strip())
    except OSError:
        pass
    return out


def _extract_observed(log_path: str) -> dict[str, str]:
    """Values the workers actually echoed (override YAML defaults where present)."""
    out: dict[str, str] = {}
    pat = re.compile(r"max_retries=(\S+)\s+tool_parser=(\S+)")
    try:
        with open(log_path, "r", errors="replace") as f:
            for raw in f:
                m = pat.search(raw)
                if m:
                    out["max_retries"] = m.group(1)
                    out["tool_parser"] = m.group(2)
                    break
    except OSError:
        pass
    return out


def _cfg_get(cfg: dict, path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _fmt(v: Any) -> Any:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "—"
    return v


def _build_config_payload(cfg: dict, nodes: list[dict], yaml_defaults: dict,
                          observed: dict) -> dict[str, Any]:
    """Curate the raw config into grouped, human-labelled sections for the UI."""
    g = lambda p, d=None: _cfg_get(cfg, p, d)
    base = os.path.basename

    def env(var: str, suffix: str = "") -> Any:
        v = observed.get(var.lower()) or yaml_defaults.get(var)
        return f"{v}{suffix}" if v not in (None, "") else "—"

    ar = "actor_rollout_ref."
    # Labels ship in both languages: the UI's own strings live in src/i18n.ts, but
    # these are generated per-config here, so the client cannot translate them by
    # key. `label_en`/`title_en` are omitted where the label is already language
    # -neutral (max_model_len, ppo_epochs, ...) and the client falls back.
    sections: list[dict] = [
        {"title": "模型 & 数据", "title_en": "Model & Data", "items": [
            {"label": "模型", "label_en": "Model", "value": g(ar + "model.path")},
            {"label": "训练后端", "label_en": "Training backend", "value": f"{g(ar + 'actor.strategy')} / {g('model_engine')}"},
            {"label": "训练数据", "label_en": "Train data", "value": base(str(g("data.train_files"))), "hint": g("data.train_files")},
            {"label": "验证数据", "label_en": "Val data", "value": base(str(g("data.val_files"))), "hint": g("data.val_files")},
            {"label": "训练样本上限", "label_en": "Train sample cap",
             "value": "All" if g("data.train_max_samples") == -1 else g("data.train_max_samples"),
             "value_en": "All" if g("data.train_max_samples") == -1 else None},
        ]},
        {"title": "生成 / Rollout", "title_en": "Generation / Rollout", "items": [
            {"label": "推理引擎", "label_en": "Inference engine", "value": f"{g(ar + 'rollout.name')} ({g(ar + 'rollout.mode')})"},
            {"label": "采样温度", "label_en": "Temperature", "value": g(ar + "rollout.temperature")},
            {"label": "top_p / top_k", "value": f"{g(ar + 'rollout.top_p')} / {g(ar + 'rollout.top_k')}"},
            {"label": "每 prompt 采样数 (n)", "label_en": "Samples per prompt (n)", "value": g(ar + "rollout.n")},
            {"label": "最大回复长度", "label_en": "Max response length", "value": g("data.max_response_length"), "hint": "response_length"},
            {"label": "最大 prompt 长度", "label_en": "Max prompt length", "value": g("data.max_prompt_length")},
            {"label": "max_model_len", "value": g(ar + "rollout.max_model_len")},
            {"label": "GPU 显存占用率", "label_en": "GPU memory utilization", "value": g(ar + "rollout.gpu_memory_utilization")},
            {"label": "张量并行 (TP)", "label_en": "Tensor parallel (TP)", "value": g(ar + "rollout.tensor_model_parallel_size")},
        ]},
        {"title": "Agent · 轮数 / 超时 / 重试", "title_en": "Agent · turns / timeouts / retries",
         "note": "值取自 agent-loop 配置默认，启动脚本的同名环境变量可覆盖；带*为日志实测值",
         "note_en": "Values are the agent-loop config defaults; same-named env vars in the launch script override them. * = observed in the log",
         "items": [
            {"label": "最大迭代轮数", "label_en": "Max iterations", "value": env("HARBOR_AGENT_MAX_ITERATIONS")},
            {"label": "重试次数", "label_en": "Retries", "value": f"{observed.get('max_retries', yaml_defaults.get('HARBOR_MAX_RETRIES', '—'))}" + ("*" if "max_retries" in observed else "")},
            {"label": "工具解析器", "label_en": "Tool parser", "value": (observed.get("tool_parser", yaml_defaults.get("HARBOR_TOOL_PARSER", "—"))) + ("*" if "tool_parser" in observed else "")},
            {"label": "Pod 启动超时", "label_en": "Pod startup timeout", "value": env("K8S_POD_STARTUP_TIMEOUT", " s")},
            {"label": "环境构建超时倍率", "label_en": "Env build timeout multiplier", "value": env("HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER")},
            {"label": "Agent setup 超时倍率", "label_en": "Agent setup timeout multiplier", "value": env("HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER")},
            {"label": "Agent 运行超时倍率", "label_en": "Agent run timeout multiplier", "value": env("HARBOR_AGENT_TIMEOUT_MULTIPLIER")},
            {"label": "Agent 单次超时上限", "label_en": "Agent max timeout", "value": env("HARBOR_AGENT_MAX_TIMEOUT_SEC")},
            {"label": "工具响应截断长度", "label_en": "Tool response truncation", "value": g(ar + "rollout.multi_turn.max_tool_response_length")},
            {"label": "多轮 multi_turn", "label_en": "Multi-turn", "value": g(ar + "rollout.multi_turn.enable")},
        ]},
        {"title": "训练 / 算法", "title_en": "Training / Algorithm", "items": [
            {"label": "优势估计", "label_en": "Advantage estimator", "value": g("algorithm.adv_estimator")},
            {"label": "策略损失", "label_en": "Policy loss", "value": g(ar + "actor.policy_loss.loss_mode")},
            {"label": "学习率", "label_en": "Learning rate", "value": f"{g(ar + 'actor.optim.lr')} ({g(ar + 'actor.optim.lr_scheduler_type')})"},
            {"label": "KL loss", "value": f"{g(ar + 'actor.use_kl_loss')} · coef={g(ar + 'actor.kl_loss_coef')}"},
            {"label": "reward 内 KL", "label_en": "KL in reward", "value": f"{g('algorithm.use_kl_in_reward')} · coef={g('algorithm.kl_ctrl.kl_coef')}"},
            {"label": "mini-batch", "value": g(ar + "actor.ppo_mini_batch_size")},
            {"label": "ppo_epochs", "value": g(ar + "actor.ppo_epochs")},
            {"label": "clip low / high", "value": f"{g(ar + 'actor.clip_ratio_low')} / {g(ar + 'actor.clip_ratio_high')}"},
        ]},
        {"title": "R3 · 轨迹过滤 · Rollout 校正", "title_en": "R3 · Trajectory filter · Rollout correction", "items": [
            {"label": "R3 路由回放", "label_en": "R3 routing replay", "value": f"{g(ar + 'rollout.enable_rollout_routing_replay')} (actor: {g(ar + 'actor.router_replay.mode')})"},
            {"label": "轨迹过滤", "label_en": "Trajectory filter", "value": g("algorithm.trajectory_filter.enable")},
            {"label": "Drop reasons", "value": g("algorithm.trajectory_filter.drop_reasons")},
            {"label": "Rollout IS 校正", "label_en": "Rollout IS correction", "value": g("algorithm.rollout_correction.rollout_is")},
            {"label": "IS 阈值", "label_en": "IS threshold", "value": g("algorithm.rollout_correction.rollout_is_threshold")},
            {"label": "序列分布指标", "label_en": "Sequence-dist metrics", "value": g("algorithm.rollout_correction.seq_dist_metrics")},
            {"label": "partial_rollout", "value": g("async_training.partial_rollout")},
            {"label": "staleness 阈值", "label_en": "Staleness threshold", "value": g("async_training.staleness_threshold")},
        ]},
        {"title": "验证 (Validation)", "title_en": "Validation", "items": [
            {"label": "val 温度", "label_en": "Val temperature", "value": g(ar + "rollout.val_kwargs.temperature")},
            {"label": "val 采样数 n", "label_en": "Val samples (n)", "value": g(ar + "rollout.val_kwargs.n")},
            {"label": "val do_sample", "value": g(ar + "rollout.val_kwargs.do_sample")},
            {"label": "验证频率 (每N步)", "label_en": "Val frequency (every N steps)", "value": g("trainer.test_freq")},
            {"label": "训练前验证", "label_en": "Val before train", "value": g("trainer.val_before_train")},
        ]},
        {"title": "集群 / 训练器", "title_en": "Cluster / Trainer", "items": [
            {"label": "节点数", "label_en": "Nodes", "value": g("trainer.nnodes")},
            {"label": "每节点 GPU 数", "label_en": "GPUs per node", "value": g("trainer.n_gpus_per_node")},
            {"label": "project", "value": g("trainer.project_name")},
            {"label": "experiment", "value": g("trainer.experiment_name")},
            {"label": "总 epoch 数", "label_en": "Total epochs", "value": g("trainer.total_epochs")},
            {"label": "保存频率", "label_en": "Save frequency", "value": g("trainer.save_freq")},
            {"label": "checkpoint 目录", "label_en": "Checkpoint dir", "value": g("trainer.default_local_dir"), "wide": True},
        ]},
    ]
    for sec in sections:
        for it in sec["items"]:
            it["value"] = _fmt(it.get("value"))
            if it.get("value_en") is None:
                it.pop("value_en", None)
    return {"sections": sections, "nodes": nodes}


def _run_config(log_path: str) -> dict[str, Any]:
    cfg = _extract_config_dict(log_path)
    nodes = _extract_nodes(log_path)
    if not cfg:
        return {"available": False, "nodes": nodes,
                "error": "Could not parse a training-config block out of the log"}
    yaml_path = _cfg_get(cfg, "actor_rollout_ref.rollout.agent.agent_loop_config_path")
    yaml_defaults = _extract_agent_loop_defaults(yaml_path) if isinstance(yaml_path, str) else {}
    observed = _extract_observed(log_path)
    payload = _build_config_payload(cfg, nodes, yaml_defaults, observed)
    payload["available"] = True
    payload["raw"] = cfg
    return payload


def _read_reward(task_dir: str) -> float | None:
    # fast path: verifier/reward.txt is a bare number
    rt = os.path.join(task_dir, "verifier", "reward.txt")
    if os.path.isfile(rt):
        try:
            return float(open(rt).read().strip())
        except Exception:
            pass
    # fallback: result.json -> verifier_result.rewards.reward
    result_file = os.path.join(task_dir, "result.json")
    if os.path.isfile(result_file):
        try:
            with open(result_file) as f:
                rdata = json.load(f)
            vr = rdata.get("verifier_result", {})
            if isinstance(vr, dict):
                return vr.get("rewards", {}).get("reward")
        except Exception:
            pass
    return None


def _has_trajectory(task_dir: str) -> bool:
    return os.path.isfile(
        os.path.join(task_dir, "agent", "trajectory.json")
    ) or os.path.isfile(
        os.path.join(task_dir, "agent", "litellm-trajectory.jsonl")
    )


def _list_steps_multi(exp_dirs: list[str]) -> list[str]:
    # NOTE: os.scandir + DirEntry name filtering avoids a per-entry stat()
    # (os.path.isdir), which on the networked trials FS is ~100x slower
    # (51s vs 0.4s across ~7500 entries).
    s: set[str] = set()
    for e in exp_dirs:
        try:
            with os.scandir(e) as it:
                for de in it:
                    if de.name.startswith("step_"):
                        s.add(de.name)
        except (FileNotFoundError, NotADirectoryError):
            continue
    return sorted(s)


def _list_tasks_in_step_multi(exp_dirs: list[str], step: str) -> list[dict]:
    # Collect candidate (name, exp_dir, task_dir) via scandir (no per-entry
    # stat), then read reward + check trajectory CONCURRENTLY. A val step has
    # ~500 tasks and the trials FS is networked (~75ms/op); doing this serially
    # took tens of seconds (the source of the trajectory-open lag).
    from concurrent.futures import ThreadPoolExecutor

    cand: list[tuple[str, str, str]] = []
    for e in exp_dirs:
        try:
            with os.scandir(os.path.join(e, step)) as it:
                for de in it:
                    cand.append((de.name, e, de.path))
        except (FileNotFoundError, NotADirectoryError):
            continue

    def _info(c):
        name, e, td = c
        if not _has_trajectory(td):
            return None
        return {"task": name, "reward": _read_reward(td), "exp_dir": e}

    with ThreadPoolExecutor(max_workers=32) as ex:
        rows = [r for r in ex.map(_info, cand) if r]
    rows.sort(key=lambda r: r["task"])
    return rows


def _pick_trial_multi(
    exp_dirs: list[str], step: str | None = None, task: str | None = None
) -> dict | None:
    import random

    steps = _list_steps_multi(exp_dirs)
    if not steps:
        return None
    chosen = step if step and step in steps else random.choice(steps)

    if task:
        # Targeted lookup: locate the one requested task by directory name
        # WITHOUT listing/reward-reading the whole step (a val step has ~500
        # tasks; scanning them all just to open one is what made this slow).
        # Match exact trial-name first, then bare task_id (strip -<hash>).
        for e in exp_dirs:
            sd = os.path.join(e, chosen)
            names: list[str] = []
            try:
                with os.scandir(sd) as it:
                    names = [de.name for de in it]
            except (FileNotFoundError, NotADirectoryError):
                continue
            match = next((nm for nm in names if nm == task), None)
            if match is None:
                match = next((nm for nm in names if _strip_trial_hash(nm) == task), None)
            if match is not None:
                td = os.path.join(sd, match)
                return {
                    "step": chosen,
                    "task": match,
                    "reward": _read_reward(td),
                    "task_dir": td,
                }
        return None

    cand = _list_tasks_in_step_multi(exp_dirs, chosen)
    if not cand:
        return None
    ti = random.choice(cand)
    return {
        "step": chosen,
        "task": ti["task"],
        "reward": ti["reward"],
        "task_dir": os.path.join(ti["exp_dir"], chosen, ti["task"]),
    }


def _collect_groups(
    exp_dirs: list[str],
    expected_n: int = 8,
    step: str | None = None,
    max_steps: int = 500,
) -> dict:
    """Scan trials and group rewards by (step, task_id) across exp-dirs.

    Trial dirs live on a (slow, networked) FS, so we avoid reading rewards for
    validation-style step dirs: a training step holds ~expected_n trials total
    across all exp-dirs, whereas a validation dump holds the whole val set
    (hundreds of tasks, n=1 each). Steps whose total entry count is far above
    expected_n are counted as val_like without reading any reward files.

    When `step` is None, the most-recent `max_steps` step dirs are scanned;
    `truncated` flags when older steps were skipped. Returns the per-group
    rewards plus the scanned-step order (for early/late analysis)."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    all_steps = [step] if step else _list_steps_multi(exp_dirs)
    truncated = False
    if not step and len(all_steps) > max_steps:
        all_steps = all_steps[-max_steps:]
        truncated = True

    big_threshold = expected_n * 3

    # Phase 1 (parallel scandir): gather entries per step across exp-dirs. Each
    # directory open on the networked FS is ~75ms, so scanning serially would
    # take minutes — fan out across threads.
    def _scan(pair):
        st, e = pair
        out: list[tuple[str, str]] = []
        try:
            with os.scandir(os.path.join(e, st)) as it:
                for de in it:
                    out.append((de.path, de.name))
        except (FileNotFoundError, NotADirectoryError):
            pass
        return st, out

    pairs = [(st, e) for st in all_steps for e in exp_dirs]
    step_entries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if pairs:
        with ThreadPoolExecutor(max_workers=32) as ex:
            for st, out in ex.map(_scan, pairs):
                step_entries[st].extend(out)

    # Classify each step: validation-style dumps are counted but their rewards
    # are not read; training steps (~expected_n trials) are grouped for lookup.
    val_like = 0
    pending: dict[tuple, list[str]] = defaultdict(list)  # key -> [task_dir,...]
    for st, entries in step_entries.items():
        if len(entries) > big_threshold:
            val_like += len({_strip_trial_hash(tn) for _td, tn in entries})
            continue
        for td, tn in entries:
            pending[(st, _strip_trial_hash(tn))].append(td)

    # Phase 2 (parallel): read rewards (I/O-bound on the networked FS).
    all_dirs = [td for tds in pending.values() for td in tds]
    rewards: dict[str, float | None] = {}
    if all_dirs:
        with ThreadPoolExecutor(max_workers=32) as ex:
            for td, rv in zip(all_dirs, ex.map(_read_reward, all_dirs)):
                rewards[td] = rv

    groups = {key: [rewards.get(td) for td in tds] for key, tds in pending.items()}
    return {
        "groups": groups,
        # Same keys as `groups`, but the trial dirs themselves — callers that need
        # more than the reward (trajectory length, turn count, …) would otherwise
        # have to re-walk the whole tree to find the dirs we just enumerated.
        "group_dirs": dict(pending),
        "scanned_steps": all_steps,
        "val_like": val_like,
        "truncated": truncated,
    }


def _rollout_distribution(
    exp_dirs: list[str],
    expected_n: int = 8,
    step: str | None = None,
    max_steps: int = 500,
) -> dict:
    """Histogram, over prompt-groups, of how many of the expected_n rollouts
    were correct. Groups at 0 or expected_n correct give zero GRPO advantage."""
    col = _collect_groups(exp_dirs, expected_n, step, max_steps)
    hist = [0] * (expected_n + 1)
    train_groups = 0
    other = 0
    val_like = col["val_like"]
    solve_rates: list[float] = []
    for rs in col["groups"].values():
        n = len(rs)
        correct = sum(1 for r in rs if r is not None and r > 0)
        if n == expected_n:
            hist[correct] += 1
            train_groups += 1
            solve_rates.append(correct / n)
        elif n == 1:
            val_like += 1
        else:
            other += 1
    return {
        "expected_n": expected_n,
        "histogram": hist,
        "train_groups": train_groups,
        "val_like_groups": val_like,
        "other_groups": other,
        "all_correct": hist[expected_n],
        "all_wrong": hist[0],
        "no_signal": hist[0] + hist[expected_n],
        "mean_solve_rate": (sum(solve_rates) / len(solve_rates))
        if solve_rates
        else None,
        "num_steps_scanned": len(col["scanned_steps"]),
        "truncated": col["truncated"],
    }


def _rollout_dist_series(
    exp_dirs: list[str],
    expected_n: int = 8,
    buckets: int = 24,
    max_steps: int = 100000,
    bucket_size: int = 0,
) -> dict:
    """How the correct-out-of-N distribution evolves over training. Training
    groups are ordered by step (≈ training progress) and split into chunks; each
    chunk returns its histogram + normalized fractions so the UI can stack them
    into an evolution chart.

    NOTE the x-axis is rollout-GENERATION order (the on-disk step_NNNN is a global
    sample index, NOT the training step — and async staleness can reorder it), so
    a chunk ≈ training progress but is not an exact training step. When
    `bucket_size` > 0 each chunk holds ~bucket_size prompt-groups (set it to
    train_batch_size so one bar ≈ one training step); otherwise the groups are
    split into `buckets` equal-count chunks."""
    col = _collect_groups(exp_dirs, expected_n, None, max_steps)
    order = {st: i for i, st in enumerate(col["scanned_steps"])}

    tg: list[tuple[int, str, int]] = []  # (step_order, step, correct)
    for (st, _task_id), rs in col["groups"].items():
        if len(rs) == expected_n:
            correct = sum(1 for r in rs if r is not None and r > 0)
            tg.append((order.get(st, 0), st, correct))
    tg.sort(key=lambda x: x[0])

    n = len(tg)
    if bucket_size and bucket_size > 0:
        # one bar ≈ bucket_size groups (≈ one training step). Cap bar count at
        # 200 so a huge run can't blow up the payload.
        b = max(1, min((n + bucket_size - 1) // bucket_size, 200))
    else:
        b = max(1, min(buckets, 100))
    out = []
    for i in range(b):
        chunk = tg[i * n // b : (i + 1) * n // b]
        if not chunk:
            continue
        hist = [0] * (expected_n + 1)
        for _o, _st, c in chunk:
            hist[c] += 1
        cnt = len(chunk)
        frac = [h / cnt for h in hist]
        mean_sr = sum(c for _o, _st, c in chunk) / (cnt * expected_n)
        out.append(
            {
                "idx": i,
                "count": cnt,
                "step_start": chunk[0][1],
                "step_end": chunk[-1][1],
                "histogram": hist,
                "frac": frac,
                "no_signal_frac": (hist[0] + hist[expected_n]) / cnt,
                "mean_solve_rate": mean_sr,
            }
        )
    return {
        "expected_n": expected_n,
        "buckets": out,
        "train_groups": n,
        "num_steps_scanned": len(col["scanned_steps"]),
        "truncated": col["truncated"],
    }


_VAL_REPO_RE = re.compile(r"-\d+$")


def _val_repo(task_id: str) -> str:
    """Capability bucket = the repo, e.g. django__django-16100 -> django__django."""
    return _VAL_REPO_RE.sub("", task_id)


# Validation dumps are named `step_{rollouter.global_steps:04d}` -- the ROLLOUT
# counter (bumped per generated trajectory, continuously, async), not the trainer
# step the metrics charts are keyed on. They diverge hard: rollout 2808 == train
# step 40. So the trial-backed panels labelled their events "step_2808" while the
# curve read "40", with nothing tying them together. Recover the pairing so events
# can carry a `train_step`:
#   - `[stitch] val_map: step_NNNN -> train_step M` lines, when a stitched log
#     states it outright (authoritative -- a stitched log may carry the two source
#     lines out of order, so it resolves the pairing at the source and emits this).
#   - otherwise pair in file order: verl logs `test_gen_batch ... 'validate': True,
#     'global_steps': X` when a val STARTS, then `step:N - rollouter/validate_time`
#     when it ENDS, so X pairs with the next val-result line.
# Events whose pairing cannot be established keep train_step = None; the panel
# falls back to the dir name rather than guessing.
VAL_MAP_RE = re.compile(r"val_map:\s*(step_\d+)\s*->\s*train_step\s*(\d+)")
TEST_GEN_RE = re.compile(
    r"test_gen_batch meta info:.*'validate':\s*True.*'global_steps':\s*(\d+)"
)
VAL_RESULT_RE = re.compile(r"step:(\d+)\s*-.*rollouter/validate_time")


def _val_step_train_map(log_path: str | None) -> dict[str, int]:
    """{'step_2808': 40, ...} -- val dump dir name -> trainer global_step."""
    out: dict[str, int] = {}
    if not log_path:
        return out
    pending: int | None = None
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = VAL_MAP_RE.search(line)
                if m:
                    out[m.group(1)] = int(m.group(2))  # explicit wins
                    continue
                m = TEST_GEN_RE.search(line)
                if m:
                    pending = int(m.group(1))
                    continue
                if pending is None:
                    continue
                m = VAL_RESULT_RE.search(line)
                if m:
                    out.setdefault(f"step_{pending:04d}", int(m.group(1)))
                    pending = None
    except OSError:
        pass
    return out


def _val_analysis(
    exp_dirs: list[str], expected_n: int = 8, train_steps: dict[str, int] | None = None
) -> dict:
    """Validation analysis: validation runs the whole val set (n=1 per task)
    periodically, dumped as large step_NNNN dirs. We detect those val events,
    compute per-repo (capability) solve rates per event, and diff the FIRST vs
    LAST val to show per-capability and per-task change before/after training."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    steps = _list_steps_multi(exp_dirs)

    def _count(st):
        n = 0
        for e in exp_dirs:
            try:
                with os.scandir(os.path.join(e, st)) as it:
                    n += sum(1 for _ in it)
            except (FileNotFoundError, NotADirectoryError):
                pass
        return st, n

    with ThreadPoolExecutor(max_workers=32) as ex:
        counts = dict(ex.map(_count, steps))
    threshold = max(50, expected_n * 4)
    val_steps = sorted(st for st, n in counts.items() if n > threshold)
    if not val_steps:
        return {"expected_n": expected_n, "repos": [], "events": [], "tasks": [], "num_events": 0}

    # task_id -> [task_dir, ...] per val event. Multi-node async stamps a
    # separate exp-dir per node, and the same val task can land on more than one
    # node (retry / overlap). Collect ALL sibling copies per task instead of
    # overwriting to the last one — otherwise a task solved on node B but failed
    # on node A is miscounted as unsolved, systematically undercounting the val
    # solve rate on multi-node runs.
    event_tasks: dict[str, dict[str, list[str]]] = {}
    for st in val_steps:
        d: dict[str, list[str]] = defaultdict(list)
        for e in exp_dirs:
            try:
                with os.scandir(os.path.join(e, st)) as it:
                    for de in it:
                        d[_strip_trial_hash(de.name)].append(de.path)
            except (FileNotFoundError, NotADirectoryError):
                pass
        event_tasks[st] = dict(d)

    all_dirs = list(
        {td for d in event_tasks.values() for tds in d.values() for td in tds}
    )
    with ThreadPoolExecutor(max_workers=32) as ex:
        rew = dict(zip(all_dirs, ex.map(_read_reward, all_dirs)))

    def solved_of(tds):
        # pass@1: count ONE attempt per task (first copy), NOT best-of. Async val
        # re-dispatches ~15% of tasks to a 2nd node for coverage; best-of over those
        # extra copies inflates the rate (+~0.02) and varies run-to-run. tds[0] = a
        # single attempt, consistent with the trainer's logged val-core mean@1.
        r = rew.get(tds[0]) if tds else None
        return 1 if (r is not None and r > 0) else 0

    n_events = len(val_steps)
    # cross-event presence / solves per task
    appeared: dict[str, int] = defaultdict(int)
    solved_across: dict[str, int] = defaultdict(int)
    for st in val_steps:
        for tid, td in event_tasks[st].items():
            appeared[tid] += 1
            solved_across[tid] += solved_of(td)
    # canonical val set = tasks present in >= half the events. The val set is a
    # fixed ~500; a handful of stragglers leak into a single eval dump and would
    # otherwise inflate the count (e.g. 508).
    core = {tid for tid in appeared if appeared[tid] * 2 >= n_events}
    dropped = len(appeared) - len(core)

    events = []
    for i, st in enumerate(val_steps):
        repo_agg: dict[str, list] = defaultdict(lambda: [0, 0])  # repo -> [solved, total]
        solved = tot = 0
        for tid, td in event_tasks[st].items():
            if tid not in core:
                continue
            ok = solved_of(td)
            ra = repo_agg[_val_repo(tid)]
            ra[0] += ok
            ra[1] += 1
            solved += ok
            tot += 1
        events.append(
            {
                "idx": i,
                "step": st,
                "train_step": (train_steps or {}).get(st),
                "num_tasks": tot,
                "solved": solved,
                "mean_solve_rate": (solved / tot) if tot else None,
                "repo_rates": {rp: (s / t if t else None) for rp, (s, t) in repo_agg.items()},
                "repo_counts": {rp: t for rp, (_s, t) in repo_agg.items()},
                "repo_solved": {rp: s for rp, (s, _t) in repo_agg.items()},
            }
        )

    last = events[-1]
    repos = sorted(last["repo_counts"], key=lambda r: -last["repo_counts"][r])

    events_histogram = [0] * (n_events + 1)
    for tid in core:
        events_histogram[solved_across[tid]] += 1

    first_st, last_st = val_steps[0], val_steps[-1]
    fd, ld = event_tasks[first_st], event_tasks[last_st]
    tasks = []
    for tid in core:
        first = solved_of(fd[tid]) if tid in fd else None
        last_v = solved_of(ld[tid]) if tid in ld else None
        tasks.append(
            {
                "task": tid,
                "repo": _val_repo(tid),
                "first": first,
                "last": last_v,
                "delta": (last_v - first) if (first is not None and last_v is not None) else None,
                "solved_events": solved_across[tid],
                "appeared_events": appeared[tid],
            }
        )
    tasks.sort(key=lambda t: (t["last"] if t["last"] is not None else -1, t["task"]))

    return {
        "expected_n": expected_n,
        "repos": repos,
        "events": events,
        "tasks": tasks,
        "num_events": n_events,
        "unique_tasks": len(core),
        "dropped_stragglers": dropped,
        "events_histogram": events_histogram,
        "first_step": first_st,
        "last_step": last_st,
    }


# ---------------------------------------------------------------------------
# Validation failure-mode classification
# ---------------------------------------------------------------------------
# The agent (OpenHands-SDK scaffold) emits a human-readable log at
# agent/openhands_sdk.txt whose only action type in this run is FileEditorAction
# (it never runs tests). We classify each FAILED val rollout into the failure
# modes below. The C1(self_verif) vs C2(wrong_fix) split needs an LLM to compare
# the model patch against gold, so it is NOT computed here: deterministically we
# emit the combined "gold_edited_failed" bucket and let an offline judgment file
# (webui/val_judgments/<run>__<step>.json) override it when present.
VAL_FAILURE_CATS = [
    "success",
    "self_verif",        # C1: edited right place, plausible near-miss, unverified (LLM only)
    "wrong_fix",         # C2: edited right place, patch clearly wrong/incomplete (LLM only)
    "gold_edited_failed",  # C1+C2 combined (deterministic fallback)
    "file_loc_fail",     # C4: only edited non-gold files
    "no_impl",           # C5: no real code edit (explored/described, never patched)
    "dead_loop",         # C3: repetition collapse / budget burned, no progress
    "infra",             # patch failed to apply / missing report (env noise)
]

_GOLD_FILE_CACHE: dict[str, list[str]] = {}
_GOLD_DIFF_RE = re.compile(r"^\+\+\+ b/(\S+)", re.M)
_FE_ACTION_RE = re.compile(
    r'Action:\s*FileEditorAction\s*\n+\s*Arguments:\s*\n(.*?)(?=\nTokens:|\nObservation|\n\s*kind:)',
    re.S,
)
_FE_CMD_RE = re.compile(r'command:\s*"([^"]+)"')
_FE_PATH_RE = re.compile(r'path:\s*"([^"]+)"')
_OUT_TOK_RE = re.compile(r'output\s+([\d.]+)K')


def _val_cot_trend(
    exp_dirs: list[str],
    expected_n: int = 8,
    train_steps: dict[str, int] | None = None,
    sample: int = 120,
) -> dict:
    """CoT share of the val response, per val event.

    Val runs the whole set at n=1, so a per-event mean would seem straightforward
    — but CoT share varies far more between tasks than it does between steps, and
    the task list drifts across dumps (stragglers, retries, a node dropping out).
    A naive per-event mean therefore tracks which tasks happened to land in the
    dump, not what the model is doing. So the cohort is fixed first: the canonical
    val set (present in >= half the events), sorted, truncated to `sample`. Every
    event is then measured over that same cohort, and events missing too much of
    it are reported with their coverage rather than silently averaged.

    Ratio is CoT chars over response chars, where the response includes
    `tool_calls[].function.arguments` — see _traj_metrics: scoring against
    `content` alone reports roughly double the true share for a coding agent.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    steps = _list_steps_multi(exp_dirs)

    def _count(st):
        n = 0
        for e in exp_dirs:
            try:
                with os.scandir(os.path.join(e, st)) as it:
                    n += sum(1 for _ in it)
            except (FileNotFoundError, NotADirectoryError):
                pass
        return st, n

    with ThreadPoolExecutor(max_workers=32) as ex:
        counts = dict(ex.map(_count, steps))
    threshold = max(50, expected_n * 4)
    val_steps = sorted(st for st, n in counts.items() if n > threshold)
    if not val_steps:
        return {"expected_n": expected_n, "points": [], "num_events": 0, "cohort_size": 0}

    # Same sibling-aware collection as _val_analysis: one task can land on more
    # than one node, and we take copy [0] to stay pass@1-consistent with it.
    event_tasks: dict[str, dict[str, list[str]]] = {}
    for st in val_steps:
        d: dict[str, list[str]] = defaultdict(list)
        for e in exp_dirs:
            try:
                with os.scandir(os.path.join(e, st)) as it:
                    for de in it:
                        d[_strip_trial_hash(de.name)].append(de.path)
            except (FileNotFoundError, NotADirectoryError):
                pass
        event_tasks[st] = dict(d)

    appeared: dict[str, int] = defaultdict(int)
    for st in val_steps:
        for tid in event_tasks[st]:
            appeared[tid] += 1
    half = max(1, len(val_steps) // 2)
    cohort = sorted(tid for tid, n in appeared.items() if n >= half)[: max(1, sample)]
    if not cohort:
        return {"expected_n": expected_n, "points": [], "num_events": len(val_steps), "cohort_size": 0}

    # One read per (event, task): first copy only.
    todo: list[tuple[str, str, str]] = []
    for st in val_steps:
        for tid in cohort:
            tds = event_tasks[st].get(tid)
            if tds:
                todo.append((st, tid, sorted(tds)[0]))

    with ThreadPoolExecutor(max_workers=32) as ex:
        mets = list(ex.map(lambda x: _traj_metrics(x[2]), todo))

    per_event: dict[str, list[dict]] = defaultdict(list)
    for (st, _tid, _td), m in zip(todo, mets):
        if m and m.get("resp_chars"):
            per_event[st].append(m)

    points = []
    for st in val_steps:
        ms = per_event.get(st) or []
        if not ms:
            points.append(
                {
                    "step": st,
                    "train_step": (train_steps or {}).get(st),
                    "cot_ratio": None,
                    "n": 0,
                    "coverage": 0.0,
                }
            )
            continue
        ratios = [m["cot_chars"] / m["resp_chars"] for m in ms if m.get("cot_chars") is not None]
        tot_cot = sum(m.get("cot_chars") or 0 for m in ms)
        tot_resp = sum(m.get("resp_chars") or 0 for m in ms)
        points.append(
            {
                "step": st,
                "train_step": (train_steps or {}).get(st),
                # per-task mean: what a typical task spends on reasoning
                "cot_ratio": (sum(ratios) / len(ratios)) if ratios else None,
                # char-weighted: what share of all emitted text was reasoning
                "cot_ratio_weighted": (tot_cot / tot_resp) if tot_resp else None,
                "cot_chars": tot_cot / len(ms),
                "resp_chars": tot_resp / len(ms),
                "turns": _mean([m.get("turns") for m in ms]),
                "n": len(ms),
                "coverage": len(ms) / len(cohort),
            }
        )

    points.sort(key=lambda p: (p.get("train_step") is None, p.get("train_step"), p["step"]))
    return {
        "expected_n": expected_n,
        "num_events": len(val_steps),
        "cohort_size": len(cohort),
        "points": points,
    }


def _gold_files_for_task(task_dir: str) -> list[str]:
    """Files the reference (gold) patch touches, parsed from the task's
    solution/solve.sh. The task source dir is read from result.json so we don't
    hardcode the val-set root."""
    src = None
    rj = os.path.join(task_dir, "result.json")
    try:
        with open(rj) as f:
            d = json.load(f)
        src = (d.get("task_id") or {}).get("path") or d.get("config", {}).get("task", {}).get("path")
    except Exception:
        pass
    if not src:
        return []
    if src in _GOLD_FILE_CACHE:
        return _GOLD_FILE_CACHE[src]
    files: list[str] = []
    try:
        txt = open(os.path.join(src, "solution", "solve.sh"), errors="replace").read()
        files = sorted(set(_GOLD_DIFF_RE.findall(txt)))
    except Exception:
        files = []
    _GOLD_FILE_CACHE[src] = files
    return files


def _classify_val_trajectory(task_dir: str, gold_files: list[str]) -> str:
    """Deterministic failure-mode label for one task dir (no LLM)."""
    reward = _read_reward(task_dir)
    resolved = None
    patch_applied = None
    rp = os.path.join(task_dir, "verifier", "report.json")
    has_report = os.path.isfile(rp)
    if has_report:
        try:
            with open(rp) as f:
                rj = json.load(f)
            rep = next(iter(rj.values())) if rj else None
            if isinstance(rep, dict):
                resolved = rep.get("resolved")
                patch_applied = rep.get("patch_successfully_applied")
        except Exception:
            pass
    if (reward is not None and reward > 0) or resolved:
        return "success"
    if not has_report:
        return "infra"
    if patch_applied is False:
        return "infra"

    # parse the trajectory log. Cap the read: normal trajectories are well
    # under this; only repetition-collapse logs balloon to multiple MB, and the
    # repetition (-> dead_loop) and the agent's edits are already present in the
    # head, so a cap bounds per-file I/O without changing the classification.
    tpath = os.path.join(task_dir, "agent", "openhands_sdk.txt")
    try:
        with open(tpath, errors="replace") as f:
            txt = f.read(1_500_000)
    except Exception:
        return "infra"

    gbase = {os.path.basename(g) for g in gold_files}
    n_edit = 0
    gold_edit = False
    for m in _FE_ACTION_RE.finditer(txt):
        args = m.group(1)
        cm = _FE_CMD_RE.search(args)
        cmd = cm.group(1) if cm else ""
        pm = _FE_PATH_RE.search(args)
        rel = (pm.group(1) if pm else "").replace("/testbed/", "").lstrip("/")
        if cmd in ("str_replace", "insert", "append"):
            n_edit += 1
            if rel in gold_files or os.path.basename(rel) in gbase:
                gold_edit = True
        elif cmd == "create":
            if rel in gold_files:
                gold_edit = True

    # repetition collapse / budget burn
    out_tokens = 0.0
    outs = _OUT_TOK_RE.findall(txt)
    if outs:
        try:
            out_tokens = float(outs[-1]) * 1000
        except Exception:
            pass
    best = run = 1
    prev = None
    for ln in txt.split("\n"):
        s = ln.strip()
        if s and s == prev:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
        prev = s
    if best >= 40 or out_tokens >= 60000:
        return "dead_loop"
    if gold_edit:
        return "gold_edited_failed"
    if n_edit > 0:
        return "file_loc_fail"
    return "no_impl"


def _valfail_cache_path(run_id: str, step: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".valcache")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{run_id}__{step}")
    return os.path.join(d, f"valfail_{safe}.json")


def _valfail_disk_get(run_id: str, step: str) -> dict | None:
    """Per-(run, step) failure-mode result persisted to disk. A val dump for a
    given step is immutable once written, so this never goes stale and survives
    server restarts (the cold compute reads ~500 trajectory files, ~40s)."""
    try:
        with open(_valfail_cache_path(run_id, step)) as f:
            return json.load(f)
    except Exception:
        return None


def _valfail_disk_put(run_id: str, step: str, result: dict) -> None:
    try:
        p = _valfail_cache_path(run_id, step)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, p)
    except Exception:
        pass


def _load_val_judgments(run_id: str, step: str) -> dict[str, str]:
    """Optional offline LLM judgments that split gold_edited_failed into
    self_verif/wrong_fix (and refine others). File:
    webui/val_judgments/<run_id>__<step>.json = {task_id: cat}."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_judgments", f"{run_id}__{step}.json")
    try:
        with open(p) as f:
            d = json.load(f)
        return {k: str(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except Exception:
        return {}


def _val_failure_modes(
    exp_dirs: list[str], run_id: str, expected_n: int = 8, step: str | None = None
) -> dict:
    """Classify the failure modes of one validation event (default: the LAST,
    i.e. final-model, val dump). Returns per-category counts and a per-task list."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    steps = _list_steps_multi(exp_dirs)

    def _count(st):
        n = 0
        for e in exp_dirs:
            try:
                with os.scandir(os.path.join(e, st)) as it:
                    n += sum(1 for _ in it)
            except (FileNotFoundError, NotADirectoryError):
                pass
        return st, n

    with ThreadPoolExecutor(max_workers=32) as ex:
        counts = dict(ex.map(_count, steps))
    threshold = max(50, expected_n * 4)
    val_steps = sorted(st for st, n in counts.items() if n > threshold)
    if not val_steps:
        return {"val_steps": [], "step": None, "categories": VAL_FAILURE_CATS, "counts": {}, "tasks": [], "num_tasks": 0}

    target = step if (step and step in val_steps) else val_steps[-1]

    # Canonical val set = task ids present in >= half the val events (same rule as
    # val-analysis). Drops stragglers: a training rollout stamped with the same
    # global_steps as a val dump leaks ONE task into a single event's dir (an
    # openswe_filtered task in the official-val dump) -> otherwise shows 501/500.
    appeared: dict[str, int] = defaultdict(int)
    for st in val_steps:
        seen_st: set[str] = set()
        for e in exp_dirs:
            try:
                with os.scandir(os.path.join(e, st)) as it:
                    for de in it:
                        seen_st.add(_strip_trial_hash(de.name))
            except (FileNotFoundError, NotADirectoryError):
                pass
        for tid in seen_st:
            appeared[tid] += 1
    core = {tid for tid, c in appeared.items() if c * 2 >= len(val_steps)}

    # task_id -> ONE attempt per task (first copy) = pass@1, NOT best-of. Async
    # val re-dispatches ~15% of tasks to a 2nd node for coverage; counting best-of
    # across those extra copies inflates the rate and varies run-to-run. One copy
    # per task = a single attempt, consistent with the trainer's logged val-core.
    # Restricted to the canonical core set (drops stragglers).
    tdirs: dict[str, str] = {}
    for e in exp_dirs:
        try:
            with os.scandir(os.path.join(e, target)) as it:
                for de in it:
                    tid = _strip_trial_hash(de.name)
                    if tid in core:
                        tdirs.setdefault(tid, de.path)
        except (FileNotFoundError, NotADirectoryError):
            pass

    items = list(tdirs.items())

    # Serve the per-step result from disk ONLY if it still covers what is on disk.
    # A fully-async MULTI-NODE val dump is NOT immutable when first written: each
    # node writes to its own sibling exp-dir progressively, so the task set keeps
    # growing for hours. A cache written mid-val froze an incomplete/biased
    # snapshot (e.g. 64/500, infra-heavy). Trust it only when its num_tasks is not
    # smaller than the current unique-task count; otherwise recompute.
    disk = _valfail_disk_get(run_id, target)
    if disk is not None and disk.get("num_tasks", 0) >= len(items):
        disk["val_steps"] = val_steps
        disk["from_cache"] = True
        return disk

    def _one(item):
        tid, td = item
        gold = _gold_files_for_task(td)
        return tid, _classify_val_trajectory(td, gold)

    with ThreadPoolExecutor(max_workers=48) as ex:
        det = dict(ex.map(_one, items))

    judgments = _load_val_judgments(run_id, target)
    counts: dict[str, int] = defaultdict(int)
    tasks = []
    for tid, td in items:
        cat = det[tid]
        # overlay offline judgment for any non-trivial (failed) task
        if cat not in ("success", "infra") and tid in judgments:
            cat = judgments[tid]
        counts[cat] += 1
        tasks.append({"task": tid, "repo": _val_repo(tid), "cat": cat,
                      "step": target, "task_dir": td})
    tasks.sort(key=lambda t: (t["cat"], t["task"]))
    result = {
        "val_steps": val_steps,
        "step": target,
        "categories": VAL_FAILURE_CATS,
        "counts": dict(counts),
        "num_tasks": len(items),
        "has_judgments": bool(judgments),
        "tasks": tasks,
    }
    _valfail_disk_put(run_id, target, result)
    return result


def _traj_metrics(task_dir: str) -> dict | None:
    """Shape of ONE rollout: how many turns it took, how long the response was,
    how much of it was chain-of-thought.

    Sources, cheapest first:
      turns / cot / response chars — `proxy_trajectory.json`, the LLM-level view
        (`messages_snapshot`), so a "turn" is one assistant completion, not a
        harness step. Reasoning is emitted with only a CLOSING `</think>` (the
        opening tag lives in the prompt template), so CoT is everything before it.
      response tokens — `result.json`'s `agent_result.n_output_tokens`, the
        tokenizer's own count, which is what the training-side response_length
        metric measures. Chars are reported too since CoT has no token count.

    The response is `content` PLUS the tool-call arguments: a coding agent emits
    its edits as `tool_calls[].function.arguments` (the whole patch body), which
    is a sibling field of `content`, not part of it. Those arguments are ~as many
    chars as the reasoning here, so scoring CoT against `content` alone reported
    ~89% when the real share is ~45% — and disagreed with the token count, which
    does include tool calls.
    """
    out: dict = {}
    try:
        with open(os.path.join(task_dir, "result.json"), errors="replace") as f:
            ar = (json.load(f) or {}).get("agent_result") or {}
        out["out_tok"] = ar.get("n_output_tokens")
        # Summed over every LLM call in the trajectory, not one call's context —
        # divide by turns for the average context size per call.
        out["in_tok"] = ar.get("n_input_tokens")
    except Exception:
        return None
    try:
        with open(os.path.join(task_dir, "proxy_trajectory.json"), errors="replace") as f:
            snap = (json.load(f) or {}).get("messages_snapshot") or []
        cot = resp = turns = tool = 0
        for m in snap:
            if m.get("role") != "assistant":
                continue
            turns += 1
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
            c = c or ""
            resp += len(c)
            if "</think>" in c:
                cot += len(c.split("</think>", 1)[0])
            for t in m.get("tool_calls") or []:
                a = (t.get("function") or {}).get("arguments")
                tool += len(a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
        out.update(
            turns=turns, cot_chars=cot, tool_chars=tool, resp_chars=resp + tool
        )
    except Exception:
        out.update(turns=None, cot_chars=None, tool_chars=None, resp_chars=None)
    return out


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v)) if v else None


def _task_traj_stats(
    exp_dirs: list[str], expected_n: int = 8, max_steps: int = 100000
) -> dict:
    """First-epoch vs last-epoch trajectory SHAPE per task, to sit alongside the
    solve-rate grid: a task whose solve rate held steady while its turn count
    halved got genuinely better, and one that improved only by tripling its CoT
    got more expensive. Solve rate alone shows neither.

    Only tasks rolled out in >= 2 epochs can be compared. Each epoch's value is
    the mean over that epoch's rollouts, so a single degenerate rollout doesn't
    swing the cell."""
    from collections import defaultdict

    col = _collect_groups(exp_dirs, expected_n, None, max_steps)
    order = {st: i for i, st in enumerate(col["scanned_steps"])}
    gd = col.get("group_dirs") or {}

    by_task: dict[str, list[tuple]] = defaultdict(list)
    for (st, task_id), tds in gd.items():
        by_task[task_id].append((order.get(st, 0), st, tds))

    # Only read trials we will actually compare — first + last epoch of the
    # multi-epoch tasks. On a long run that is a small slice of the whole tree.
    todo: dict[str, list[tuple]] = {}
    wanted: set[str] = set()
    for task_id, occ in by_task.items():
        if len(occ) < 2:
            continue
        occ.sort(key=lambda x: x[0])
        todo[task_id] = [occ[0], occ[-1]]
        wanted.update(occ[0][2])
        wanted.update(occ[-1][2])

    from concurrent.futures import ThreadPoolExecutor

    dirs = sorted(wanted)
    met: dict[str, dict] = {}
    if dirs:
        with ThreadPoolExecutor(max_workers=32) as ex:
            for td, m in zip(dirs, ex.map(_traj_metrics, dirs)):
                if m:
                    met[td] = m

    def agg(tds: list[str]) -> dict:
        ms = [met[td] for td in tds if td in met]
        return {
            "turns": _mean([m.get("turns") for m in ms]),
            "in_tok": _mean([m.get("in_tok") for m in ms]),
            "out_tok": _mean([m.get("out_tok") for m in ms]),
            "cot_chars": _mean([m.get("cot_chars") for m in ms]),
            "tool_chars": _mean([m.get("tool_chars") for m in ms]),
            "resp_chars": _mean([m.get("resp_chars") for m in ms]),
            "n": len(ms),
        }

    KEYS = ("turns", "in_tok", "out_tok", "cot_chars", "tool_chars", "resp_chars")
    tasks = []
    for task_id, ((_o1, st1, tds1), (_o2, st2, tds2)) in todo.items():
        a, b = agg(tds1), agg(tds2)
        if not a["n"] or not b["n"]:
            continue
        row = {"task": task_id, "first_step": st1, "last_step": st2,
               "first_n": a["n"], "last_n": b["n"]}
        for k in KEYS:
            row[f"first_{k}"] = a[k]
            row[f"last_{k}"] = b[k]
            row[f"d_{k}"] = (b[k] - a[k]) if (a[k] is not None and b[k] is not None) else None
        # CoT share of the response — unit-free, so it survives the fact that CoT
        # is measured in chars while the token count covers the whole response.
        for side, m in (("first", a), ("last", b)):
            rc, cc = m["resp_chars"], m["cot_chars"]
            row[f"{side}_cot_ratio"] = (cc / rc) if (rc and cc is not None) else None
        row["d_cot_ratio"] = (
            (row["last_cot_ratio"] - row["first_cot_ratio"])
            if (row["first_cot_ratio"] is not None and row["last_cot_ratio"] is not None)
            else None
        )
        tasks.append(row)

    tasks.sort(key=lambda t: t["task"])
    summary = {}
    for k in KEYS + ("cot_ratio",):
        f = _mean([t.get(f"first_{k}") for t in tasks])
        l = _mean([t.get(f"last_{k}") for t in tasks])
        summary[k] = {"first": f, "last": l,
                      "delta": (l - f) if (f is not None and l is not None) else None,
                      "pct": ((l - f) / f) if (f and l is not None) else None}
    return {
        "expected_n": expected_n,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "trials_read": len(met),
        "summary": summary,
        "num_steps_scanned": len(col["scanned_steps"]),
        "truncated": col["truncated"],
    }


def _task_solve_stats(
    exp_dirs: list[str], expected_n: int = 8, max_steps: int = 100000
) -> dict:
    """Per-task solve stats aggregated across the run.

    A task (prompt) is re-rolled out once per epoch (expected_n rollouts each
    time), so each (step, task_id) group is one epoch's worth of rollouts. We
    therefore report the per-epoch solve-rate sequence per task (groups ordered
    by step) and compare the FIRST epoch vs the LAST epoch — not an arbitrary
    global early/late split."""
    from collections import defaultdict

    col = _collect_groups(exp_dirs, expected_n, None, max_steps)
    order = {st: i for i, st in enumerate(col["scanned_steps"])}

    # task_id -> list of (step_order, step, n, correct), one entry per epoch
    by_task: dict[str, list[tuple]] = defaultdict(list)
    for (st, task_id), rs in col["groups"].items():
        n = len(rs)
        correct = sum(1 for r in rs if r is not None and r > 0)
        by_task[task_id].append((order.get(st, 0), st, n, correct))

    tasks = []
    for task_id, occ in by_task.items():
        occ.sort(key=lambda x: x[0])
        epochs = [
            {"step": st, "n": n, "solved": c, "rate": (c / n if n else None)}
            for (_o, st, n, c) in occ
        ]
        total_n = sum(e["n"] for e in epochs)
        total_solved = sum(e["solved"] for e in epochs)
        first = epochs[0]["rate"]
        last = epochs[-1]["rate"]
        delta = (
            (last - first)
            if (len(epochs) >= 2 and first is not None and last is not None)
            else None
        )
        tasks.append(
            {
                "task": task_id,
                "epochs": epochs,
                "num_epochs": len(epochs),
                "rollouts": total_n,
                "solved": total_solved,
                "solve_rate": (total_solved / total_n if total_n else None),
                "first_rate": first,
                "last_rate": last,
                "first_step": epochs[0]["step"],
                "last_step": epochs[-1]["step"],
                "delta": delta,
            }
        )
    tasks.sort(key=lambda t: (t["solve_rate"] if t["solve_rate"] is not None else -1))

    never = sum(1 for t in tasks if t["solve_rate"] == 0)
    always = sum(1 for t in tasks if t["solve_rate"] == 1)
    multi = sum(1 for t in tasks if t["num_epochs"] >= 2)
    improved = sum(1 for t in tasks if t["delta"] is not None and t["delta"] > 0)
    regressed = sum(1 for t in tasks if t["delta"] is not None and t["delta"] < 0)
    return {
        "expected_n": expected_n,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "never_solved": never,
        "always_solved": always,
        "multi_epoch": multi,
        "improved": improved,
        "regressed": regressed,
        "num_steps_scanned": len(col["scanned_steps"]),
        "truncated": col["truncated"],
    }


# Full ANSI/terminal escape stripping for trajectory text. The agent runs real
# shell commands, so observations carry CSI colour/cursor codes (…m), private
# modes (e.g. \x1b[?2004l bracketed-paste), OSC sequences, and stray control
# bytes — all render as garbage in the UI. (The log parser's ANSI_RE only
# handles colour codes; this is the broad version for trajectory content.)
_TERM_ESC_RE = re.compile(
    r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))"
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(s: str) -> str:
    if not s:
        return s
    s = _TERM_ESC_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_RE.sub("", s)
    return s


def _strip_think(s: str) -> str:
    """Drop the dangling </think> marker from a raw assistant message.

    The Qwen3.5 chat template pre-fills the OPENING <think>, so raw content is
    `<cot></think><visible answer>`; the tag itself is template plumbing, not
    something the reader should see. Keeps both halves (blank-line separated)
    since the thought panel shows the whole message."""
    if "</think>" not in s:
        return s
    pre, post = s.split("</think>", 1)
    post = post.strip()
    pre = pre.rstrip()
    return pre + "\n\n" + post if post else pre


def _parse_trajectory_json(
    path: str, max_content: int = 800
) -> tuple[str, str, list[dict]]:
    """Parse the new event-stream agent/trajectory.json format.

    `steps` is a flat list of events: source in {system, user, agent}.
    A single `agent` event is one turn (message=thought, tool_calls=actions,
    observation.results=observations)."""
    system_prompt = ""
    problem_statement = ""
    turns: list[dict] = []
    try:
        with open(path, "r", errors="replace") as f:
            data = json.load(f)
    except Exception:
        return "", "", []

    turn_idx = 0
    for ev in data.get("steps", []):
        src = ev.get("source")
        msg = _clean_text(ev.get("message", "") or "")
        if src == "system":
            if not system_prompt:
                system_prompt = msg
            continue
        if src == "user":
            if not problem_statement:
                problem_statement = msg
            continue

        actions = []
        for tc in ev.get("tool_calls") or []:
            args = tc.get("arguments")
            args_str = (
                args
                if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False, indent=1)
            )
            args_str = _clean_text(args_str)
            if len(args_str) > max_content:
                args_str = args_str[:max_content] + "..."
            actions.append(
                {"name": tc.get("function_name", ""), "arguments": args_str}
            )

        observations = []
        obs = ev.get("observation") or {}
        if isinstance(obs, dict):
            for r in obs.get("results", []):
                c = _clean_text(str(r.get("content", "")))
                if len(c) > max_content:
                    c = c[:max_content] + "..."
                observations.append(c)

        thought = _strip_think(msg)
        if len(thought) > max_content:
            thought = thought[:max_content] + "..."

        turns.append(
            {
                "turn": turn_idx,
                "thought": thought,
                "actions": actions,
                "observations": observations,
                "usage": {"prompt": 0, "completion": 0},
                "duration_ms": None,
            }
        )
        turn_idx += 1
    return system_prompt, problem_statement, turns


def _load_trajectory(
    task_dir: str, max_content: int = 800
) -> tuple[str, str, list[dict]]:
    """Load a trajectory from a task dir, preferring the new json format."""
    j = os.path.join(task_dir, "agent", "trajectory.json")
    if os.path.isfile(j):
        return _parse_trajectory_json(j, max_content)
    jl = os.path.join(task_dir, "agent", "litellm-trajectory.jsonl")
    if os.path.isfile(jl):
        return _parse_trajectory(jl, max_content)
    return "", "", []


# ---------------------------------------------------------------------------
# Uploaded snapshots (colleague-supplied single-step result bundles)
#
# A bundle (3 files: brief jsonl + full trajectories + token tensors) is
# materialized by ingest_snapshot.py into the on-disk trial-dir layout, so it
# plugs into the existing run-centric analysis endpoints unchanged. A snapshot
# is exposed as a run with id "snap__<slug>" and source "snapshot".
# ---------------------------------------------------------------------------
SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
SNAP_PREFIX = "snap__"

_ingest_jobs: dict[str, dict] = {}
_ingest_lock = threading.Lock()


def _snap_slug(run_id: str) -> str:
    return run_id[len(SNAP_PREFIX):] if run_id.startswith(SNAP_PREFIX) else run_id


def _snap_dir(run_id: str) -> str:
    return os.path.join(SNAPSHOTS_DIR, _snap_slug(run_id))


def _load_snapshot_meta(slug: str) -> dict | None:
    sj = os.path.join(SNAPSHOTS_DIR, slug, "snapshot.json")
    if not os.path.isfile(sj):
        return None
    try:
        with open(sj) as f:
            return json.load(f)
    except Exception:
        return None


def _list_snapshot_runs() -> list[dict]:
    runs: list[dict] = []
    if not os.path.isdir(SNAPSHOTS_DIR):
        return runs
    for slug in sorted(os.listdir(SNAPSHOTS_DIR)):
        d = os.path.join(SNAPSHOTS_DIR, slug)
        if not os.path.isdir(d):
            continue
        meta = _load_snapshot_meta(slug)
        if not meta:
            continue
        summary = meta.get("summary", {})
        runs.append(
            {
                "id": f"{SNAP_PREFIX}{slug}",
                "name": summary.get("display_name", slug),
                "state": "snapshot",
                "source": "snapshot",
                "created_at": datetime.fromtimestamp(
                    os.path.getmtime(d), tz=timezone.utc
                ).isoformat(),
                "summary": summary,
            }
        )
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


def _detect_exp_dirs(path: str) -> list[str] | None:
    """If `path` is a REAL harbor run directory (not a 3-file bundle), return
    the exp-dir(s) to reference. An exp-dir contains ``step_NNNN/<task>/`` with
    an ``agent/`` or ``verifier/`` subdir. Two shapes are accepted:
      1. `path` is itself an exp-dir            -> [path]
      2. `path` is a project dir of exp-dirs    -> [child, ...]  (async multi-run)
    Returns None when `path` doesn't look like a run dir (caller falls back to
    the rollout_trajectories bundle ingest)."""
    if not os.path.isdir(path):
        return None

    def _is_exp_dir(d: str) -> bool:
        try:
            with os.scandir(d) as it:
                for de in it:
                    if not (de.is_dir() and de.name.startswith("step_")):
                        continue
                    with os.scandir(de.path) as it2:
                        for t in it2:
                            if t.is_dir() and (
                                os.path.isdir(os.path.join(t.path, "agent"))
                                or os.path.isdir(os.path.join(t.path, "verifier"))
                            ):
                                return True
        except OSError:
            pass
        return False

    if _is_exp_dir(path):
        return [os.path.abspath(path)]
    subs: list[str] = []
    try:
        with os.scandir(path) as it:
            for de in it:
                if de.is_dir() and _is_exp_dir(de.path):
                    subs.append(os.path.abspath(de.path))
    except OSError:
        pass
    return sorted(subs) or None


def _summarize_exp_dirs(exp_dirs: list[str], progress=None) -> dict:
    """Cheap summary for a referenced run: step/group/sample counts + mean
    reward. Rewards read in parallel (the trials FS is slow)."""
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor

    steps = _list_steps_multi(exp_dirs)
    pairs = [(st, e) for st in steps for e in exp_dirs]

    def _scan(pair):
        st, e = pair
        out = []
        try:
            with os.scandir(os.path.join(e, st)) as it:
                for de in it:
                    if de.is_dir():
                        out.append((st, _strip_trial_hash(de.name), de.path))
        except OSError:
            pass
        return out

    groups: dict[tuple, list[str]] = defaultdict(list)
    if pairs:
        with ThreadPoolExecutor(max_workers=32) as ex:
            for i, out in enumerate(ex.map(_scan, pairs)):
                for st, key, td in out:
                    groups[(st, key)].append(td)
                if progress:
                    progress(i + 1, len(pairs) + 1, "scanning steps")
    all_td = [td for tds in groups.values() for td in tds]
    rewards: dict[str, float | None] = {}
    if all_td:
        with ThreadPoolExecutor(max_workers=32) as ex:
            for td, rv in zip(all_td, ex.map(_read_reward, all_td)):
                rewards[td] = rv
    vals = [rewards[td] for td in all_td if rewards.get(td) is not None]
    sizes = Counter(len(v) for v in groups.values())
    expected_n = sizes.most_common(1)[0][0] if sizes else 0
    zero_adv = sum(
        1
        for v in groups.values()
        if len({rewards.get(td) for td in v if rewards.get(td) is not None}) <= 1
    )
    repos = {k[1].split("__")[0] for k in groups.keys() if k[1]}
    if progress:
        progress(len(pairs) + 1, len(pairs) + 1, "done")
    return {
        "kind": "run_dir",
        "num_steps": len(steps),
        "num_groups": len(groups),
        "expected_n": expected_n,
        "num_samples": len(all_td),
        "mean_reward": (sum(vals) / len(vals)) if vals else 0.0,
        "zero_advantage_groups": zero_adv,
        "num_repos": len(repos),
        "resp_len": {"p50": None, "max": None},
        "mean_logprob": None,
        "has_token_file": False,
        "reward_source": "verifier (result.json/reward.txt)",
    }


def _run_ingest_job(job_id: str, bundle_path: str, name: str, log_path: str | None = None):
    import ingest_snapshot

    slug = _SAFE_SLUG.sub("_", name).strip("_") or "snapshot"
    out_dir = os.path.join(SNAPSHOTS_DIR, slug)
    # avoid clobbering an existing snapshot with the same name
    base = out_dir
    k = 2
    while os.path.exists(out_dir):
        out_dir = f"{base}_{k}"
        slug = os.path.basename(out_dir)
        k += 1

    def progress(done, total, msg):
        with _ingest_lock:
            j = _ingest_jobs.get(job_id, {})
            j.update({"done": done, "total": total, "message": msg})
            _ingest_jobs[job_id] = j

    try:
        os.makedirs(out_dir, exist_ok=True)
        with _ingest_lock:
            _ingest_jobs[job_id].update({"status": "running", "slug": slug})
        exp_dirs = _detect_exp_dirs(bundle_path)
        if exp_dirs:
            # Real run directory: reference in place (no data copy). The slug
            # dir holds only snapshot.json with absolute exp-dir paths.
            summary = _summarize_exp_dirs(exp_dirs, progress=progress)
            summary["display_name"] = name
            summary["has_log"] = bool(log_path)
            meta = {"source": "run_dir", "exp_dirs": exp_dirs, "summary": summary}
            if log_path:
                meta["log_path"] = os.path.abspath(log_path)
            with open(os.path.join(out_dir, "snapshot.json"), "w") as f:
                json.dump(meta, f, indent=2)
        else:
            summary = ingest_snapshot.ingest(bundle_path, out_dir, progress=progress)
            summary["display_name"] = name
            # persist the display name into snapshot.json
            sj = os.path.join(out_dir, "snapshot.json")
            with open(sj) as f:
                data = json.load(f)
            data["summary"]["display_name"] = name
            with open(sj, "w") as f:
                json.dump(data, f, indent=2)
        with _ingest_lock:
            _ingest_jobs[job_id].update(
                {
                    "status": "done",
                    "run_id": f"{SNAP_PREFIX}{slug}",
                    "summary": summary,
                }
            )
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        with _ingest_lock:
            _ingest_jobs[job_id].update({"status": "error", "error": str(e)})


_SAFE_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


class DashboardHandler(SimpleHTTPRequestHandler):
    log_dirs: list[str] = []
    static_dir: str = ""
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_api_key: str = ""
    _run_path_cache: dict[str, str] = {}
    _exp_dirs_cache: dict[str, list[str]] = {}
    _analysis_empty_cache: dict[str, bool] = {}
    _rollout_dist_cache: dict[str, tuple[float, dict]] = {}

    def _exp_dirs_for(self, run_id: str) -> list[str]:
        """Resolve all harbor_trials exp-dirs for a run (async => several).

        First tries a direct <project>/<run_id> match; otherwise parses the
        run's console log for the exp-dirs it actually wrote to."""
        if run_id.startswith(SNAP_PREFIX):
            # Referenced real run directories (source="run_dir"): the snapshot
            # only stores absolute exp-dir paths in its meta — no data copy — so
            # read straight from the shared-storage originals.
            meta = _load_snapshot_meta(_snap_slug(run_id)) or {}
            ext = [d for d in (meta.get("exp_dirs") or []) if os.path.isdir(d)]
            if ext:
                return ext
            # ingested bundles: materialized trial-dir layout under the slug.
            trials = os.path.join(_snap_dir(run_id), "trials")
            return [trials] if os.path.isdir(trials) else []
        if run_id in self._exp_dirs_cache:
            return self._exp_dirs_cache[run_id]
        dirs: list[str] = []
        direct = _find_exp_dir(run_id)
        if direct:
            dirs = [direct]
        else:
            log_path = self._resolve_run_path(run_id)
            if log_path:
                dirs = _exp_dirs_from_log(log_path)
        # Recover sibling per-node exp-dirs the log-parse missed (multi-node
        # async stamps a distinct -HHMMSS per node; rollouts split across them).
        if dirs:
            dirs = _expand_sibling_exp_dirs(dirs)
        self._exp_dirs_cache[run_id] = dirs
        return dirs

    def _run_analysis_empty(self, run_id: str) -> bool:
        """True when a run has no on-disk trial dirs with step_ data, i.e. its
        analysis panels would render empty. Cached (class-level) — a finished
        run's trial dirs don't come back once cleaned."""
        if run_id in self._analysis_empty_cache:
            return self._analysis_empty_cache[run_id]
        dirs = self._exp_dirs_for(run_id)
        has = any(os.path.isdir(d) and _has_step_dirs(d) for d in dirs)
        self._analysis_empty_cache[run_id] = not has
        return not has

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/analysis/generate":
            return self._handle_analysis_generate()

        if path == "/api/snapshots/ingest":
            return self._handle_snapshot_ingest()

        if path == "/api/snapshots/delete":
            return self._handle_snapshot_delete()

        self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/config":
            has_logs = any(self.log_dirs)
            return self._json(
                {
                    "log_dirs": self.log_dirs,
                    "wandb_entity": self.wandb_entity,
                    "wandb_project": self.wandb_project,
                    "data_source": "both"
                    if self.wandb_api_key and has_logs
                    else ("wandb" if self.wandb_api_key else "log"),
                }
            )

        if path == "/api/runs":
            return self._handle_runs()

        if path == "/api/snapshots":
            return self._json(_list_snapshot_runs())

        m = re.match(r"^/api/snapshots/jobs/([^/]+)$", path)
        if m:
            with _ingest_lock:
                job = _ingest_jobs.get(m.group(1))
            return self._json(job or {"status": "unknown"}, 200 if job else 404)

        m = re.match(r"^/api/runs/([^/]+)/metrics$", path)
        if m:
            return self._handle_metrics(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/latest$", path)
        if m:
            return self._handle_latest(m.group(1))

        m = re.match(r"^/api/runs/([^/]+)/logs$", path)
        if m:
            return self._handle_logs(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/keys$", path)
        if m:
            return self._handle_keys(m.group(1))

        m = re.match(r"^/api/runs/([^/]+)/config$", path)
        if m:
            return self._handle_config(m.group(1))

        if path == "/api/analysis/prompt":
            return self._json({"template": _load_prompt_template()})

        if path == "/api/analysis/demo-reports":
            return self._handle_demo_reports()

        m = re.match(r"^/api/runs/([^/]+)/analysis-context$", path)
        if m:
            return self._handle_analysis_context(m.group(1))

        m = re.match(r"^/api/runs/([^/]+)/trials$", path)
        if m:
            return self._handle_trials(m.group(1))

        m = re.match(r"^/api/runs/([^/]+)/trajectory$", path)
        if m:
            return self._handle_trajectory(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/step-tasks$", path)
        if m:
            return self._handle_step_tasks(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/step-export$", path)
        if m:
            return self._handle_step_export(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/rollout-dist$", path)
        if m:
            return self._handle_rollout_dist(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/rollout-dist-series$", path)
        if m:
            return self._handle_rollout_dist_series(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/task-traj-stats$", path)
        if m:
            return self._handle_task_traj_stats(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/task-stats$", path)
        if m:
            return self._handle_task_stats(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/val-cot$", path)
        if m:
            return self._handle_val_cot(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/val-analysis$", path)
        if m:
            return self._handle_val_analysis(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/val-failure-modes$", path)
        if m:
            return self._handle_val_failure_modes(m.group(1), qs)

        m = re.match(r"^/api/runs/([^/]+)/token-stats$", path)
        if m:
            return self._handle_token_stats(m.group(1))

        # static files
        if self.static_dir:
            self._serve_static(path)
        else:
            self.send_error(404)

    def _json(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _resolve_run_path(self, run_id: str) -> str | None:
        # Referenced run dirs may carry an optional training-console log path in
        # their meta; that drives the metrics/logs/keys endpoints (training
        # curves live in the log, not in the trajectory dir).
        if run_id.startswith(SNAP_PREFIX):
            meta = _load_snapshot_meta(_snap_slug(run_id)) or {}
            lp = meta.get("log_path")
            return lp if lp and os.path.exists(lp) else None
        cached = self._run_path_cache.get(run_id)
        if cached and os.path.exists(cached):
            return cached
        for d in self.log_dirs:
            if not d:
                continue
            for ext in LOG_EXTENSIONS:
                p = os.path.join(d, f"{run_id}{ext}")
                if os.path.exists(p):
                    self._run_path_cache[run_id] = p
                    return p
        return None

    def _handle_runs(self):
        runs: list[dict] = []
        runs.extend(_list_snapshot_runs())
        # Only surface log runs that actually trained: keep currently-running
        # runs (regardless of step count) plus any run that reached > 5 training
        # steps. Everything else (crashed-early / smoke / never-trained jobs) is
        # hidden from the panel. Direct /api/runs/<id>/* access still works —
        # discover_runs() is unfiltered, only this list view is trimmed.
        # A run mid-`update_actor` (can exceed an hour at long context) stops
        # writing its log, so the 120s "running" heuristic misses it and it would
        # be hidden behind the >5-step gate. Keep any run whose log was touched
        # within a step's worth of time so genuinely-live runs are never gated.
        _now = time.time()

        def _recently_active(r: dict) -> bool:
            try:
                return (_now - os.path.getmtime(r["path"])) < 3 * 3600
            except OSError:
                return False

        log_runs = [
            r for r in discover_runs(self.log_dirs)
            if r.get("state") == "running"
            or r.get("steps", 0) > 5
            or _recently_active(r)
        ]
        # Mark runs whose analysis panels (Rollout/Task Grid/Trajectory/
        # Validation) would be empty because their on-disk trial dirs are gone
        # (old runs whose harbor_trials were cleaned up). The training curves
        # still load from the log, so we don't drop them — we tag them so the UI
        # can hide them by default behind a "show empty runs" toggle. Running
        # runs are never tagged (they may not have dumped trials yet).
        now = time.time()
        for r in log_runs:
            # Never hide a run that is running, or whose log was touched in the
            # last 12h — a still-active / just-finished run may not have dumped
            # its trial dirs yet, and the user explicitly wants live tasks shown.
            if r.get("state") == "running":
                r["analysis_empty"] = False
                continue
            try:
                lp = self._resolve_run_path(r["id"])
                if lp and (now - os.path.getmtime(lp)) < 12 * 3600:
                    r["analysis_empty"] = False
                    continue
                r["analysis_empty"] = self._run_analysis_empty(r["id"])
            except Exception:
                r["analysis_empty"] = False
        runs.extend(log_runs)
        if self.wandb_api_key and self.wandb_entity and self.wandb_project:
            try:
                runs.extend(
                    wandb_runs(
                        self.wandb_entity, self.wandb_project, self.wandb_api_key
                    )
                )
            except Exception:
                pass
        self._json(runs)

    def _handle_metrics(self, run_id: str, qs: dict):
        cache_key = f"metrics:{run_id}"
        cached = _cache.get(cache_key)
        if cached:
            metrics = cached
        else:
            metrics = self._load_metrics(run_id)
            _cache.put(cache_key, metrics)

        if not metrics:
            return self._json({"error": "run not found"}, 404)

        all_keys = set()
        for p in metrics:
            all_keys.update(k for k in p if k != "step")

        key_filter = qs.get("keys", [None])[0]
        if key_filter:
            wanted = set(key_filter.split(","))
            metrics = [
                {k: v for k, v in p.items() if k == "step" or k in wanted}
                for p in metrics
            ]

        runs = discover_runs(self.log_dirs)
        run_info = next((r for r in runs if r["id"] == run_id), None)
        if not run_info:
            run_info = {
                "id": run_id,
                "name": run_id,
                "state": "unknown",
                "created_at": "",
                "source": "log",
            }
        run_info.pop("path", None)
        run_info.pop("size", None)

        self._json(
            {
                "run": run_info,
                "metrics": metrics,
                "available_keys": sorted(all_keys),
            }
        )

    def _handle_latest(self, run_id: str):
        metrics = self._load_metrics(run_id)
        if metrics:
            self._json(metrics[-1])
        else:
            self._json(None)

    def _handle_logs(self, run_id: str, qs: dict):
        path = self._resolve_run_path(run_id)
        if not path:
            return self._json({"error": "log file not found"}, 404)
        try:
            tail = int(qs.get("tail", ["500"])[0])
            offset = int(qs.get("offset", ["0"])[0])
        except (ValueError, IndexError):
            tail = 500
            offset = 0
        tail = min(tail, 2000)
        with open(path, "r", errors="replace") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        if offset == 0:
            # initial load: return last `tail` lines
            start = max(0, total - tail)
            lines = [l.rstrip("\n") for l in all_lines[start:]]
            self._json({"lines": lines, "total_lines": total, "offset": start})
        else:
            # incremental: return lines from offset onward
            lines = [l.rstrip("\n") for l in all_lines[offset:]]
            self._json({"lines": lines, "total_lines": total, "offset": offset})

    def _handle_keys(self, run_id: str):
        metrics = self._load_metrics(run_id)
        all_keys: set[str] = set()
        for p in metrics:
            all_keys.update(k for k in p if k != "step")
        self._json(sorted(all_keys))

    def _handle_config(self, run_id: str):
        cache_key = f"config:{run_id}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return self._json(cached)
        log_path = self._resolve_run_path(run_id)
        if not log_path:
            return self._json({"available": False, "error": "No log found for this run"}, 404)
        payload = _run_config(log_path)
        _cache.put(cache_key, payload)
        self._json(payload)

    def _handle_demo_reports(self):
        demo_dir = os.path.join(os.path.dirname(__file__), "demo_reports")
        reports = []
        if os.path.isdir(demo_dir):
            for fname in sorted(os.listdir(demo_dir)):
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(demo_dir, fname)
                with open(fpath, "r", errors="replace") as f:
                    content = f.read()
                run_id = fname.removesuffix(".md")
                reports.append({
                    "id": f"demo-{run_id}",
                    "runId": run_id,
                    "model": "demo",
                    "createdAt": datetime.fromtimestamp(
                        os.path.getmtime(fpath), tz=timezone.utc
                    ).isoformat(),
                    "content": content,
                })
        self._json(reports)

    def _handle_trials(self, run_id: str):
        exp_dirs = self._exp_dirs_for(run_id)
        steps = _list_steps_multi(exp_dirs)
        self._json(
            {
                "run_id": run_id,
                "steps": steps,
                "total": len(steps),
                "num_exp_dirs": len(exp_dirs),
                # step_NNNN is a global rollout/sample index, NOT the training
                # step (async). Surfaced so the UI can label it accordingly.
                "step_is_training_step": False,
            }
        )

    def _handle_trajectory(self, run_id: str, qs: dict):
        step = qs.get("step", [None])[0]
        task = qs.get("task", [None])[0]
        exp_dirs = self._exp_dirs_for(run_id)
        trial = _pick_trial_multi(exp_dirs, step, task)
        if not trial:
            return self._json({"error": "no trials found"}, 404)

        system_prompt, problem_statement, turns = _load_trajectory(trial["task_dir"])
        self._json({
            "run_id": run_id,
            "step": trial["step"],
            "task": trial["task"],
            "reward": trial["reward"],
            "num_turns": len(turns),
            "system_prompt": system_prompt,
            "problem_statement": problem_statement,
            "turns": turns,
        })

    def _handle_step_tasks(self, run_id: str, qs: dict):
        step = qs.get("step", [None])[0]
        if not step:
            return self._json({"error": "step param required"}, 400)
        exp_dirs = self._exp_dirs_for(run_id)
        tasks = _list_tasks_in_step_multi(exp_dirs, step)
        # drop internal exp_dir field from the response
        tasks = [{"task": t["task"], "reward": t["reward"]} for t in tasks]
        self._json({"run_id": run_id, "step": step, "tasks": tasks, "total": len(tasks)})

    def _handle_step_export(self, run_id: str, qs: dict):
        step = qs.get("step", [None])[0]
        if not step:
            return self._json({"error": "step param required"}, 400)
        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        tasks = _list_tasks_in_step_multi(exp_dirs, step)
        all_trajs: list[dict] = []
        for ti in tasks:
            task_dir = os.path.join(ti["exp_dir"], step, ti["task"])
            sys_p, prob, turns = _load_trajectory(task_dir)
            all_trajs.append({
                "task": ti["task"],
                "reward": ti["reward"],
                "num_turns": len(turns),
                "system_prompt": sys_p,
                "problem_statement": prob,
                "turns": turns,
            })
        self._json({
            "run_id": run_id, "step": step,
            "total": len(all_trajs), "trajectories": all_trajs,
        })

    def _handle_rollout_dist(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))
        step = qs.get("step", [None])[0]
        try:
            max_steps = int(qs.get("max_steps", ["2000"])[0])
        except (TypeError, ValueError):
            max_steps = 2000
        max_steps = max(1, min(max_steps, 5000))

        cache_key = f"{run_id}|n={expected_n}|step={step}|m={max_steps}"
        cached = self._rollout_dist_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < 120:
            return self._json(cached[1])

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        result = _rollout_distribution(exp_dirs, expected_n, step, max_steps)
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        self._rollout_dist_cache[cache_key] = (time.time(), result)
        self._json(result)

    def _handle_val_failure_modes(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))
        step = qs.get("step", [None])[0]

        # A multi-node async val dump GROWS for hours (siblings write progressively),
        # so it is NOT immutable — short in-memory TTL so a snapshot cached mid-val
        # self-heals quickly. The disk cache in _val_failure_modes has its own
        # grow-aware guard (recomputes when the on-disk task set exceeds the cache).
        cache_key = f"valfail|{run_id}|n={expected_n}|s={step}"
        cached = self._rollout_dist_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < 600:
            return self._json(cached[1])

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        result = _val_failure_modes(exp_dirs, run_id, expected_n, step)
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        self._rollout_dist_cache[cache_key] = (time.time(), result)
        self._json(result)

    def _handle_rollout_dist_series(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))
        try:
            buckets = int(qs.get("buckets", ["24"])[0])
        except (TypeError, ValueError):
            buckets = 24
        buckets = max(1, min(buckets, 100))
        # bucket_size > 0 => each bar holds ~bucket_size groups (set to
        # train_batch_size so one bar ≈ one training step); overrides `buckets`.
        try:
            bucket_size = int(qs.get("bucket_size", ["0"])[0])
        except (TypeError, ValueError):
            bucket_size = 0
        bucket_size = max(0, min(bucket_size, 100000))

        cache_key = f"series|{run_id}|n={expected_n}|b={buckets}|bs={bucket_size}"
        cached = self._rollout_dist_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < 120:
            return self._json(cached[1])

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        result = _rollout_dist_series(
            exp_dirs, expected_n, buckets, bucket_size=bucket_size
        )
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        self._rollout_dist_cache[cache_key] = (time.time(), result)
        self._json(result)

    def _handle_val_cot(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))
        try:
            sample = int(qs.get("sample", ["120"])[0])
        except (TypeError, ValueError):
            sample = 120
        sample = max(10, min(sample, 500))

        # Two JSON reads per (event, cohort task) on a networked FS. Same TTL
        # shape as val-analysis: the answer only moves when a new val event
        # lands, which is every test_freq steps, i.e. hours.
        cache_key = f"valcot|{run_id}|n={expected_n}|s={sample}"
        cached = self._rollout_dist_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < 1800:
            return self._json(cached[1])

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        train_steps = _val_step_train_map(self._resolve_run_path(run_id))
        result = _val_cot_trend(exp_dirs, expected_n, train_steps, sample)
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        self._rollout_dist_cache[cache_key] = (time.time(), result)
        self._json(result)

    def _handle_val_analysis(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))

        cache_key = f"val|{run_id}|n={expected_n}"
        cached = self._rollout_dist_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < 1800:
            return self._json(cached[1])

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        train_steps = _val_step_train_map(self._resolve_run_path(run_id))
        result = _val_analysis(exp_dirs, expected_n, train_steps)
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        self._rollout_dist_cache[cache_key] = (time.time(), result)
        self._json(result)

    def _handle_task_stats(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))
        # default: scan the whole run (epoch comparison needs full history)
        try:
            max_steps = int(qs.get("max_steps", ["100000"])[0])
        except (TypeError, ValueError):
            max_steps = 100000
        max_steps = max(1, min(max_steps, 100000))

        # A cold compute walks every step dir of the whole run and takes minutes
        # on a networked FS. A 120s TTL was self-defeating — it expired before it
        # could ever be hit, so almost every open recomputed, and anything served
        # through a Cloudflare quick tunnel died on its ~100s limit with an HTML
        # error page (which the client then tried to JSON.parse). Same two-layer
        # shape as task-traj-stats: the answer only drifts as tasks pick up new
        # epochs, which is slow.
        TTL = 1800
        DISK_TTL = 6 * 3600
        cache_key = f"tasks|{run_id}|n={expected_n}|m={max_steps}"
        cached = self._rollout_dist_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < TTL:
            return self._json(cached[1])
        dk = f"taskstats_n{expected_n}_m{max_steps}"
        disk = _valfail_disk_get(run_id, dk)
        if disk and (time.time() - (disk.get("computed_at") or 0)) < DISK_TTL:
            self._rollout_dist_cache[cache_key] = (time.time(), disk)
            return self._json(disk)

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)
        result = _task_solve_stats(exp_dirs, expected_n, max_steps)
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        result["computed_at"] = time.time()
        self._rollout_dist_cache[cache_key] = (time.time(), result)
        _valfail_disk_put(run_id, dk, result)
        self._json(result)

    def _handle_task_traj_stats(self, run_id: str, qs: dict):
        try:
            expected_n = int(qs.get("n", ["8"])[0])
        except (TypeError, ValueError):
            expected_n = 8
        expected_n = max(1, min(expected_n, 64))

        exp_dirs = self._exp_dirs_for(run_id)
        if not exp_dirs:
            return self._json({"error": "run not found"}, 404)

        # Cold compute opens two JSON files per rollout across the first and last
        # epoch of every multi-epoch task — ~12k reads on a networked FS, ~60s.
        # Do NOT key this on the newest step dir: a live run mints a new rollout
        # dir every few seconds, so that key misses on essentially every request
        # and re-pays the full 60s. The answer only drifts as tasks pick up new
        # epochs, which is slow, so a plain TTL is the right invalidation.
        TTL = 1800
        DISK_TTL = 6 * 3600
        ck = f"trajstats|{run_id}|n={expected_n}"
        mem = self._rollout_dist_cache.get(ck)
        if mem and (time.time() - mem[0]) < TTL:
            return self._json(mem[1])
        dk = f"trajstats_n{expected_n}"
        disk = _valfail_disk_get(run_id, dk)
        if disk and (time.time() - (disk.get("computed_at") or 0)) < DISK_TTL:
            self._rollout_dist_cache[ck] = (time.time(), disk)
            return self._json(disk)

        result = _task_traj_stats(exp_dirs, expected_n)
        result["run_id"] = run_id
        result["num_exp_dirs"] = len(exp_dirs)
        result["computed_at"] = time.time()
        self._rollout_dist_cache[ck] = (time.time(), result)
        _valfail_disk_put(run_id, dk, result)
        self._json(result)

    def _handle_analysis_context(self, run_id: str):
        metrics = self._load_metrics(run_id)
        if not metrics:
            return self._json({"error": "run not found"}, 404)
        prompt = _build_analysis_prompt(run_id, metrics, self.log_dirs)
        self._json({"run_id": run_id, "prompt": prompt, "num_steps": len(metrics)})

    def _handle_token_stats(self, run_id: str):
        if not run_id.startswith(SNAP_PREFIX):
            return self._json({"error": "token stats only available for snapshots"}, 404)
        ts = os.path.join(_snap_dir(run_id), "token_stats.json")
        if not os.path.isfile(ts):
            return self._json({"error": "no token stats for this snapshot"}, 404)
        with open(ts) as f:
            self._json(json.load(f))

    def _handle_snapshot_ingest(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        bundle_path = (body.get("path") or "").strip()
        name = (body.get("name") or "").strip()
        # Optional training-console log for a referenced run dir — enables the
        # training-curve (metrics) panels alongside the trajectory panels.
        log_path = (body.get("log_path") or "").strip()
        if not bundle_path:
            return self._json({"error": "path is required"}, 400)
        if not os.path.exists(bundle_path):
            return self._json({"error": f"path not found: {bundle_path}"}, 400)
        if not (zipfile.is_zipfile(bundle_path) or os.path.isdir(bundle_path)):
            return self._json(
                {"error": "path must be a .zip or a directory of the 3 files"}, 400
            )
        if log_path and not os.path.isfile(log_path):
            return self._json({"error": f"log_path not found: {log_path}"}, 400)
        if not name:
            name = os.path.splitext(os.path.basename(bundle_path.rstrip("/")))[0]
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        with _ingest_lock:
            _ingest_jobs[job_id] = {
                "status": "queued",
                "name": name,
                "path": bundle_path,
                "done": 0,
                "total": 0,
            }
        threading.Thread(
            target=_run_ingest_job,
            args=(job_id, bundle_path, name, log_path or None),
            daemon=True,
        ).start()
        self._json({"job_id": job_id})

    def _handle_snapshot_delete(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        run_id = (body.get("run_id") or "").strip()
        if not run_id.startswith(SNAP_PREFIX):
            return self._json({"error": "invalid snapshot run_id"}, 400)
        d = _snap_dir(run_id)
        # guard: must live under SNAPSHOTS_DIR
        if os.path.commonpath([os.path.abspath(d), SNAPSHOTS_DIR]) != SNAPSHOTS_DIR:
            return self._json({"error": "invalid path"}, 400)
        if not os.path.isdir(d):
            return self._json({"error": "snapshot not found"}, 404)
        shutil.rmtree(d, ignore_errors=True)
        self._json({"ok": True})

    def _handle_analysis_generate(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        run_id = body.get("run_id", "")
        api_key = body.get("api_key", "")
        base_url = body.get("base_url", "https://api.openai.com/v1")
        model = body.get("model", "gpt-4o")
        custom_prompt = body.get("custom_prompt", "")

        if not run_id:
            return self._json({"error": "run_id is required"}, 400)
        if not api_key:
            return self._json({"error": "api_key is required"}, 400)

        metrics = self._load_metrics(run_id)
        if not metrics:
            return self._json({"error": "run not found"}, 404)

        evidence = self._analysis_evidence(run_id, metrics)
        prompt = _build_analysis_prompt(
            run_id, metrics, self.log_dirs, custom_prompt, evidence
        )

        try:
            report = call_llm_api(api_key, base_url, model, prompt)
            self._json({"report": report, "run_id": run_id, "model": model})
        except Exception as e:
            self._json({"error": f"LLM API call failed: {e}"}, 502)

    def _analysis_evidence(self, run_id: str, metrics: list[dict]) -> dict:
        """Trial-dir evidence for the diagnostic report.

        Cost split, deliberately: the group scan (reward files only, ~20s) runs
        INLINE because zero-advantage accounting is the single most load-bearing
        fact in the report and its 120s mem cache is almost never warm by the
        time someone clicks Generate. The trajectory-SHAPE scan opens two JSON
        files per rollout (~12k reads, ~60s) so it is taken CACHE-ONLY — its own
        handler keeps a 30min mem + 6h disk cache, so opening the Task Grid panel
        once makes it available here, and its absence degrades one section
        instead of stalling the whole request.
        """
        ev: dict = {}
        exp_dirs = self._exp_dirs_for(run_id)

        expected_n = 8
        try:
            path = self._resolve_run_path(run_id)
            if path:
                n = _cfg_get(_extract_config_dict(path) or {}, "actor_rollout_ref.rollout.n")
                if isinstance(n, (int, float)) and 1 <= int(n) <= 64:
                    expected_n = int(n)
        except Exception:
            pass
        ev["expected_n"] = expected_n

        # Nominal prompts per step = tagged rollouts per step / n. Read off the
        # last step that logged termination reasons rather than the config, so a
        # resumed or retopologized run reports what it actually ran.
        for m in reversed([x for x in metrics if len(x) > 10]):
            tot = sum(
                float(v)
                for k, v in m.items()
                if k.startswith("trajectory_filter/reason/")
                and isinstance(v, (int, float))
                and v == v
            )
            if tot > 0:
                ev["batch_prompts"] = tot / expected_n
                break

        if not exp_dirs:
            return ev

        try:
            ev["dist"] = _rollout_distribution(exp_dirs, expected_n, None, 2000)
        except Exception:
            pass
        try:
            ev["task_solve"] = _task_solve_stats(exp_dirs, expected_n, 100000)
        except Exception:
            pass

        # cache-only (see docstring)
        ck = f"trajstats|{run_id}|n={expected_n}"
        mem = self._rollout_dist_cache.get(ck)
        if mem and (time.time() - mem[0]) < 1800:
            ev["traj_shape"] = mem[1]
        else:
            disk = _valfail_disk_get(run_id, f"trajstats_n{expected_n}")
            if disk and (time.time() - (disk.get("computed_at") or 0)) < 6 * 3600:
                ev["traj_shape"] = disk
        return ev

    def _load_metrics(self, run_id: str) -> list[dict]:
        if run_id.startswith(SNAP_PREFIX):
            meta = _load_snapshot_meta(_snap_slug(run_id)) or {}
            # run_dir with an optional training log -> parse curves from it;
            # ingested bundles carry a precomputed "metrics" array in meta.
            lp = meta.get("log_path")
            if lp and os.path.exists(lp):
                return parse_log_file(lp)
            return meta.get("metrics", [])
        path = self._resolve_run_path(run_id)
        if path:
            return parse_log_file(path)
        if self.wandb_api_key and self.wandb_entity and self.wandb_project:
            try:
                return wandb_history(
                    self.wandb_entity,
                    self.wandb_project,
                    self.wandb_api_key,
                    run_id,
                )
            except Exception:
                pass
        return []

    def _serve_static(self, path: str):
        if path == "/":
            path = "/index.html"
        fpath = os.path.join(self.static_dir, path.lstrip("/"))
        if not os.path.isfile(fpath):
            fpath = os.path.join(self.static_dir, "index.html")
        try:
            with open(fpath, "rb") as f:
                data = f.read()
            ext = os.path.splitext(fpath)[1]
            ct = {
                ".html": "text/html",
                ".js": "application/javascript",
                ".css": "text/css",
                ".json": "application/json",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".ico": "image/x-icon",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence per-request logs


def _prewarm_loop(port: int, host: str, n_runs: int, interval: int) -> None:
    """Keep the expensive endpoints warm so nobody has to wait for a cold one.

    Caching alone only moves the cost: whoever arrives after an entry expires
    still pays it, and for task-stats that is ~450s of walking the trials tree —
    long enough that a Cloudflare quick tunnel (~100s) gives up and hands the
    client an HTML error page instead of JSON. Recomputing on a timer means the
    entry is refreshed before it lapses and every real request is a cache hit.

    Driven through the loopback HTTP interface on purpose: the handlers already
    own the two-layer (memory + disk) caching, and going through them keeps
    exactly one code path warm rather than a parallel one that might diverge.

    Only the newest `n_runs` are swept — a full pass over every historical run
    would not finish inside one interval, and nobody is watching those. Each
    sweep is one request at a time; this shares a networked filesystem with
    training, so it must not become a second load generator.
    """
    import urllib.parse

    base = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"

    def _get(path: str, timeout: int):
        try:
            t0 = time.time()
            with urlopen(f"{base}{path}", timeout=timeout) as r:
                r.read()
            return time.time() - t0
        except Exception:
            return None

    time.sleep(20)  # let the port bind and the first run listing settle
    while True:
        try:
            dt = _get("/api/runs", 900)
            if dt is not None:
                runs = json.loads(urlopen(f"{base}/api/runs", timeout=900).read())
                for r in runs[: max(0, n_runs)]:
                    rid = r.get("id")
                    if not rid:
                        continue
                    q = urllib.parse.quote(str(rid), safe="")
                    for path, budget in (
                        (f"/api/runs/{q}/task-stats?n=8", 1800),
                        (f"/api/runs/{q}/val-cot?sample=120", 900),
                        (f"/api/runs/{q}/task-traj-stats?n=8", 900),
                    ):
                        _get(path, budget)
        except Exception:
            pass  # a prewarm failure must never take the server down
        time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description="RL Training Dashboard Server")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--log-dir", default="", help="Primary log directory")
    p.add_argument(
        "--extra-log-dir",
        action="append",
        default=[],
        help="Additional directories to scan for .log/.out files (repeatable)",
    )
    p.add_argument("--static-dir", default="")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-api-key", default="")
    p.add_argument(
        "--prewarm-runs",
        type=int,
        default=2,
        help="Recompute the slow endpoints for the N newest runs in the "
        "background so they are never served cold. 0 disables.",
    )
    p.add_argument(
        "--prewarm-interval",
        type=int,
        default=1800,
        help="Seconds between prewarm sweeps (default 1800).",
    )
    args = p.parse_args()

    log_dir = args.log_dir
    if not log_dir:
        candidate = os.path.join(os.path.dirname(__file__), "..", "logs")
        if os.path.isdir(candidate):
            log_dir = os.path.abspath(candidate)
    static_dir = args.static_dir
    if not static_dir:
        candidate = os.path.join(os.path.dirname(__file__), "dist")
        if os.path.isdir(candidate):
            static_dir = os.path.abspath(candidate)

    log_dirs = [log_dir] if log_dir else []
    for extra in args.extra_log_dir:
        d = os.path.abspath(extra)
        if os.path.isdir(d) and d not in log_dirs:
            log_dirs.append(d)

    global TRIALS_DIRS
    TRIALS_DIRS = _discover_trials_dirs()
    for extra in args.extra_log_dir:
        candidate = os.path.join(os.path.dirname(extra), "harbor_trials")
        if os.path.isdir(candidate) and candidate not in TRIALS_DIRS:
            TRIALS_DIRS.append(os.path.abspath(candidate))

    DashboardHandler.log_dirs = log_dirs
    DashboardHandler.static_dir = static_dir
    DashboardHandler.wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY", "")
    DashboardHandler.wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "swe-lego-live-rl")
    DashboardHandler.wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY", "")

    import socket
    # ThreadingHTTPServer so a slow analytics request (e.g. rollout-dist
    # scanning the networked trials FS) doesn't block the rest of the dashboard.
    class ReusableHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True
        daemon_threads = True
    # Line-buffer stdout: this normally runs under nohup with stdout redirected
    # to a file, where block buffering means the banner below never reaches the
    # log until the process exits — exactly when you need it to diagnose a start.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    srv = ReusableHTTPServer((args.host, args.port), DashboardHandler)
    if args.prewarm_runs > 0:
        threading.Thread(
            target=_prewarm_loop,
            args=(args.port, args.host, args.prewarm_runs, args.prewarm_interval),
            daemon=True,
        ).start()
    print(f"Dashboard server on http://{args.host}:{args.port}")
    print(f"  Log dirs:    {log_dirs or '(none)'}")
    print(f"  Static dir:  {static_dir or '(none — API only, use vite dev for frontend)'}")
    print(f"  Trials dirs: {TRIALS_DIRS or '(none)'}")
    print(f"  wandb:       {'enabled' if DashboardHandler.wandb_api_key else 'disabled'}")
    print(
        f"  prewarm:     {f'{args.prewarm_runs} newest run(s) every {args.prewarm_interval}s' if args.prewarm_runs > 0 else 'disabled'}"
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
