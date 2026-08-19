#!/usr/bin/env python3
"""Export one run as a self-contained static dashboard.

Walks the read-only GET endpoints of a *running* server.py, writes each response
to ``<out>/data/<slug>.json``, and drops a manifest next to them. The built
frontend picks the bundle up through ``src/staticMode.ts`` and serves the run
with no backend at all, so the result can go on any static host.

Driving the real server rather than importing its internals is deliberate: the
panels get byte-identical payloads to what they see locally, including every
derived statistic the handlers compute.

Usage:
    # 1. serve the run locally
    python webui/server.py --port 8123 --log-dir <logs> --static-dir webui/dist

    # 2. build the frontend, then export into that same dist/
    npx vite build
    python webui/export_static.py --run <run-id> --out webui/dist

    # 3. dist/ is now standalone
    npx serve webui/dist
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

# Endpoints every panel needs, with the query strings the UI actually sends.
# Keep these in sync with the fetch() call sites -- a slug the frontend asks for
# but we never exported degrades to an empty panel, not an error page.
RUN_ENDPOINTS: list[str] = [
    "metrics",
    "latest",
    "keys",
    "config",
    "trials",
    "token-stats",
    "val-analysis",
    "val-failure-modes",
    "analysis-context",
    "task-stats?n=8",
    "task-traj-stats?n=8",
    "rollout-dist?n=8",
    "rollout-dist-series?n=8&bucket_size=64",
    "val-cot?sample=120",
    "logs?tail=500&offset=0",
]

GLOBAL_ENDPOINTS: list[str] = [
    "config",
    "runs",
    "analysis/demo-reports",
]


def _slug_hash(s: str) -> str:
    """FNV-1a, 32-bit. Kept byte-identical to `fnv1a` in src/staticMode.ts."""
    h = 0x811C9DC5
    for ch in s.encode("utf-8", "surrogatepass"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def api_slug(path_with_query: str) -> str:
    """`/api/runs/foo/metrics?keys=b,a` -> `runs_foo_metrics__<hash>`."""
    raw_path, _, raw_query = path_with_query.partition("?")
    path = raw_path.removeprefix("/api/").rstrip("/")
    params = sorted(urllib.parse.parse_qsl(raw_query, keep_blank_values=True))
    canonical = path + ("?" + "&".join(f"{k}={v}" for k, v in params) if params else "")
    readable = "".join(c if (c.isalnum() or c in "._-") else "_" for c in path)[:80]
    return f"{readable}__{_slug_hash(canonical)}"


def fetch(base: str, path_with_query: str, timeout: int) -> tuple[bytes | None, str]:
    url = base.rstrip("/") + path_with_query
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read(), "ok"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - the reason is only ever printed
        return None, type(e).__name__


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id as /api/runs reports it")
    ap.add_argument("--out", required=True, help="built dist/ directory to export into")
    ap.add_argument("--server", default="http://127.0.0.1:8123", help="running server.py")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-request seconds; the whole-run scans are slow from cold")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="rollout-batch steps to include, sampled evenly across the run. "
                         "A long run has thousands; exporting all of them is mostly dead weight")
    ap.add_argument("--max-trajectories", type=int, default=40,
                    help="trials to include in the Trajectory panel (0 = none)")
    args = ap.parse_args()

    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)

    exported: list[str] = []
    skipped: list[tuple[str, str]] = []

    def grab(path_with_query: str) -> bytes | None:
        body, status = fetch(args.server, path_with_query, args.timeout)
        slug = api_slug(path_with_query)
        if body is None:
            skipped.append((path_with_query, status))
            print(f"  skip  {path_with_query}  ({status})", flush=True)
            return None
        with open(os.path.join(data_dir, f"{slug}.json"), "wb") as f:
            f.write(body)
        exported.append(slug)
        print(f"  ok    {path_with_query}  -> {slug}.json  ({len(body):,} B)", flush=True)
        return body

    run = urllib.parse.quote(args.run, safe="")

    print(f"[export] server={args.server} run={args.run}")
    for ep in GLOBAL_ENDPOINTS:
        grab(f"/api/{ep}")

    # /api/runs must list only the exported run: the sidebar is built from it,
    # and every other entry would be a dead link in the bundle.
    runs_path = os.path.join(data_dir, f"{api_slug('/api/runs')}.json")
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            all_runs = json.load(f)
        kept = [r for r in all_runs if r.get("id") == args.run]
        if not kept:
            print(f"[export] FATAL: run id {args.run!r} is not in /api/runs", file=sys.stderr)
            return 1
        with open(runs_path, "w") as f:
            json.dump(kept, f)
        print(f"[export] /api/runs trimmed to 1 of {len(all_runs)} runs")

    for ep in RUN_ENDPOINTS:
        path, _, query = ep.partition("?")
        grab(f"/api/runs/{run}/{path}" + (f"?{query}" if query else ""))

    # Per-step and per-trial endpoints. `trials` gives the step list the
    # Trajectory panel populates its selectors from. A production run has
    # thousands of rollout-batch steps, so sample evenly and rewrite the
    # exported step list to match -- otherwise the selector offers thousands of
    # steps whose step-tasks were never exported and which render empty.
    trials_path = os.path.join(data_dir, f"{api_slug(f'/api/runs/{run}/trials')}.json")
    steps: list = []
    if os.path.exists(trials_path):
        with open(trials_path) as f:
            trials = json.load(f)
        steps = trials.get("steps", []) if isinstance(trials, dict) else []

    def step_id(s):
        return s.get("step") if isinstance(s, dict) else s

    if args.max_steps > 0 and len(steps) > args.max_steps:
        stride = len(steps) / args.max_steps
        sampled = [steps[min(int(i * stride), len(steps) - 1)] for i in range(args.max_steps)]
        # Always keep the newest: it is what the panel opens on.
        if step_id(sampled[-1]) != step_id(steps[-1]):
            sampled[-1] = steps[-1]
        print(f"[export] {len(steps)} steps -> sampling {len(sampled)} (every ~{stride:.0f})")
        steps = sampled
        if os.path.exists(trials_path):
            trials["sampled_from_total"] = trials.get("total")
            trials["steps"] = steps
            trials["total"] = len(steps)
            with open(trials_path, "w") as f:
                json.dump(trials, f)
    else:
        print(f"[export] {len(steps)} steps with trials")

    budget = args.max_trajectories
    # Spread trajectories across the sampled steps instead of draining the first
    # one, so the panel shows behaviour from early and late in the run.
    per_step = max(1, budget // max(1, len(steps))) if steps else 0
    for step in steps:
        sid = step_id(step)
        if sid is None:
            continue
        body = grab(f"/api/runs/{run}/step-tasks?step={urllib.parse.quote(str(sid), safe='')}")
        if body is None or budget <= 0:
            continue
        try:
            tasks = json.loads(body)
            names = tasks.get("tasks", []) if isinstance(tasks, dict) else []
        except json.JSONDecodeError:
            continue
        # Selecting a step with the task dropdown left on "All Tasks (random)"
        # -- which is how the panel opens -- requests this exact URL. Without it
        # every step but the pre-picked ones renders "No trajectory data found".
        # The server picks a trial at random, so the bundle freezes one per step
        # and the Random Trial button becomes a no-op here.
        grab(f"/api/runs/{run}/trajectory?step={urllib.parse.quote(str(sid), safe='')}")

        taken = 0
        for t in names:
            if budget <= 0 or taken >= per_step:
                break
            tid = t.get("task") if isinstance(t, dict) else t
            if not tid:
                continue
            q = urllib.parse.urlencode({"step": str(sid), "task": str(tid)})
            if grab(f"/api/runs/{run}/trajectory?{q}") is not None:
                budget -= 1
                taken += 1

    # No step selected at all: the very first request the panel makes.
    grab(f"/api/runs/{run}/trajectory?")

    manifest = {"run_id": args.run, "files": sorted(set(exported))}
    with open(os.path.join(data_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(
        os.path.getsize(os.path.join(data_dir, n)) for n in os.listdir(data_dir)
    )
    print(f"\n[export] {len(manifest['files'])} files, {total / 1e6:.1f} MB -> {data_dir}")
    if skipped:
        print(f"[export] {len(skipped)} endpoint(s) not exported:")
        for p, why in skipped:
            print(f"           {why:<10} {p}")
        print("         Those panels will render their empty state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
